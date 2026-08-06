// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include "strand/orientation_cache.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>

#ifdef _WIN32
#include <io.h>
#include <process.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "common/image_io.h"
#include "common/logger.h"
#include "mvs/gabor_orientation.h"
#include "mvs/image_loader.h"

namespace orientation_cache {
namespace detail {

// Implemented in orientation_cache.cu.  Keeping CUDA half types out of the
// public header makes PreparedView a stable host-side record.
bool PackDeviceMapsToHost(const float* d_theta, size_t theta_pitch_bytes,
                          const float* d_variance,
                          size_t variance_pitch_bytes, int width, int height,
                          uint32_t* packed, std::string* error);

}  // namespace detail
namespace {

namespace fs = std::filesystem;

constexpr char kMagic[8] = {'H', 'G', 'R', 'O', 'W', '2', 'D', '\0'};
constexpr uint32_t kFormatVersion = 1;
constexpr uint32_t kEndianMarker = 0x01020304u;
constexpr uint32_t kFlagBoundary = 1u;
constexpr uint32_t kPackedPixelBytes = sizeof(uint32_t);
constexpr uint64_t kFnvOffset = 14695981039346656037ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

#pragma pack(push, 1)
struct CacheHeader {
  char magic[8];
  uint32_t version;
  uint32_t header_size;
  uint32_t endian_marker;
  uint32_t flags;
  uint32_t camera_id;
  int32_t camera_index;
  uint32_t width;
  uint32_t height;
  uint32_t packed_pixel_bytes;
  uint32_t reserved;
  uint64_t pixel_count;
  uint64_t fingerprint;
  uint64_t orientation_hash;
  uint64_t boundary_hash;
  uint64_t payload_bytes;
};
#pragma pack(pop)

static_assert(sizeof(CacheHeader) == 88, "cache header layout changed");

class FingerprintBuilder {
 public:
  void AddBytes(const void* bytes, size_t size) {
    const auto* p = static_cast<const unsigned char*>(bytes);
    for (size_t i = 0; i < size; ++i) {
      value_ ^= static_cast<uint64_t>(p[i]);
      value_ *= kFnvPrime;
    }
  }

  template <typename T>
  void Add(const T& value) {
    AddBytes(&value, sizeof(T));
  }

  void AddString(const std::string& value) {
    const uint64_t size = static_cast<uint64_t>(value.size());
    Add(size);
    AddBytes(value.data(), value.size());
  }

  uint64_t value() const {
    return value_;
  }

 private:
  uint64_t value_ = kFnvOffset;
};

uint64_t HashBytes(const void* data, size_t size) {
  FingerprintBuilder hash;
  hash.AddBytes(data, size);
  return hash.value();
}

bool CheckedPixelCount(int width, int height, size_t* count) {
  if (!count || width <= 0 || height <= 0)
    return false;
  const uint64_t n = static_cast<uint64_t>(static_cast<uint32_t>(width)) *
                     static_cast<uint64_t>(static_cast<uint32_t>(height));
  if (n > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
      n > static_cast<uint64_t>(std::vector<uint32_t>().max_size()))
    return false;
  *count = static_cast<size_t>(n);
  return true;
}

bool ForegroundGateEnabled(const Config& config) {
  return config.grow_use_mask || config.grow_min_intensity > 0.0f;
}

std::string NormalizedAbsolutePath(const std::string& path) {
  std::error_code ec;
  fs::path absolute = fs::absolute(fs::path(path), ec);
  if (ec)
    return fs::path(path).lexically_normal().string();
  return absolute.lexically_normal().string();
}

bool AddFileFingerprint(const std::string& path, FingerprintBuilder* hash) {
  if (!hash || path.empty())
    return false;

  const std::string normalized = NormalizedAbsolutePath(path);
  hash->AddString(normalized);

  std::error_code ec;
  const fs::file_status status = fs::status(path, ec);
  if (ec || !fs::is_regular_file(status))
    return false;

  const uint64_t size = static_cast<uint64_t>(fs::file_size(path, ec));
  if (ec)
    return false;
  const auto write_time = fs::last_write_time(path, ec);
  if (ec)
    return false;
  const int64_t write_ticks =
      static_cast<int64_t>(write_time.time_since_epoch().count());

  hash->Add(size);
  hash->Add(write_ticks);

  // Metadata makes the common unchanged case cheap to distinguish in the
  // serialized fingerprint, while hashing the bytes prevents a same-size
  // replacement with a restored/coarse timestamp from being accepted as
  // current. This is deliberately streamed so large captures do not require a
  // second full image-sized host allocation.
  std::ifstream input(path, std::ios::binary);
  if (!input)
    return false;
  std::array<char, 1 << 20> buffer{};
  while (input) {
    input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    const std::streamsize bytes = input.gcount();
    if (bytes > 0)
      hash->AddBytes(buffer.data(), static_cast<size_t>(bytes));
  }
  if (!input.eof())
    return false;
  return true;
}

void AddFloat(float value, FingerprintBuilder* hash) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
  std::memcpy(&bits, &value, sizeof(bits));
  hash->Add(bits);
}

bool LoadPreparedGray(const Config& config, const Camera& camera, Image* gray) {
  if (!gray)
    return false;
  const std::string image_path =
      image_loader::FormatPath(config.image_dir, camera.id());
  Image loaded = image_loader::LoadGrayImage(image_path, config.downsample);
  if (loaded.Empty() || loaded.width <= 0 || loaded.height <= 0 ||
      loaded.channels != 1) {
    LOG_WARN("Grow cache: skipping camera %u; unreadable image %s", camera.id(),
             image_path.c_str());
    return false;
  }

  if (config.distorted_images) {
    const GpuCamera gpu_camera = camera.BuildGpuCamera();
    image_loader::UndistortImage(loaded, gpu_camera.K, gpu_camera.dist);
  }
  *gray = std::move(loaded);
  return true;
}

bool BuildBoundary(const Config& config, const Camera& camera, int width,
                   int height, const Image* prepared_gray,
                   std::vector<uint8_t>* boundary) {
  if (!boundary)
    return false;
  boundary->clear();
  if (!ForegroundGateEnabled(config))
    return true;

  size_t pixel_count = 0;
  if (!CheckedPixelCount(width, height, &pixel_count))
    return false;

  Image loaded_gray;
  const Image* gray = prepared_gray;
  if (config.grow_min_intensity > 0.0f) {
    if (!gray) {
      if (!LoadPreparedGray(config, camera, &loaded_gray))
        return false;
      gray = &loaded_gray;
    }
    if (gray->width != width || gray->height != height || gray->channels != 1) {
      LOG_WARN("Grow cache: camera %u image is %dx%d, expected %dx%d", camera.id(),
               gray->width, gray->height, width, height);
      return false;
    }
  }

  Image mask;
  if (config.grow_use_mask) {
    if (config.mask_dir.empty()) {
      LOG_WARN("Grow cache: skipping camera %u; grow.use_mask is enabled but "
               "data.mask_dir is empty",
               camera.id());
      return false;
    }
    const std::string mask_path =
        image_loader::FormatPath(config.mask_dir, camera.id());
    mask = image_loader::LoadMaskImage(mask_path, config.downsample);
    if (mask.Empty() || mask.width != width || mask.height != height ||
        mask.channels != 1) {
      LOG_WARN("Grow cache: skipping camera %u; invalid mask %s (%dx%d, expected "
               "%dx%d)",
               camera.id(), mask_path.c_str(), mask.width, mask.height, width,
               height);
      return false;
    }
    if (config.distorted_images) {
      const GpuCamera gpu_camera = camera.BuildGpuCamera();
      image_loader::UndistortImage(mask, gpu_camera.K, gpu_camera.dist);
    }
  }

  boundary->resize(pixel_count);
  for (size_t i = 0; i < pixel_count; ++i) {
    bool accepted = true;
    if (config.grow_use_mask)
      accepted = std::isfinite(mask.data[i]) && mask.data[i] > 0.5f;
    if (accepted && config.grow_min_intensity > 0.0f) {
      accepted = std::isfinite(gray->data[i]) &&
                 gray->data[i] >= config.grow_min_intensity;
    }
    (*boundary)[i] = accepted ? 255u : 0u;
  }
  return true;
}

std::string TemporaryPath(const std::string& final_path) {
  static std::atomic<uint64_t> sequence{0};
#ifdef _WIN32
  const unsigned long process_id = static_cast<unsigned long>(_getpid());
#else
  const unsigned long process_id = static_cast<unsigned long>(getpid());
#endif
  const uint64_t thread_id = static_cast<uint64_t>(
      std::hash<std::thread::id>{}(std::this_thread::get_id()));
  return final_path + ".tmp." + std::to_string(process_id) + "." +
         std::to_string(thread_id) + "." +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed));
}

bool FlushFile(FILE* file) {
  if (!file || std::fflush(file) != 0)
    return false;
#ifdef _WIN32
  return _commit(_fileno(file)) == 0;
#else
  return fsync(fileno(file)) == 0;
#endif
}

bool AtomicReplace(const std::string& from, const std::string& to) {
#ifdef _WIN32
  return MoveFileExA(from.c_str(), to.c_str(),
                     MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
  return std::rename(from.c_str(), to.c_str()) == 0;
#endif
}

bool WriteAll(FILE* file, const void* data, size_t bytes) {
  if (bytes == 0)
    return true;
  return file && std::fwrite(data, 1, bytes, file) == bytes;
}

bool WriteCacheWithFingerprint(const Config& config, const PreparedView& view,
                               uint64_t fingerprint) {
  size_t pixel_count = 0;
  if (!CheckedPixelCount(view.width, view.height, &pixel_count) ||
      view.camera_index < 0 || view.orientation_variance.size() != pixel_count ||
      (!view.boundary.empty() && view.boundary.size() != pixel_count)) {
    LOG_WARN("Grow cache: refusing invalid host map for camera %u", view.camera_id);
    return false;
  }

  const bool has_boundary = !view.boundary.empty();
  if (has_boundary != ForegroundGateEnabled(config)) {
    LOG_WARN("Grow cache: camera %u boundary payload does not match grow gating",
             view.camera_id);
    return false;
  }

  const uint64_t orient_bytes =
      static_cast<uint64_t>(pixel_count) * sizeof(uint32_t);
  const uint64_t boundary_bytes = has_boundary ? pixel_count : 0;
  if (orient_bytes > std::numeric_limits<uint64_t>::max() - boundary_bytes)
    return false;

  CacheHeader header{};
  std::memcpy(header.magic, kMagic, sizeof(kMagic));
  header.version = kFormatVersion;
  header.header_size = sizeof(CacheHeader);
  header.endian_marker = kEndianMarker;
  header.flags = has_boundary ? kFlagBoundary : 0u;
  header.camera_id = view.camera_id;
  header.camera_index = view.camera_index;
  header.width = static_cast<uint32_t>(view.width);
  header.height = static_cast<uint32_t>(view.height);
  header.packed_pixel_bytes = kPackedPixelBytes;
  header.pixel_count = pixel_count;
  header.fingerprint = fingerprint;
  header.orientation_hash =
      HashBytes(view.orientation_variance.data(), static_cast<size_t>(orient_bytes));
  header.boundary_hash = has_boundary
                             ? HashBytes(view.boundary.data(), view.boundary.size())
                             : 0;
  header.payload_bytes = orient_bytes + boundary_bytes;

  const std::string final_path = CachePath(config, view.camera_id);
  std::error_code ec;
  fs::create_directories(fs::path(final_path).parent_path(), ec);
  if (ec) {
    LOG_WARN("Grow cache: cannot create %s: %s",
             fs::path(final_path).parent_path().string().c_str(),
             ec.message().c_str());
    return false;
  }

  const std::string temporary_path = TemporaryPath(final_path);
  FILE* file = std::fopen(temporary_path.c_str(), "wb");
  if (!file) {
    LOG_WARN("Grow cache: cannot open temporary file %s: %s",
             temporary_path.c_str(), std::strerror(errno));
    return false;
  }

  bool ok = WriteAll(file, &header, sizeof(header)) &&
            WriteAll(file, view.orientation_variance.data(),
                     static_cast<size_t>(orient_bytes)) &&
            WriteAll(file, view.boundary.data(),
                     static_cast<size_t>(boundary_bytes)) &&
            FlushFile(file);
  if (std::fclose(file) != 0)
    ok = false;

  if (ok)
    ok = AtomicReplace(temporary_path, final_path);
  if (!ok) {
    LOG_WARN("Grow cache: failed to atomically write %s", final_path.c_str());
    std::remove(temporary_path.c_str());
    return false;
  }

  LOG_DEBUG("Grow cache: wrote camera %u (%dx%d) to %s", view.camera_id,
            view.width, view.height, final_path.c_str());
  return true;
}

bool ReadExact(std::ifstream* input, void* data, size_t bytes) {
  if (!input ||
      bytes >
          static_cast<size_t>(std::numeric_limits<std::streamsize>::max()))
    return false;
  input->read(static_cast<char*>(data), static_cast<std::streamsize>(bytes));
  return input->good() || (input->eof() && input->gcount() ==
                                              static_cast<std::streamsize>(bytes));
}

void LogInvalidCache(unsigned int camera_id, const std::string& reason) {
  LOG_WARN("Grow cache: ignoring camera %u cache: %s", camera_id, reason.c_str());
}

}  // namespace

std::string CachePath(const Config& config, unsigned int camera_id) {
  return (fs::path(config.output_dir) / ".grow_cache" /
          ("camera_" + std::to_string(camera_id) + ".half2"))
      .string();
}

bool ComputeFingerprint(const Config& config, const Camera& camera,
                        int camera_index, uint64_t* fingerprint) {
  if (!fingerprint || camera_index < 0)
    return false;

  FingerprintBuilder hash;
  hash.AddString("multiview-hair-grow-cache-v1");
  hash.Add(kFormatVersion);
  hash.Add(camera_index);
  const unsigned int camera_id = camera.id();
  hash.Add(camera_id);

  const std::string image_path =
      image_loader::FormatPath(config.image_dir, camera_id);
  if (!AddFileFingerprint(image_path, &hash))
    return false;

  hash.AddString(config.image_dir);
  AddFloat(config.downsample, &hash);
  const uint8_t distorted = config.distorted_images ? 1u : 0u;
  hash.Add(distorted);

  hash.Add(config.gabor_num_orientations);
  hash.Add(config.gabor_kernel_size);
  AddFloat(config.gabor_sigma, &hash);
  AddFloat(config.gabor_gamma, &hash);
  AddFloat(config.gabor_lambda, &hash);
  AddFloat(config.gabor_min_contrast, &hash);
  AddFloat(config.gabor_min_response, &hash);

  const uint8_t use_mask = config.grow_use_mask ? 1u : 0u;
  hash.Add(use_mask);
  AddFloat(config.grow_min_intensity, &hash);
  if (config.grow_use_mask) {
    if (config.mask_dir.empty())
      return false;
    const std::string mask_path =
        image_loader::FormatPath(config.mask_dir, camera_id);
    if (!AddFileFingerprint(mask_path, &hash))
      return false;
    hash.AddString(config.mask_dir);
  }

  const GpuCamera gpu_camera = camera.BuildGpuCamera();
  for (float value : gpu_camera.K)
    AddFloat(value, &hash);
  for (float value : gpu_camera.R)
    AddFloat(value, &hash);
  for (float value : gpu_camera.t)
    AddFloat(value, &hash);
  for (float value : gpu_camera.center)
    AddFloat(value, &hash);
  for (float value : gpu_camera.dist)
    AddFloat(value, &hash);

  *fingerprint = hash.value();
  return true;
}

bool StoreView(const Config& config, const Camera& camera,
               const PreparedView& view) {
  if (view.camera_id != camera.id()) {
    LOG_WARN("Grow cache: camera id mismatch (%u != %u)", view.camera_id,
             camera.id());
    return false;
  }
  uint64_t fingerprint = 0;
  if (!ComputeFingerprint(config, camera, view.camera_index, &fingerprint)) {
    LOG_WARN("Grow cache: cannot fingerprint source for camera %u", camera.id());
    return false;
  }
  return WriteCacheWithFingerprint(config, view, fingerprint);
}

bool LoadView(const Config& config, const Camera& camera, int camera_index,
              PreparedView* view) {
  if (!view || camera_index < 0)
    return false;

  uint64_t current_fingerprint = 0;
  if (!ComputeFingerprint(config, camera, camera_index, &current_fingerprint))
    return false;

  const std::string path = CachePath(config, camera.id());
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return false;
  const std::streamoff file_size = input.tellg();
  if (file_size < static_cast<std::streamoff>(sizeof(CacheHeader))) {
    LogInvalidCache(camera.id(), "truncated header");
    return false;
  }
  input.seekg(0, std::ios::beg);

  CacheHeader header{};
  if (!ReadExact(&input, &header, sizeof(header))) {
    LogInvalidCache(camera.id(), "unreadable header");
    return false;
  }
  if (std::memcmp(header.magic, kMagic, sizeof(kMagic)) != 0 ||
      header.version != kFormatVersion ||
      header.header_size != sizeof(CacheHeader) ||
      header.endian_marker != kEndianMarker ||
      header.packed_pixel_bytes != kPackedPixelBytes ||
      (header.flags & ~kFlagBoundary) != 0) {
    LogInvalidCache(camera.id(), "unsupported format");
    return false;
  }
  if (header.camera_id != camera.id() || header.camera_index != camera_index) {
    LogInvalidCache(camera.id(), "camera identity mismatch");
    return false;
  }
  if (header.fingerprint != current_fingerprint) {
    LOG_DEBUG("Grow cache: camera %u entry is stale", camera.id());
    return false;
  }
  if (header.width == 0 || header.height == 0 ||
      header.width > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
      header.height > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
    LogInvalidCache(camera.id(), "invalid dimensions");
    return false;
  }

  size_t pixel_count = 0;
  if (!CheckedPixelCount(static_cast<int>(header.width),
                         static_cast<int>(header.height), &pixel_count) ||
      header.pixel_count != pixel_count) {
    LogInvalidCache(camera.id(), "invalid pixel count");
    return false;
  }

  const bool has_boundary = (header.flags & kFlagBoundary) != 0;
  if (has_boundary != ForegroundGateEnabled(config)) {
    LogInvalidCache(camera.id(), "foreground gate mismatch");
    return false;
  }
  const uint64_t orient_bytes =
      static_cast<uint64_t>(pixel_count) * sizeof(uint32_t);
  const uint64_t boundary_bytes = has_boundary ? pixel_count : 0;
  const uint64_t max_stream_size =
      static_cast<uint64_t>(std::numeric_limits<std::streamoff>::max());
  if (orient_bytes > std::numeric_limits<uint64_t>::max() - boundary_bytes ||
      header.payload_bytes != orient_bytes + boundary_bytes ||
      header.payload_bytes > max_stream_size - sizeof(CacheHeader) ||
      file_size != static_cast<std::streamoff>(sizeof(CacheHeader)) +
                       static_cast<std::streamoff>(header.payload_bytes)) {
    LogInvalidCache(camera.id(), "payload size mismatch");
    return false;
  }

  PreparedView loaded;
  loaded.camera_index = camera_index;
  loaded.camera_id = camera.id();
  loaded.width = static_cast<int>(header.width);
  loaded.height = static_cast<int>(header.height);
  loaded.camera = camera.BuildGpuCamera();
  loaded.orientation_variance.resize(pixel_count);
  if (!ReadExact(&input, loaded.orientation_variance.data(),
                 static_cast<size_t>(orient_bytes))) {
    LogInvalidCache(camera.id(), "truncated orientation payload");
    return false;
  }
  if (has_boundary) {
    loaded.boundary.resize(pixel_count);
    if (!ReadExact(&input, loaded.boundary.data(), loaded.boundary.size())) {
      LogInvalidCache(camera.id(), "truncated boundary payload");
      return false;
    }
  }

  if (HashBytes(loaded.orientation_variance.data(),
                static_cast<size_t>(orient_bytes)) != header.orientation_hash ||
      (has_boundary &&
       HashBytes(loaded.boundary.data(), loaded.boundary.size()) !=
           header.boundary_hash) ||
      (!has_boundary && header.boundary_hash != 0)) {
    LogInvalidCache(camera.id(), "payload checksum mismatch");
    return false;
  }

  *view = std::move(loaded);
  return true;
}

bool StoreDeviceMaps(const Config& config, const Camera& camera,
                     int camera_index, int width, int height,
                     const float* d_theta, const float* d_variance,
                     size_t theta_pitch_bytes, size_t variance_pitch_bytes,
                     int gpu_id) {
  size_t pixel_count = 0;
  if (!d_theta || !d_variance ||
      !CheckedPixelCount(width, height, &pixel_count)) {
    LOG_WARN("Grow cache: invalid GPU maps for camera %u", camera.id());
    return false;
  }
  if (theta_pitch_bytes == 0)
    theta_pitch_bytes = static_cast<size_t>(width) * sizeof(float);
  if (variance_pitch_bytes == 0)
    variance_pitch_bytes = static_cast<size_t>(width) * sizeof(float);
  if (theta_pitch_bytes < static_cast<size_t>(width) * sizeof(float) ||
      variance_pitch_bytes < static_cast<size_t>(width) * sizeof(float)) {
    LOG_WARN("Grow cache: invalid GPU pitch for camera %u", camera.id());
    return false;
  }

  if (gpu_id >= 0) {
    const cudaError_t error = cudaSetDevice(gpu_id);
    if (error != cudaSuccess) {
      LOG_WARN("Grow cache: cannot select GPU %d for camera %u: %s", gpu_id,
               camera.id(), cudaGetErrorString(error));
      return false;
    }
  }

  PreparedView prepared;
  prepared.camera_index = camera_index;
  prepared.camera_id = camera.id();
  prepared.width = width;
  prepared.height = height;
  prepared.camera = camera.BuildGpuCamera();
  prepared.orientation_variance.resize(pixel_count);

  std::string error;
  if (!detail::PackDeviceMapsToHost(
          d_theta, theta_pitch_bytes, d_variance, variance_pitch_bytes, width,
          height, prepared.orientation_variance.data(), &error)) {
    LOG_WARN("Grow cache: cannot pack camera %u maps: %s", camera.id(),
             error.c_str());
    return false;
  }
  if (!BuildBoundary(config, camera, width, height, nullptr,
                     &prepared.boundary))
    return false;
  return StoreView(config, camera, prepared);
}

bool EnsureCache(const Config& config, const Camera& camera, int camera_index,
                 int gpu_id, PreparedView* view) {
  if (!view)
    return false;
  if (LoadView(config, camera, camera_index, view))
    return true;

  uint64_t source_fingerprint = 0;
  if (!ComputeFingerprint(config, camera, camera_index, &source_fingerprint)) {
    LOG_WARN("Grow cache: skipping camera %u; source metadata is unavailable",
             camera.id());
    return false;
  }

  Image gray;
  if (!LoadPreparedGray(config, camera, &gray))
    return false;
  size_t pixel_count = 0;
  if (!CheckedPixelCount(gray.width, gray.height, &pixel_count))
    return false;

  PreparedView generated;
  generated.camera_index = camera_index;
  generated.camera_id = camera.id();
  generated.width = gray.width;
  generated.height = gray.height;
  generated.camera = camera.BuildGpuCamera();
  if (!BuildBoundary(config, camera, gray.width, gray.height, &gray,
                     &generated.boundary))
    return false;
  generated.orientation_variance.resize(pixel_count);

  float* d_gray = nullptr;
  float* d_theta = nullptr;
  float* d_variance = nullptr;
  bool succeeded = false;
  try {
    cudaError_t error = cudaSetDevice(gpu_id);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice: ") +
                               cudaGetErrorString(error));
    error = cudaMalloc(&d_gray, pixel_count * sizeof(float));
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("cudaMalloc(gray): ") +
                               cudaGetErrorString(error));
    error = cudaMalloc(&d_theta, pixel_count * sizeof(float));
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("cudaMalloc(theta): ") +
                               cudaGetErrorString(error));
    error = cudaMalloc(&d_variance, pixel_count * sizeof(float));
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("cudaMalloc(variance): ") +
                               cudaGetErrorString(error));
    error = cudaMemcpy(d_gray, gray.data, pixel_count * sizeof(float),
                       cudaMemcpyHostToDevice);
    if (error != cudaSuccess)
      throw std::runtime_error(std::string("cudaMemcpy(gray): ") +
                               cudaGetErrorString(error));

    GaborParams params{};
    params.ksize = config.gabor_kernel_size;
    params.sigma = config.gabor_sigma;
    params.gamma = config.gabor_gamma;
    params.lambd = config.gabor_lambda;
    params.min_contrast = config.gabor_min_contrast;
    params.min_response = config.gabor_min_response;
    ComputeGaborOrientation(d_theta, d_variance, d_gray,
                            static_cast<unsigned int>(gray.width),
                            static_cast<unsigned int>(gray.height),
                            config.gabor_num_orientations, params, nullptr);

    std::string error_message;
    succeeded = detail::PackDeviceMapsToHost(
        d_theta, static_cast<size_t>(gray.width) * sizeof(float), d_variance,
        static_cast<size_t>(gray.width) * sizeof(float), gray.width, gray.height,
        generated.orientation_variance.data(), &error_message);
    if (!succeeded)
      LOG_WARN("Grow cache: Gabor packing failed for camera %u: %s", camera.id(),
               error_message.c_str());
  } catch (const std::exception& exception) {
    LOG_WARN("Grow cache: Gabor computation failed for camera %u on GPU %d: %s",
             camera.id(), gpu_id, exception.what());
  }

  if (d_variance)
    cudaFree(d_variance);
  if (d_theta)
    cudaFree(d_theta);
  if (d_gray)
    cudaFree(d_gray);
  cudaGetLastError();
  if (!succeeded)
    return false;

  uint64_t final_fingerprint = 0;
  if (!ComputeFingerprint(config, camera, camera_index, &final_fingerprint) ||
      final_fingerprint != source_fingerprint) {
    LOG_WARN("Grow cache: camera %u source changed while its map was generated",
             camera.id());
    return false;
  }
  if (!WriteCacheWithFingerprint(config, generated, source_fingerprint))
    return false;

  *view = std::move(generated);
  return true;
}

std::vector<PreparedView> PrepareViews(const CameraArray& cameras,
                                       const Config& config, int gpu_id) {
  std::vector<PreparedView> prepared;
  prepared.reserve(static_cast<size_t>(std::max(0, cameras.NumCameras())));
  for (int camera_index = 0; camera_index < cameras.NumCameras(); ++camera_index) {
    PreparedView view;
    if (!EnsureCache(config, cameras.GetCamera(camera_index), camera_index, gpu_id,
                     &view)) {
      LOG_WARN("Grow cache: camera %u is unavailable",
               cameras.GetCamId(camera_index));
      continue;
    }
    prepared.push_back(std::move(view));
  }

  if (prepared.empty())
    return prepared;

  // Layered CUDA images require one resolution.  Retain the modal resolution
  // so a single malformed first camera cannot discard an otherwise valid rig.
  std::map<std::pair<int, int>, size_t> resolution_counts;
  for (const PreparedView& view : prepared)
    ++resolution_counts[{view.width, view.height}];
  const auto modal = std::max_element(
      resolution_counts.begin(), resolution_counts.end(),
      [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
  const std::pair<int, int> expected = modal->first;

  prepared.erase(
      std::remove_if(prepared.begin(), prepared.end(), [&](const PreparedView& view) {
        if (view.width == expected.first && view.height == expected.second)
          return false;
        LOG_WARN("Grow cache: skipping camera %u with incompatible size %dx%d "
                 "(expected %dx%d)",
                 view.camera_id, view.width, view.height, expected.first,
                 expected.second);
        return true;
      }),
      prepared.end());

  LOG_INFO("Grow cache: prepared %zu/%d camera views at %dx%d", prepared.size(),
           cameras.NumCameras(), expected.first, expected.second);
  return prepared;
}

}  // namespace orientation_cache

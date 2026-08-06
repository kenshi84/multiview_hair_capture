// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.

#include <cuda_runtime.h>

#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;

int failures = 0;

#define CHECK(condition)                                                        \
  do {                                                                          \
    if (!(condition)) {                                                         \
      ++failures;                                                               \
      std::cerr << "[FAIL] " << __FILE__ << ':' << __LINE__                   \
                << ": " #condition "\n";                                     \
    }                                                                           \
  } while (0)

class TempDirectory {
 public:
  TempDirectory() {
    const auto seed = static_cast<unsigned long long>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    path_ = fs::temp_directory_path() /
            ("multiview-hair-grow-cli-" + std::to_string(seed));
    fs::create_directories(path_);
  }

  ~TempDirectory() {
    std::error_code ignored;
    fs::remove_all(path_, ignored);
  }

  const fs::path& path() const { return path_; }

 private:
  fs::path path_;
};

std::string TomlPath(const fs::path& path) {
  std::string result = fs::absolute(path).generic_string();
  size_t offset = 0;
  while ((offset = result.find('"', offset)) != std::string::npos) {
    result.insert(offset, "\\");
    offset += 2;
  }
  return result;
}

void WriteText(const fs::path& path, const std::string& contents) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path);
  if (!output)
    throw std::runtime_error("cannot write " + path.string());
  output << contents;
}

std::vector<char> ReadBytes(const fs::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input)
    return {};
  const std::streamoff size = input.tellg();
  if (size < 0)
    return {};
  std::vector<char> bytes(static_cast<size_t>(size));
  input.seekg(0);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  return input || bytes.empty() ? bytes : std::vector<char>{};
}

void WriteLe16(std::ofstream* output, uint16_t value) {
  const char bytes[2] = {static_cast<char>(value),
                         static_cast<char>(value >> 8)};
  output->write(bytes, sizeof(bytes));
}

void WriteLe32(std::ofstream* output, uint32_t value) {
  const char bytes[4] = {static_cast<char>(value),
                         static_cast<char>(value >> 8),
                         static_cast<char>(value >> 16),
                         static_cast<char>(value >> 24)};
  output->write(bytes, sizeof(bytes));
}

void WriteStripedBmp(const fs::path& path, int phase) {
  constexpr int width = 64;
  constexpr int height = 64;
  constexpr uint32_t row_bytes = width * 3;
  constexpr uint32_t pixel_bytes = row_bytes * height;
  constexpr uint32_t header_bytes = 14 + 40;
  std::ofstream output(path, std::ios::binary);
  if (!output)
    throw std::runtime_error("cannot write " + path.string());

  output.put('B');
  output.put('M');
  WriteLe32(&output, header_bytes + pixel_bytes);
  WriteLe16(&output, 0);
  WriteLe16(&output, 0);
  WriteLe32(&output, header_bytes);
  WriteLe32(&output, 40);
  WriteLe32(&output, width);
  WriteLe32(&output, height);
  WriteLe16(&output, 1);
  WriteLe16(&output, 24);
  WriteLe32(&output, 0);
  WriteLe32(&output, pixel_bytes);
  WriteLe32(&output, 2835);
  WriteLe32(&output, 2835);
  WriteLe32(&output, 0);
  WriteLe32(&output, 0);

  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const unsigned char value = phase < 0
                                      ? (x < width / 2 ? 255 : 0)
                                      : (((x + phase) / 2) % 2 ? 240 : 15);
      output.put(static_cast<char>(value));
      output.put(static_cast<char>(value));
      output.put(static_cast<char>(value));
    }
  }
}

struct BinaryPoint {
  std::array<float, 3> position;
  std::array<float, 3> direction;
};

void WriteStrands(const fs::path& path,
                  const std::vector<std::vector<BinaryPoint>>& strands) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path, std::ios::binary);
  const int count = static_cast<int>(strands.size());
  output.write(reinterpret_cast<const char*>(&count), sizeof(count));
  for (const auto& strand : strands) {
    const int points = static_cast<int>(strand.size());
    output.write(reinterpret_cast<const char*>(&points), sizeof(points));
    for (const BinaryPoint& point : strand) {
      output.write(reinterpret_cast<const char*>(point.position.data()),
                   3 * sizeof(float));
      output.write(reinterpret_cast<const char*>(point.direction.data()),
                   3 * sizeof(float));
    }
  }
  if (!output)
    throw std::runtime_error("cannot write " + path.string());
}

bool HasReadableStrandHeader(const fs::path& path, int expected_strands) {
  std::ifstream input(path, std::ios::binary);
  int count = -1;
  input.read(reinterpret_cast<char*>(&count), sizeof(count));
  return static_cast<bool>(input) && count == expected_strands;
}

bool HasPlyHeader(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  std::string signature;
  std::getline(input, signature);
  return signature == "ply";
}

uint32_t ReadLe32(const std::vector<char>& bytes, size_t offset) {
  if (offset + 4 > bytes.size())
    return 0;
  return static_cast<uint32_t>(static_cast<unsigned char>(bytes[offset])) |
         (static_cast<uint32_t>(
              static_cast<unsigned char>(bytes[offset + 1]))
          << 8) |
         (static_cast<uint32_t>(
              static_cast<unsigned char>(bytes[offset + 2]))
          << 16) |
         (static_cast<uint32_t>(
              static_cast<unsigned char>(bytes[offset + 3]))
          << 24);
}

int CountCacheForeground(const fs::path& path) {
  const std::vector<char> bytes = ReadBytes(path);
  constexpr size_t flags_offset = 20;
  constexpr size_t width_offset = 32;
  constexpr size_t height_offset = 36;
  if (bytes.size() < 88 || (ReadLe32(bytes, flags_offset) & 1u) == 0)
    return -1;
  const size_t width = ReadLe32(bytes, width_offset);
  const size_t height = ReadLe32(bytes, height_offset);
  const size_t pixels = width * height;
  const size_t boundary_offset = 88 + pixels * sizeof(uint32_t);
  if (width == 0 || height == 0 || boundary_offset + pixels != bytes.size())
    return -1;
  int foreground = 0;
  for (size_t i = 0; i < pixels; ++i)
    foreground += bytes[boundary_offset + i] != 0;
  return foreground;
}

std::string Quote(const fs::path& path) {
  std::string value = path.string();
  std::string quoted = "\"";
  for (char c : value) {
    if (c == '"')
      quoted += '\\';
    quoted += c;
  }
  return quoted + '"';
}

int Run(const fs::path& executable, const std::string& arguments,
        const fs::path& log) {
  const std::string command = Quote(executable) + " " + arguments + " >" +
                              Quote(log) + " 2>&1";
  return std::system(command.c_str());
}

void PrintLogOnFailure(const fs::path& log) {
  if (failures == 0)
    return;
  std::ifstream input(log);
  std::cerr << input.rdbuf();
}

std::string MinimalConfig(const fs::path& output) {
  std::ostringstream config;
  config << "[data]\n"
         << "cameras_json = \"missing-cameras.json\"\n"
         << "image_dir = \"missing_%u.bmp\"\n"
         << "output_dir = \"" << TomlPath(output) << "\"\n";
  return config.str();
}

void TestBasicCli(const fs::path& executable, const fs::path& root) {
  const fs::path help_log = root / "help.log";
  CHECK(Run(executable, "", help_log) != 0);
  const std::vector<char> help = ReadBytes(help_log);
  CHECK(std::string(help.begin(), help.end()).find("grow") != std::string::npos);

  const fs::path missing_output = root / "missing-output";
  fs::create_directories(missing_output);
  const fs::path missing_config = root / "missing.toml";
  WriteText(missing_config, MinimalConfig(missing_output));
  CHECK(Run(executable, "grow " + Quote(missing_config),
            root / "missing-grow.log") != 0);

  // A clean file must never be accepted as a mesh fallback.
  WriteStrands(missing_output / "strands_clean.bin", {});
  CHECK(Run(executable, "mesh " + Quote(missing_config),
            root / "strict-mesh.log") != 0);

  const fs::path empty_output = root / "empty-output";
  fs::create_directories(empty_output);
  WriteStrands(empty_output / "strands_clean.bin", {});
  const fs::path empty_config = root / "empty.toml";
  WriteText(empty_config, MinimalConfig(empty_output));
  const fs::path empty_log = root / "empty-grow.log";
  CHECK(Run(executable, "grow " + Quote(empty_config), empty_log) == 0);
  CHECK(HasReadableStrandHeader(empty_output / "strands_grown.bin", 0));
  CHECK(HasPlyHeader(empty_output / "strands_grown.ply"));
  CHECK(!fs::exists(empty_output / ".grow_cache"));
  CHECK(Run(executable, "mesh " + Quote(empty_config),
            root / "empty-mesh.log") == 0);
  CHECK(HasPlyHeader(empty_output / "hair_mesh.ply"));
  PrintLogOnFailure(empty_log);
}

void WriteCameras(const fs::path& path) {
  std::ostringstream json;
  json << "{\"cameras\":{";
  constexpr double pi = 3.14159265358979323846;
  for (int i = 0; i < 8; ++i) {
    const double angle = 2.0 * pi * i / 8.0;
    const double cx = 2.0 * std::cos(angle);
    const double cy = 2.0 * std::sin(angle);
    if (i)
      json << ',';
    json << '\"' << (100 + i) << "\":{";
    json << "\"intrinsic\":[[50,0,32],[0,50,32],[0,0,1]],";
    json << "\"extrinsic\":[[1,0,0," << -cx << "],[0,1,0," << -cy
         << "],[0,0,1,0]],";
    json << "\"distortion\":[0,0,0,0,0]}";
  }
  json << "}}";
  WriteText(path, json.str());
}

void TestCacheAndOutputs(const fs::path& executable, const fs::path& root) {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    std::cout << "[SKIP] CLI cache smoke: no CUDA device is available\n";
    return;
  }

  const fs::path capture = root / "synthetic";
  const fs::path images = capture / "images";
  const fs::path masks = capture / "masks";
  const fs::path output = capture / "output";
  fs::create_directories(images);
  fs::create_directories(masks);
  fs::create_directories(output);
  WriteCameras(capture / "cameras.json");
  for (int i = 0; i < 8; ++i) {
    WriteStripedBmp(images / ("image_" + std::to_string(100 + i) + ".bmp"), 0);
    WriteStripedBmp(masks / ("mask_" + std::to_string(100 + i) + ".bmp"), -1);
  }

  WriteStrands(output / "strands_clean.bin",
               {{{{{0.0f, 0.0f, 20.0f}}, {{0.0f, 0.0f, 1.0f}}},
                 {{{0.0f, 0.0f, 21.0f}}, {{0.0f, 0.0f, 1.0f}}}}});

  const fs::path config_path = capture / "config.toml";
  const auto write_config = [&](bool use_mask, float min_intensity) {
    std::ostringstream config;
    config << "[data]\n"
           << "cameras_json = \"" << TomlPath(capture / "cameras.json")
           << "\"\n"
           << "image_dir = \"" << TomlPath(images / "image_%u.bmp")
           << "\"\n"
           << "mask_dir = \"" << TomlPath(masks / "mask_%u.bmp") << "\"\n"
           << "output_dir = \"" << TomlPath(output) << "\"\n"
           << "downsample = 1.0\n"
           << "distorted_images = false\n\n"
           << "[mvs]\nnum_gpus = 1\n\n"
           << "[mvs.gabor]\nnum_orientations = 18\nkernel_size = 9\n"
           << "sigma = 1.12\ngamma = 0.28\nlambda = 3.0\n\n"
           << "[grow]\nstep_size = 0.1\nmax_growth_length = 0.1\n"
           << "min_views = 8\nuse_mask = "
           << (use_mask ? "true" : "false") << "\nmin_intensity = "
           << min_intensity << "\n\n"
           << "[debug]\ngpu_id = 0\nlog_level = \"warning\"\n";
    WriteText(config_path, config.str());
  };
  write_config(false, 0.0f);

  const fs::path first_log = capture / "first.log";
  CHECK(Run(executable, "grow " + Quote(config_path), first_log) == 0);
  CHECK(HasReadableStrandHeader(output / "strands_grown.bin", 1));
  CHECK(HasPlyHeader(output / "strands_grown.ply"));
  for (int i = 0; i < 8; ++i) {
    CHECK(fs::is_regular_file(output / ".grow_cache" /
                              ("camera_" + std::to_string(100 + i) + ".half2")));
  }

  const fs::path cache = output / ".grow_cache" / "camera_100.half2";
  const std::vector<char> before = ReadBytes(cache);
  const fs::path changed_image = images / "image_100.bmp";
  const auto previous_time = fs::last_write_time(changed_image);
  WriteStripedBmp(changed_image, 1);
  fs::last_write_time(changed_image, previous_time + std::chrono::seconds(2));
  CHECK(Run(executable, "grow " + Quote(config_path),
            capture / "second.log") == 0);
  const std::vector<char> after = ReadBytes(cache);
  CHECK(!before.empty());
  CHECK(before != after);

  // Foreground policies are support maps, and the combined mode is a logical
  // AND. The striped image and half-frame mask make the three counts exact.
  write_config(false, 0.5f);
  CHECK(Run(executable, "grow " + Quote(config_path),
            capture / "intensity.log") == 0);
  CHECK(CountCacheForeground(cache) == 2048);
  write_config(true, 0.0f);
  CHECK(Run(executable, "grow " + Quote(config_path),
            capture / "mask.log") == 0);
  CHECK(CountCacheForeground(cache) == 2048);
  write_config(true, 0.5f);
  CHECK(Run(executable, "grow " + Quote(config_path),
            capture / "mask-intensity.log") == 0);
  CHECK(CountCacheForeground(cache) == 1024);

  CHECK(Run(executable, "mesh " + Quote(config_path),
            capture / "mesh.log") == 0);
  CHECK(HasPlyHeader(output / "hair_mesh.ply"));
  PrintLogOnFailure(first_log);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: grow_cli_test <hair_recon>\n";
    return 2;
  }
  TempDirectory temp;
  const fs::path executable = fs::absolute(argv[1]);
  TestBasicCli(executable, temp.path());
  TestCacheAndOutputs(executable, temp.path());
  if (failures == 0)
    std::cout << "[PASS] grow CLI smoke tests\n";
  return failures == 0 ? 0 : 1;
}

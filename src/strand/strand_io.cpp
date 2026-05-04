// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/strand_io.h"

#include <cmath>
#include <cstdio>

#include "common/logger.h"

float Strand::Length() const {
  float len = 0;
  for (size_t i = 1; i < NumPoints(); ++i) {
    float dx = positions[i * 3] - positions[(i - 1) * 3];
    float dy = positions[i * 3 + 1] - positions[(i - 1) * 3 + 1];
    float dz = positions[i * 3 + 2] - positions[(i - 1) * 3 + 2];
    len += std::sqrt(dx * dx + dy * dy + dz * dz);
  }
  return len;
}

namespace strand_io {

bool WriteBinary(const std::string& path, const std::vector<Strand>& strands) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    LOG_ERROR("Failed to open %s for writing", path.c_str());
    return false;
  }

  int num_strands = static_cast<int>(strands.size());
  fwrite(&num_strands, sizeof(int), 1, f);

  for (const auto& s : strands) {
    int np = static_cast<int>(s.NumPoints());
    fwrite(&np, sizeof(int), 1, f);
    for (size_t i = 0; i < s.NumPoints(); ++i) {
      fwrite(&s.positions[i * 3], sizeof(float), 3, f);
      fwrite(&s.directions[i * 3], sizeof(float), 3, f);
    }
  }

  fclose(f);
  LOG_INFO("Wrote %d strands to %s", num_strands, path.c_str());
  return true;
}

bool ReadBinary(const std::string& path, std::vector<Strand>& strands) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    LOG_ERROR("Failed to open %s for reading", path.c_str());
    return false;
  }

  int num_strands;
  if (fread(&num_strands, sizeof(int), 1, f) != 1) {
    fclose(f);
    return false;
  }

  strands.resize(num_strands);
  for (int si = 0; si < num_strands; ++si) {
    int np;
    if (fread(&np, sizeof(int), 1, f) != 1) {
      fclose(f);
      return false;
    }
    strands[si].positions.resize(np * 3);
    strands[si].directions.resize(np * 3);
    strands[si].labels.resize(np, si);
    for (int i = 0; i < np; ++i) {
      if (fread(&strands[si].positions[i * 3], sizeof(float), 3, f) != 3 ||
          fread(&strands[si].directions[i * 3], sizeof(float), 3, f) != 3) {
        fclose(f);
        return false;
      }
    }
  }

  fclose(f);
  LOG_INFO("Read %d strands from %s", num_strands, path.c_str());
  return true;
}

bool WritePly(const std::string& path, const std::vector<Strand>& strands) {
  PointCloud cloud = StrandsToPointCloud(strands);

  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    LOG_ERROR("Failed to open %s for writing", path.c_str());
    return false;
  }

  size_t n = cloud.NumPoints();
  fprintf(f, "ply\n");
  fprintf(f, "format binary_little_endian 1.0\n");
  fprintf(f, "element vertex %zu\n", n);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  fprintf(f, "property float nx\n");
  fprintf(f, "property float ny\n");
  fprintf(f, "property float nz\n");
  fprintf(f, "property uint label\n");
  fprintf(f, "end_header\n");

  for (size_t i = 0; i < n; ++i) {
    fwrite(&cloud.positions[i * 3], sizeof(float), 3, f);
    fwrite(&cloud.directions[i * 3], sizeof(float), 3, f);
    unsigned int label = static_cast<unsigned int>(cloud.labels[i]);
    fwrite(&label, sizeof(unsigned int), 1, f);
  }

  fclose(f);
  LOG_INFO("Wrote %zu strand points (%zu strands) to %s", n, strands.size(),
           path.c_str());
  return true;
}

PointCloud StrandsToPointCloud(const std::vector<Strand>& strands) {
  PointCloud cloud;
  for (int si = 0; si < static_cast<int>(strands.size()); ++si) {
    const auto& s = strands[si];
    for (size_t i = 0; i < s.NumPoints(); ++i) {
      cloud.AddPoint(s.positions[i * 3], s.positions[i * 3 + 1], s.positions[i * 3 + 2],
                     s.directions[i * 3], s.directions[i * 3 + 1],
                     s.directions[i * 3 + 2], si);
    }
  }
  return cloud;
}

}  // namespace strand_io

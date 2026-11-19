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
#include <limits>
#include <stdexcept>

#include "common/logger.h"
#include "common/ply_io.h"

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
  try {
    return ply_io::WritePointCloud(path, StrandsToPointCloud(strands), /*binary=*/true,
                                   /*include_strand_labels=*/true);
  } catch (const std::exception& e) {
    LOG_ERROR("Cannot export strands to %s: %s", path.c_str(), e.what());
    return false;
  }
}

PointCloud StrandsToPointCloud(const std::vector<Strand>& strands) {
  if (strands.size() > static_cast<size_t>(std::numeric_limits<int>::max()) + 1)
    throw std::overflow_error("Strand index exceeds the PointCloud label range");

  size_t num_points = 0;
  for (const auto& strand : strands) {
    if (strand.positions.size() % 3 != 0 ||
        strand.directions.size() != strand.positions.size())
      throw std::invalid_argument("Expected a position and direction per strand point");
    num_points += strand.NumPoints();
  }

  PointCloud cloud;
  cloud.Reserve(num_points);
  for (size_t si = 0; si < strands.size(); ++si) {
    const auto& strand = strands[si];
    cloud.positions.insert(cloud.positions.end(), strand.positions.begin(),
                           strand.positions.end());
    cloud.directions.insert(cloud.directions.end(), strand.directions.begin(),
                            strand.directions.end());
    cloud.labels.insert(cloud.labels.end(), strand.NumPoints(), static_cast<int>(si));
  }
  return cloud;
}

}  // namespace strand_io

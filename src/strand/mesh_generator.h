// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#pragma once

#include <string>
#include <vector>

#include "strand/strand_io.h"

namespace mesh_generator {

// Generate cylinder mesh from strands.
// Each strand segment becomes a cylinder with the given radius.
void GenerateCylinderMesh(const std::vector<Strand>& strands, float radius,
                          int radial_segments, std::vector<float>& vertices,
                          std::vector<float>& normals, std::vector<int>& faces);

// Write cylinder mesh directly to PLY.
void WriteCylinderMesh(const std::string& path, const std::vector<Strand>& strands,
                       float radius = 0.1f, int radial_segments = 3);

}  // namespace mesh_generator

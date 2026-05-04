// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/mesh_generator.h"

#include <cmath>

#include "common/logger.h"
#include "common/ply_io.h"
#include "common/timer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace mesh_generator {

namespace {

void Cross(const float a[3], const float b[3], float out[3]) {
  out[0] = a[1] * b[2] - a[2] * b[1];
  out[1] = a[2] * b[0] - a[0] * b[2];
  out[2] = a[0] * b[1] - a[1] * b[0];
}

void Normalize(float v[3]) {
  float len = std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  if (len > 1e-10f) {
    v[0] /= len;
    v[1] /= len;
    v[2] /= len;
  }
}

// Compute two perpendicular basis vectors for the plane orthogonal to dir.
void GetPerpendicularBasis(const float dir[3], float u[3], float v[3]) {
  float ref[3] = {1, 0, 0};
  float dot = dir[0] * ref[0] + dir[1] * ref[1] + dir[2] * ref[2];
  if (std::abs(dot) > 0.9f) {
    ref[0] = 0;
    ref[1] = 1;
    ref[2] = 0;
  }
  Cross(dir, ref, u);
  Normalize(u);
  Cross(dir, u, v);
  Normalize(v);
}

}  // namespace

void GenerateCylinderMesh(const std::vector<Strand>& strands, float radius,
                          int radial_segments, std::vector<float>& vertices,
                          std::vector<float>& normals, std::vector<int>& faces) {
  vertices.clear();
  normals.clear();
  faces.clear();

  for (const auto& strand : strands) {
    size_t n = strand.NumPoints();
    if (n < 2)
      continue;

    int base_vert = static_cast<int>(vertices.size() / 3);

    // Create a ring of vertices around each strand point
    for (size_t i = 0; i < n; ++i) {
      const float* pos = &strand.positions[i * 3];
      const float* dir = &strand.directions[i * 3];

      float u[3], v[3];
      GetPerpendicularBasis(dir, u, v);

      for (int r = 0; r < radial_segments; ++r) {
        float angle = 2.0f * static_cast<float>(M_PI) * r / radial_segments;
        float c = std::cos(angle);
        float s = std::sin(angle);

        float nx = c * u[0] + s * v[0];
        float ny = c * u[1] + s * v[1];
        float nz = c * u[2] + s * v[2];

        vertices.push_back(pos[0] + radius * nx);
        vertices.push_back(pos[1] + radius * ny);
        vertices.push_back(pos[2] + radius * nz);
        normals.push_back(nx);
        normals.push_back(ny);
        normals.push_back(nz);
      }
    }

    // Connect consecutive rings with triangles
    for (size_t i = 0; i < n - 1; ++i) {
      int ring0 = base_vert + static_cast<int>(i) * radial_segments;
      int ring1 = ring0 + radial_segments;

      for (int r = 0; r < radial_segments; ++r) {
        int r_next = (r + 1) % radial_segments;
        // Two triangles per quad
        faces.push_back(ring0 + r);
        faces.push_back(ring1 + r);
        faces.push_back(ring1 + r_next);

        faces.push_back(ring0 + r);
        faces.push_back(ring1 + r_next);
        faces.push_back(ring0 + r_next);
      }
    }
  }
}

void WriteCylinderMesh(const std::string& path, const std::vector<Strand>& strands,
                       float radius, int radial_segments) {
  ScopedTimer timer("Mesh Generation");

  std::vector<float> vertices, normals_vec;
  std::vector<int> faces;
  GenerateCylinderMesh(strands, radius, radial_segments, vertices, normals_vec, faces);

  LOG_INFO("Cylinder mesh: %zu vertices, %zu faces", vertices.size() / 3,
           faces.size() / 3);

  ply_io::WriteMesh(path, vertices, normals_vec, faces);
}

}  // namespace mesh_generator

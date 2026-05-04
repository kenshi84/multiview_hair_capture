// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/ply_io.h"

#include <cstdio>
#include <cstring>

#include "common/logger.h"

namespace ply_io {

static bool IsLittleEndian() {
  int x = 1;
  return *reinterpret_cast<char*>(&x) == 1;
}

static void SwapBytes4(void* p) {
  unsigned char* b = static_cast<unsigned char*>(p);
  unsigned char t;
  t = b[0];
  b[0] = b[3];
  b[3] = t;
  t = b[1];
  b[1] = b[2];
  b[2] = t;
}

bool ReadPointCloud(const std::string& path, PointCloud& cloud) {
  FILE* f = fopen(path.c_str(), "rb");
  if (!f) {
    LOG_ERROR("Cannot open PLY: %s", path.c_str());
    return false;
  }

  char buf[1024];
  int nverts = 0;
  bool binary = false;
  bool big_endian = false;
  bool has_normals = false;
  int vert_len = 0;
  bool in_vertex_section = false;

  while (fgets(buf, sizeof(buf), f)) {
    if (strncmp(buf, "end_header", 10) == 0)
      break;
    if (strncmp(buf, "format binary_big_endian", 24) == 0) {
      binary = true;
      big_endian = true;
    } else if (strncmp(buf, "format binary_little_endian", 27) == 0) {
      binary = true;
      big_endian = false;
    }
    if (sscanf(buf, "element vertex %d", &nverts) == 1) {
      in_vertex_section = true;
      continue;
    }
    if (strncmp(buf, "element ", 8) == 0 &&
        sscanf(buf, "element vertex %d", &nverts) != 1) {
      in_vertex_section = false;
    }
    if (in_vertex_section) {
      if (strncmp(buf, "property float nx", 17) == 0 ||
          strncmp(buf, "property float normal_x", 23) == 0) {
        has_normals = true;
      }
      if (strncmp(buf, "property float", 14) == 0)
        vert_len++;
      if (strncmp(buf, "property uint", 13) == 0 ||
          strncmp(buf, "property int", 12) == 0)
        vert_len++;
    }
  }

  bool need_swap = binary && big_endian && IsLittleEndian();

  cloud.Clear();
  cloud.Reserve(nverts);

  for (int i = 0; i < nverts; ++i) {
    float x, y, z, nx = 0, ny = 0, nz = 0;
    if (binary) {
      float vals[16];
      if (fread(vals, sizeof(float), vert_len, f) != static_cast<size_t>(vert_len)) {
        fclose(f);
        return false;
      }
      if (need_swap) {
        for (int j = 0; j < vert_len; ++j)
          SwapBytes4(&vals[j]);
      }
      x = vals[0];
      y = vals[1];
      z = vals[2];
      if (has_normals && vert_len >= 6) {
        nx = vals[3];
        ny = vals[4];
        nz = vals[5];
      }
    } else {
      if (has_normals) {
        if (fscanf(f, "%f %f %f %f %f %f", &x, &y, &z, &nx, &ny, &nz) < 6) {
          char rest[1024];
          if (fgets(rest, sizeof(rest), f)) {
          }
        }
      } else {
        if (fscanf(f, "%f %f %f", &x, &y, &z) < 3) {
        }
        char rest[1024];
        if (fgets(rest, sizeof(rest), f)) {
        }
      }
    }
    cloud.AddPoint(x, y, z, nx, ny, nz);
  }

  fclose(f);
  LOG_INFO("Read %zu points from %s (binary=%d, big_endian=%d, %d props/vert)",
           cloud.NumPoints(), path.c_str(), binary, big_endian, vert_len);
  return true;
}

bool WritePointCloud(const std::string& path, const PointCloud& cloud, bool binary) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    LOG_ERROR("Cannot create PLY: %s", path.c_str());
    return false;
  }

  size_t n = cloud.NumPoints();
  bool has_dirs = !cloud.directions.empty();

  fprintf(f, "ply\n");
  if (binary) {
    fprintf(f, "format binary_little_endian 1.0\n");
  } else {
    fprintf(f, "format ascii 1.0\n");
  }
  fprintf(f, "element vertex %zu\n", n);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  if (has_dirs) {
    fprintf(f, "property float nx\n");
    fprintf(f, "property float ny\n");
    fprintf(f, "property float nz\n");
  }
  fprintf(f, "end_header\n");

  for (size_t i = 0; i < n; ++i) {
    float x = cloud.positions[i * 3];
    float y = cloud.positions[i * 3 + 1];
    float z = cloud.positions[i * 3 + 2];
    if (binary) {
      fwrite(&x, sizeof(float), 1, f);
      fwrite(&y, sizeof(float), 1, f);
      fwrite(&z, sizeof(float), 1, f);
      if (has_dirs) {
        float dx = cloud.directions[i * 3];
        float dy = cloud.directions[i * 3 + 1];
        float dz = cloud.directions[i * 3 + 2];
        fwrite(&dx, sizeof(float), 1, f);
        fwrite(&dy, sizeof(float), 1, f);
        fwrite(&dz, sizeof(float), 1, f);
      }
    } else {
      if (has_dirs) {
        fprintf(f, "%f %f %f %f %f %f\n", x, y, z, cloud.directions[i * 3],
                cloud.directions[i * 3 + 1], cloud.directions[i * 3 + 2]);
      } else {
        fprintf(f, "%f %f %f\n", x, y, z);
      }
    }
  }

  fclose(f);
  LOG_INFO("Wrote %zu points to %s", n, path.c_str());
  return true;
}

bool WriteMesh(const std::string& path, const std::vector<float>& vertices,
               const std::vector<float>& normals, const std::vector<int>& faces,
               bool binary) {
  FILE* f = fopen(path.c_str(), "wb");
  if (!f)
    return false;

  size_t nv = vertices.size() / 3;
  size_t nf = faces.size() / 3;
  bool has_normals = !normals.empty();

  fprintf(f, "ply\n");
  if (binary) {
    fprintf(f, "format binary_little_endian 1.0\n");
  } else {
    fprintf(f, "format ascii 1.0\n");
  }
  fprintf(f, "element vertex %zu\n", nv);
  fprintf(f, "property float x\n");
  fprintf(f, "property float y\n");
  fprintf(f, "property float z\n");
  if (has_normals) {
    fprintf(f, "property float nx\n");
    fprintf(f, "property float ny\n");
    fprintf(f, "property float nz\n");
  }
  fprintf(f, "element face %zu\n", nf);
  fprintf(f, "property list uchar int vertex_indices\n");
  fprintf(f, "end_header\n");

  if (binary) {
    for (size_t i = 0; i < nv; ++i) {
      fwrite(&vertices[i * 3], sizeof(float), 3, f);
      if (has_normals)
        fwrite(&normals[i * 3], sizeof(float), 3, f);
    }
    for (size_t i = 0; i < nf; ++i) {
      unsigned char three = 3;
      fwrite(&three, 1, 1, f);
      fwrite(&faces[i * 3], sizeof(int), 3, f);
    }
  } else {
    for (size_t i = 0; i < nv; ++i) {
      if (has_normals) {
        fprintf(f, "%f %f %f %f %f %f\n", vertices[i * 3], vertices[i * 3 + 1],
                vertices[i * 3 + 2], normals[i * 3], normals[i * 3 + 1],
                normals[i * 3 + 2]);
      } else {
        fprintf(f, "%f %f %f\n", vertices[i * 3], vertices[i * 3 + 1],
                vertices[i * 3 + 2]);
      }
    }
    for (size_t i = 0; i < nf; ++i) {
      fprintf(f, "3 %d %d %d\n", faces[i * 3], faces[i * 3 + 1], faces[i * 3 + 2]);
    }
  }

  fclose(f);
  LOG_INFO("Wrote mesh (%zu verts, %zu faces) to %s", nv, nf, path.c_str());
  return true;
}

}  // namespace ply_io

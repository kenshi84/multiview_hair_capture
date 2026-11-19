// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "common/ply_io.h"

#include <array>
#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <utility>

#include "common/boys2rgb.h"
#include "common/logger.h"
#include "happly.h"

namespace ply_io {
namespace {

using PropertyNames = std::array<const char*, 3>;
constexpr PropertyNames kPositions{"x", "y", "z"};
constexpr PropertyNames kNormals{"nx", "ny", "nz"};

std::vector<float> ReadTriples(happly::Element& vertices, const PropertyNames& names) {
  std::vector<float> values(3 * vertices.count);
  for (size_t axis = 0; axis < names.size(); ++axis) {
    const char* name = names[axis];
    auto copy_component = [&](const auto& component) {
      for (size_t i = 0; i < vertices.count; ++i)
        values[3 * i + axis] = static_cast<float>(component[i]);
    };
    // hapPLY promotes within each of the floating-point and integer type families.
    const std::string type = vertices.getPropertyPtr(name)->propertyTypeName();
    if (type == "float" || type == "double")
      copy_component(vertices.getProperty<double>(name));
    else if (type == "char" || type == "short" || type == "int")
      copy_component(vertices.getProperty<std::int32_t>(name));
    else
      copy_component(vertices.getProperty<std::uint32_t>(name));
  }
  return values;
}

template <typename T>
void AddTriples(happly::Element& vertices, const PropertyNames& names,
                const std::vector<T>& values) {
  if (values.size() != 3 * vertices.count)
    throw std::invalid_argument(std::string("Expected one triple per vertex for ") +
                                names[0]);

  std::vector<T> component(vertices.count);
  for (size_t axis = 0; axis < names.size(); ++axis) {
    for (size_t i = 0; i < vertices.count; ++i)
      component[i] = values[3 * i + axis];
    vertices.addProperty(names[axis], component);
  }
}

happly::Element& AddVertices(happly::PLYData& ply, const std::vector<float>& positions) {
  ply.addElement("vertex", positions.size() / 3);
  auto& vertices = ply.getElement("vertex");
  AddTriples(vertices, kPositions, positions);
  return vertices;
}

void WriteFile(const std::string& path, happly::PLYData& ply, bool binary) {
  std::ofstream output(path, std::ios::binary);
  output.exceptions(std::ios::failbit | std::ios::badbit);
  ply.write(output, binary ? happly::DataFormat::Binary : happly::DataFormat::ASCII);
  output.close();
}

}  // namespace

bool ReadPointCloud(const std::string& path, PointCloud& cloud) {
  try {
    std::ifstream input(path, std::ios::binary);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    happly::PLYData ply(input);
    ply.validate();
    auto& vertices = ply.getElement("vertex");

    PointCloud result;
    result.positions = ReadTriples(vertices, kPositions);
    if (vertices.hasProperty("nx")) {
      result.directions = ReadTriples(vertices, kNormals);
    } else if (vertices.hasProperty("normal_x")) {
      result.directions = ReadTriples(vertices, {"normal_x", "normal_y", "normal_z"});
    } else {
      result.directions.resize(result.positions.size(), 0.0f);
    }
    result.labels.resize(vertices.count, 0);
    cloud = std::move(result);
    LOG_INFO("Read %zu points from %s", cloud.NumPoints(), path.c_str());
    return true;
  } catch (const std::exception& e) {
    LOG_ERROR("Cannot read PLY %s: %s", path.c_str(), e.what());
    return false;
  }
}

bool WritePointCloud(const std::string& path, const PointCloud& cloud, bool binary,
                     bool include_strand_labels) {
  try {
    happly::PLYData ply;
    auto& vertices = AddVertices(ply, cloud.positions);
    // Empty strand exports declare the same vertex properties as populated ones.
    if (!cloud.directions.empty() || include_strand_labels) {
      AddTriples(vertices, kNormals, cloud.directions);
      if (include_strand_labels) {
        std::vector<std::uint32_t> labels(cloud.labels.begin(), cloud.labels.end());
        vertices.addProperty("label", labels);
      }
      std::vector<unsigned char> colors(cloud.directions.size());
      Boys2RGB(colors.data(), cloud.directions.data(), vertices.count);
      AddTriples(vertices, {"red", "green", "blue"}, colors);
    }

    WriteFile(path, ply, binary);
    LOG_INFO("Wrote %zu points to %s", vertices.count, path.c_str());
    return true;
  } catch (const std::exception& e) {
    LOG_ERROR("Cannot write point cloud %s: %s", path.c_str(), e.what());
    return false;
  }
}

bool WriteMesh(const std::string& path, const std::vector<float>& vertices,
               const std::vector<float>& normals, const std::vector<int>& faces,
               bool binary) {
  try {
    if (faces.size() % 3 != 0)
      throw std::invalid_argument("Expected three vertex indices per triangle");

    happly::PLYData ply;
    auto& mesh_vertices = AddVertices(ply, vertices);
    if (!normals.empty())
      AddTriples(mesh_vertices, kNormals, normals);
    const size_t num_faces = faces.size() / 3;
    ply.addElement("face", num_faces);

    // Use flat list storage to avoid a heap allocation per triangle.
    auto indices = std::make_unique<happly::TypedListProperty<int>>("vertex_indices", 1);
    indices->flattenedData = faces;
    indices->flattenedIndexStart.reserve(num_faces + 1);
    for (size_t i = 1; i <= num_faces; ++i)
      indices->flattenedIndexStart.push_back(3 * i);
    ply.getElement("face").properties.push_back(std::move(indices));

    WriteFile(path, ply, binary);
    LOG_INFO("Wrote mesh (%zu verts, %zu faces) to %s", vertices.size() / 3, num_faces,
             path.c_str());
    return true;
  } catch (const std::exception& e) {
    LOG_ERROR("Cannot write mesh %s: %s", path.c_str(), e.what());
    return false;
  }
}

}  // namespace ply_io

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "strand/meanshift.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include <omp.h>

#include "common/cuda_check.h"
#include "common/logger.h"
#include "common/timer.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// GPU kernel launchers (defined in meanshift.cu)
void launchMeanShiftGpu(const float* d_positions, const float* d_directions,
                        float* d_out_positions, float* d_out_directions,
                        const int2* d_grid_idx, const int* d_grid_data,
                        const int* d_point_cells, int rx, int ry, int rz,
                        int num_points, int min_neighbors, float nei_radius,
                        float sigma_pos, float sigma_orient, float convergence,
                        int max_iter);

void launchMeanShiftGpuRange(const float* d_positions, const float* d_directions,
                             float* d_out_positions, float* d_out_directions,
                             const int2* d_grid_idx, const int* d_grid_data,
                             const int* d_point_cells, int rx, int ry, int rz,
                             int num_points, int point_start, int point_count,
                             int min_neighbors, float nei_radius, float sigma_pos,
                             float sigma_orient, float convergence, int max_iter);

namespace meanshift {

// Build GPU VoxelGrid: d_grid_idx (per-cell start+count) + d_grid_data (point indices)
struct GpuVoxelGrid {
  std::vector<int2> grid_idx;    // per-cell: (start, count) into grid_data
  std::vector<int> grid_data;    // point indices sorted by cell
  std::vector<int> point_cells;  // cell index per point
  int rx, ry, rz;
  float origin[3];
  float voxel_len;

  void Build(const float* positions, size_t n, float radius) {
    voxel_len = radius;

    float minp[3] = {1e30f, 1e30f, 1e30f};
    float maxp[3] = {-1e30f, -1e30f, -1e30f};
    for (size_t i = 0; i < n; i++) {
      for (int d = 0; d < 3; d++) {
        float v = positions[i * 3 + d];
        if (v < minp[d])
          minp[d] = v;
        if (v > maxp[d])
          maxp[d] = v;
      }
    }

    for (int d = 0; d < 3; d++) {
      origin[d] = minp[d] - voxel_len;
    }
    rx = static_cast<int>(std::ceil((maxp[0] + voxel_len - origin[0]) / voxel_len));
    ry = static_cast<int>(std::ceil((maxp[1] + voxel_len - origin[1]) / voxel_len));
    rz = static_cast<int>(std::ceil((maxp[2] + voxel_len - origin[2]) / voxel_len));

    size_t total_cells = static_cast<size_t>(rx) * ry * rz;
    LOG_INFO("  VoxelGrid: %d x %d x %d = %zu cells (voxel=%.2f mm)", rx, ry, rz,
             total_cells, voxel_len);

    // Count points per cell
    std::vector<int> cell_counts(total_cells, 0);
    point_cells.resize(n);
    for (size_t i = 0; i < n; i++) {
      int ix = static_cast<int>(std::floor((positions[i * 3] - origin[0]) / voxel_len));
      int iy =
          static_cast<int>(std::floor((positions[i * 3 + 1] - origin[1]) / voxel_len));
      int iz =
          static_cast<int>(std::floor((positions[i * 3 + 2] - origin[2]) / voxel_len));
      ix = std::max(0, std::min(ix, rx - 1));
      iy = std::max(0, std::min(iy, ry - 1));
      iz = std::max(0, std::min(iz, rz - 1));
      int cell = iz * rx * ry + iy * rx + ix;
      point_cells[i] = cell;
      cell_counts[cell]++;
    }

    // Compute prefix sums for grid_idx
    grid_idx.resize(total_cells);
    int offset = 0;
    for (size_t c = 0; c < total_cells; c++) {
      grid_idx[c].x = offset;
      grid_idx[c].y = cell_counts[c];
      offset += cell_counts[c];
    }

    // Fill grid_data
    grid_data.resize(n);
    std::vector<int> cell_fill(total_cells, 0);
    for (size_t i = 0; i < n; i++) {
      int cell = point_cells[i];
      int pos = grid_idx[cell].x + cell_fill[cell];
      grid_data[pos] = static_cast<int>(i);
      cell_fill[cell]++;
    }
  }
};

PointCloud RunCuda(const PointCloud& input, const Config& config, int gpu_id,
                   int num_gpus) {
  ScopedTimer timer("Mean-shift");

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);
  if (err != cudaSuccess || device_count == 0) {
    LOG_ERROR("Mean-shift: no GPU available");
    return input;
  }
  num_gpus = std::min(num_gpus, device_count);
  if (num_gpus < 1)
    num_gpus = 1;

  size_t n = input.NumPoints();
  LOG_INFO("Mean-shift: %zu input points, %d GPU(s)", n, num_gpus);

  float radius = config.meanshift_neighbor_radius;
  int min_nei = config.meanshift_min_neighbors;
  float sigma_pos = config.meanshift_sigma_position;
  float sigma_orient =
      config.meanshift_sigma_orientation * static_cast<float>(M_PI) / 180.0f;
  float convergence = config.meanshift_convergence;
  int max_iter = config.meanshift_max_iterations;

  // Build VoxelGrid on CPU (shared across all GPUs)
  Timer vg_timer;
  GpuVoxelGrid grid;
  grid.Build(input.positions.data(), n, radius);
  LOG_INFO("  VoxelGrid built in %.1f s", vg_timer.ElapsedSeconds());

  size_t total_cells = static_cast<size_t>(grid.rx) * grid.ry * grid.rz;

  // Output buffer (host)
  PointCloud output;
  output.positions.resize(n * 3, 0.f);
  output.directions.resize(n * 3, 0.f);
  output.labels.resize(n, 0);

  Timer kernel_timer;

  if (num_gpus == 1) {
    // --- Single GPU ---
    CUDA_CHECK(cudaSetDevice(gpu_id));
    cudaGetLastError();  // Clear any sticky error from a prior pipeline stage
                         // (CUDA's last-error state is per-thread)

    float *d_pos, *d_dir, *d_out_pos, *d_out_dir;
    int2* d_grid_idx;
    int *d_grid_data, *d_point_cells;

    CUDA_CHECK(cudaMalloc(&d_pos, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dir, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_pos, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_dir, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_pos, 0, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out_dir, 0, n * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grid_idx, total_cells * sizeof(int2)));
    CUDA_CHECK(cudaMalloc(&d_grid_data, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_point_cells, n * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_pos, input.positions.data(), n * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dir, input.directions.data(), n * 3 * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_idx, grid.grid_idx.data(), total_cells * sizeof(int2),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_grid_data, grid.grid_data.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_point_cells, grid.point_cells.data(), n * sizeof(int),
                          cudaMemcpyHostToDevice));

    launchMeanShiftGpu(d_pos, d_dir, d_out_pos, d_out_dir, d_grid_idx, d_grid_data,
                       d_point_cells, grid.rx, grid.ry, grid.rz, static_cast<int>(n),
                       min_nei, radius, sigma_pos, sigma_orient, convergence, max_iter);

    CUDA_CHECK(cudaMemcpy(output.positions.data(), d_out_pos, n * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(output.directions.data(), d_out_dir, n * 3 * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_pos);
    cudaFree(d_dir);
    cudaFree(d_out_pos);
    cudaFree(d_out_dir);
    cudaFree(d_grid_idx);
    cudaFree(d_grid_data);
    cudaFree(d_point_cells);

  } else {
    // --- Multi-GPU: split points across GPUs, each gets full grid ---
    int chunk = static_cast<int>((n + num_gpus - 1) / num_gpus);
    LOG_INFO("  Multi-GPU: %d GPUs, ~%d points/GPU", num_gpus, chunk);

#pragma omp parallel num_threads(num_gpus)
    {
      int g = omp_get_thread_num();
      int start = g * chunk;
      int count = std::min(chunk, static_cast<int>(n) - start);
      if (count > 0) {
        CUDA_CHECK(cudaSetDevice(gpu_id + g));
        cudaGetLastError();  // Clear any sticky error from a prior pipeline stage
                             // (CUDA's last-error state is per-thread, and OpenMP
                             // reuses worker threads across parallel regions)

        float *d_pos, *d_dir, *d_out_pos, *d_out_dir;
        int2* d_grid_idx;
        int *d_grid_data, *d_point_cells;

        // Allocate: full positions/directions/grid for neighbor lookups,
        // full output buffer (only our chunk will be written)
        CUDA_CHECK(cudaMalloc(&d_pos, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_dir, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_pos, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_out_dir, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out_pos, 0, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMemset(d_out_dir, 0, n * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_grid_idx, total_cells * sizeof(int2)));
        CUDA_CHECK(cudaMalloc(&d_grid_data, n * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&d_point_cells, n * sizeof(int)));

        CUDA_CHECK(cudaMemcpy(d_pos, input.positions.data(), n * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_dir, input.directions.data(), n * 3 * sizeof(float),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid_idx, grid.grid_idx.data(),
                              total_cells * sizeof(int2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_grid_data, grid.grid_data.data(), n * sizeof(int),
                              cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_point_cells, grid.point_cells.data(), n * sizeof(int),
                              cudaMemcpyHostToDevice));

        launchMeanShiftGpuRange(d_pos, d_dir, d_out_pos, d_out_dir, d_grid_idx,
                                d_grid_data, d_point_cells, grid.rx, grid.ry, grid.rz,
                                static_cast<int>(n), start, count, min_nei, radius,
                                sigma_pos, sigma_orient, convergence, max_iter);

        // Download only our chunk
        CUDA_CHECK(cudaMemcpy(output.positions.data() + start * 3,
                              d_out_pos + start * 3, count * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(output.directions.data() + start * 3,
                              d_out_dir + start * 3, count * 3 * sizeof(float),
                              cudaMemcpyDeviceToHost));

        cudaFree(d_pos);
        cudaFree(d_dir);
        cudaFree(d_out_pos);
        cudaFree(d_out_dir);
        cudaFree(d_grid_idx);
        cudaFree(d_grid_data);
        cudaFree(d_point_cells);
      }
    }
  }

  LOG_INFO("  Kernel done in %.1f s", kernel_timer.ElapsedSeconds());

  // Remove zero vertices
  PointCloud filtered;
  for (size_t i = 0; i < n; i++) {
    float px = output.positions[i * 3];
    float py = output.positions[i * 3 + 1];
    float pz = output.positions[i * 3 + 2];
    if (px == 0.0f && py == 0.0f && pz == 0.0f)
      continue;
    filtered.AddPoint(px, py, pz, output.directions[i * 3],
                      output.directions[i * 3 + 1], output.directions[i * 3 + 2], 0);
  }

  LOG_INFO("Mean-shift: %zu -> %zu points", n, filtered.NumPoints());
  return filtered;
}

}  // namespace meanshift

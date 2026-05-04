// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

// GPU-native mean-shift with on-the-fly VoxelGrid neighbor search.
// Each CUDA block processes one point. Threads cooperatively collect
// neighbors from the VoxelGrid, then iterate mean-shift with
// block-level reduction.

#include <cuda_runtime.h>

#include "common/cuda_check.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

#define BLOCK_SIZE 128
#define MAX_ITEMS_PER_THREAD 512

// ---------- Device helpers ----------

__device__ float gaussian_d(float x, float sigma) {
  return expf(-0.5f * x * x / (sigma * sigma));
}

__device__ float dir_distance_d(float3 d1, float3 d2) {
  float dot = d1.x * d2.x + d1.y * d2.y + d1.z * d2.z;
  float abs_dot = fminf(1.0f, fabsf(dot));
  return acosf(abs_dot);  // radians
}

__device__ float length3(float3 v) {
  return sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
}

__device__ float distance3(float3 a, float3 b) {
  float3 d = make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
  return length3(d);
}

__device__ float3 normalize3(float3 v) {
  float len = length3(v);
  if (len > 1e-10f) {
    v.x /= len;
    v.y /= len;
    v.z /= len;
  }
  return v;
}

// Line-plane intersection: find where line (lp, ld) crosses plane (pp, pn).
// Returns intersection position. Sets valid=false if parallel.
__device__ float3 linePlaneIntersect(float3 lp, float3 ld, float3 pp, float3 pn,
                                     bool& valid) {
  float nu = pn.x * ld.x + pn.y * ld.y + pn.z * ld.z;
  if (fabsf(nu) < 1e-16f) {
    valid = false;
    return make_float3(0.f, 0.f, 0.f);
  }
  float wx = lp.x - pp.x, wy = lp.y - pp.y, wz = lp.z - pp.z;
  float s = -(pn.x * wx + pn.y * wy + pn.z * wz) / nu;
  valid = true;
  return make_float3(lp.x + s * ld.x, lp.y + s * ld.y, lp.z + s * ld.z);
}

// ---------- GPU VoxelGrid search ----------

// Get 3D voxel indices from linear index
__device__ void linearToVoxel(int idx, int rx, int ry, int& ix, int& iy, int& iz) {
  ix = idx % rx;
  iy = (idx / rx) % ry;
  iz = idx / (rx * ry);
}

// Get linear index from 3D voxel indices
__device__ int voxelToLinear(int ix, int iy, int iz, int rx, int ry) {
  return iz * rx * ry + iy * rx + ix;
}

// Collect neighbor points from VoxelGrid into per-thread arrays.
// Threads distribute neighbors in round-robin across the block.
__device__ void collectNeighbors(
    const float* __restrict__ positions, const float* __restrict__ directions,
    float3 pt_pos, float3 pt_dir,
    const int2* __restrict__ grid_idx,  // [cell] -> (start, count)
    const int* __restrict__ grid_data,  // point indices sorted by cell
    int grid_cell_idx,                  // current point's cell index
    int rx, int ry, int rz, float nei_radius, int tid, int B,
    float3* nei_pos_out,  // per-thread array
    float3* nei_dir_out,  // per-thread array
    int* n_items) {
  *n_items = 0;

  // Get 3D voxel coords of current cell
  int cx, cy, cz;
  linearToVoxel(grid_cell_idx, rx, ry, cx, cy, cz);

  int global_nei_idx = 0;  // counts all valid neighbors across all cells

  // Iterate 27 neighbor cells
  for (int dz = -1; dz <= 1; dz++) {
    int nz = cz + dz;
    if (nz < 0 || nz >= rz)
      continue;
    for (int dy = -1; dy <= 1; dy++) {
      int ny = cy + dy;
      if (ny < 0 || ny >= ry)
        continue;
      for (int dx = -1; dx <= 1; dx++) {
        int nx = cx + dx;
        if (nx < 0 || nx >= rx)
          continue;

        int cell = voxelToLinear(nx, ny, nz, rx, ry);
        int start = grid_idx[cell].x;
        int count = grid_idx[cell].y;

        for (int j = start; j < start + count; j++) {
          if (*n_items >= MAX_ITEMS_PER_THREAD)
            return;

          int ptidx = grid_data[j];
          float3 npos = make_float3(positions[ptidx * 3], positions[ptidx * 3 + 1],
                                    positions[ptidx * 3 + 2]);

          float dist = distance3(pt_pos, npos);
          if (dist < nei_radius) {
            // Round-robin: this neighbor goes to thread (global_nei_idx % B)
            if (global_nei_idx % B == tid) {
              float3 ndir =
                  make_float3(directions[ptidx * 3], directions[ptidx * 3 + 1],
                              directions[ptidx * 3 + 2]);
              // Flip direction to match current point
              float dot = pt_dir.x * ndir.x + pt_dir.y * ndir.y + pt_dir.z * ndir.z;
              if (dot < 0.f) {
                ndir.x = -ndir.x;
                ndir.y = -ndir.y;
                ndir.z = -ndir.z;
              }
              nei_pos_out[*n_items] = npos;
              nei_dir_out[*n_items] = ndir;
              (*n_items)++;
            }
            global_nei_idx++;
          }
        }
      }
    }
  }
}

// ---------- Block-level reduction ----------

__device__ void blockReduceFloat3(float3& val, volatile float* smem, int tid) {
  smem[tid] = val.x;
  smem[tid + BLOCK_SIZE] = val.y;
  smem[tid + 2 * BLOCK_SIZE] = val.z;
  __syncthreads();

  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s) {
      smem[tid] += smem[tid + s];
      smem[tid + BLOCK_SIZE] += smem[tid + BLOCK_SIZE + s];
      smem[tid + 2 * BLOCK_SIZE] += smem[tid + 2 * BLOCK_SIZE + s];
    }
    __syncthreads();
  }

  val.x = smem[0];
  val.y = smem[BLOCK_SIZE];
  val.z = smem[2 * BLOCK_SIZE];
}

__device__ float blockReduceFloat(float val, volatile float* smem, int tid) {
  smem[tid] = val;
  __syncthreads();
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  return smem[0];
}

__device__ int blockReduceInt(int val, volatile int* smem, int tid) {
  smem[tid] = val;
  __syncthreads();
  for (int s = BLOCK_SIZE / 2; s > 0; s >>= 1) {
    if (tid < s)
      smem[tid] += smem[tid + s];
    __syncthreads();
  }
  return smem[0];
}

// ---------- Mean-shift kernel (one block per point) ----------

__global__ void k_meanShiftGpu(
    const float* __restrict__ positions, const float* __restrict__ directions,
    float* out_positions, float* out_directions, const int2* __restrict__ grid_idx,
    const int* __restrict__ grid_data,
    const int* __restrict__ point_cells,  // cell index per point
    int rx, int ry, int rz, int num_points, int point_offset, int min_neighbors,
    float nei_radius, float sigma_pos, float sigma_orient, float convergence,
    int max_iter) {
  const int pid = blockIdx.x + point_offset;
  const int tid = threadIdx.x;
  if (pid >= num_points)
    return;

  // Shared memory for reductions
  __shared__ float smem[3 * BLOCK_SIZE];
  __shared__ int ismem[BLOCK_SIZE];
  __shared__ float3 s_pos, s_dir;
  __shared__ float s_shift;
  __shared__ int s_count, s_numnei;

  // Per-thread neighbor storage
  float3 nei_pos[MAX_ITEMS_PER_THREAD];
  float3 nei_dir[MAX_ITEMS_PER_THREAD];
  int n_items = 0;

  // Load initial point
  if (tid == 0) {
    s_pos =
        make_float3(positions[pid * 3], positions[pid * 3 + 1], positions[pid * 3 + 2]);
    s_dir = make_float3(directions[pid * 3], directions[pid * 3 + 1],
                        directions[pid * 3 + 2]);
    s_shift = 1e30f;
    s_count = 0;
  }
  __syncthreads();

  // Collect neighbors (done once, reused across iterations)
  int cell = point_cells[pid];
  collectNeighbors(positions, directions, s_pos, s_dir, grid_idx, grid_data, cell, rx,
                   ry, rz, nei_radius, tid, BLOCK_SIZE, nei_pos, nei_dir, &n_items);
  __syncthreads();

  // Count total neighbors
  int total_nei = blockReduceInt(n_items, (volatile int*) ismem, tid);
  if (tid == 0)
    s_numnei = total_nei;
  __syncthreads();

  // Noise removal
  if (s_numnei < min_neighbors) {
    if (tid == 0) {
      out_positions[pid * 3] = 0.f;
      out_positions[pid * 3 + 1] = 0.f;
      out_positions[pid * 3 + 2] = 0.f;
      out_directions[pid * 3] = 0.f;
      out_directions[pid * 3 + 1] = 0.f;
      out_directions[pid * 3 + 2] = 0.f;
    }
    return;
  }

  // Mean-shift iterations
  while (s_shift > convergence && s_count < max_iter) {
    float3 pt_pos = s_pos;
    float3 pt_dir = s_dir;

    // Each thread computes partial weighted sum over its neighbors
    float3 wsum_pos = make_float3(0.f, 0.f, 0.f);
    float3 wsum_dir = make_float3(0.f, 0.f, 0.f);
    float wsum = 0.f;

    for (int i = 0; i < n_items; i++) {
      // Line-plane intersection
      bool valid;
      float3 ipt = linePlaneIntersect(nei_pos[i], nei_dir[i], pt_pos, pt_dir, valid);
      if (!valid)
        continue;

      // Bilateral weight
      float dist_e = distance3(pt_pos, ipt);
      float dist_o = dir_distance_d(pt_dir, nei_dir[i]);
      float w = gaussian_d(dist_e, sigma_pos) * gaussian_d(dist_o, sigma_orient);

      wsum_pos.x += w * ipt.x;
      wsum_pos.y += w * ipt.y;
      wsum_pos.z += w * ipt.z;
      wsum_dir.x += w * nei_dir[i].x;
      wsum_dir.y += w * nei_dir[i].y;
      wsum_dir.z += w * nei_dir[i].z;
      wsum += w;
    }

    // Block-level reduction
    blockReduceFloat3(wsum_pos, (volatile float*) smem, tid);
    __syncthreads();
    blockReduceFloat3(wsum_dir, (volatile float*) smem, tid);
    __syncthreads();
    wsum = blockReduceFloat(wsum, (volatile float*) smem, tid);
    __syncthreads();

    if (tid == 0) {
      if (wsum > 1e-10f) {
        float3 new_pos =
            make_float3(wsum_pos.x / wsum, wsum_pos.y / wsum, wsum_pos.z / wsum);
        float3 new_dir = normalize3(
            make_float3(wsum_dir.x / wsum, wsum_dir.y / wsum, wsum_dir.z / wsum));
        s_shift = distance3(s_pos, new_pos);
        s_pos = new_pos;
        s_dir = new_dir;
      } else {
        s_shift = 0.f;  // no valid neighbors, stop
      }
      s_count++;
    }
    __syncthreads();
  }

  // Write output
  if (tid == 0) {
    out_positions[pid * 3] = s_pos.x;
    out_positions[pid * 3 + 1] = s_pos.y;
    out_positions[pid * 3 + 2] = s_pos.z;
    out_directions[pid * 3] = s_dir.x;
    out_directions[pid * 3 + 1] = s_dir.y;
    out_directions[pid * 3 + 2] = s_dir.z;
  }
}

// ---------- Host interface ----------

void launchMeanShiftGpu(const float* d_positions, const float* d_directions,
                        float* d_out_positions, float* d_out_directions,
                        const int2* d_grid_idx, const int* d_grid_data,
                        const int* d_point_cells, int rx, int ry, int rz,
                        int num_points, int min_neighbors, float nei_radius,
                        float sigma_pos, float sigma_orient, float convergence,
                        int max_iter) {
  // Launch in batches to avoid exceeding grid limits
  const int batch_size = 65535;
  for (int offset = 0; offset < num_points; offset += batch_size) {
    int n_batch = min(batch_size, num_points - offset);
    k_meanShiftGpu<<<n_batch, BLOCK_SIZE>>>(
        d_positions, d_directions, d_out_positions, d_out_directions, d_grid_idx,
        d_grid_data, d_point_cells, rx, ry, rz, num_points, offset, min_neighbors,
        nei_radius, sigma_pos, sigma_orient, convergence, max_iter);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  CUDA_CHECK_LAST();
}

// Launch mean-shift on a specific range of points [point_start,
// point_start+point_count). Used for multi-GPU: each GPU processes its chunk while
// reading the full dataset.
void launchMeanShiftGpuRange(const float* d_positions, const float* d_directions,
                             float* d_out_positions, float* d_out_directions,
                             const int2* d_grid_idx, const int* d_grid_data,
                             const int* d_point_cells, int rx, int ry, int rz,
                             int num_points, int point_start, int point_count,
                             int min_neighbors, float nei_radius, float sigma_pos,
                             float sigma_orient, float convergence, int max_iter) {
  const int batch_size = 65535;
  for (int offset = point_start; offset < point_start + point_count;
       offset += batch_size) {
    int n_batch = min(batch_size, point_start + point_count - offset);
    k_meanShiftGpu<<<n_batch, BLOCK_SIZE>>>(
        d_positions, d_directions, d_out_positions, d_out_directions, d_grid_idx,
        d_grid_data, d_point_cells, rx, ry, rz, num_points, offset, min_neighbors,
        nei_radius, sigma_pos, sigma_orient, convergence, max_iter);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
  CUDA_CHECK_LAST();
}

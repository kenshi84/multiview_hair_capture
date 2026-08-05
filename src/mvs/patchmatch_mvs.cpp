// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "mvs/patchmatch_mvs.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "common/cuda_check.h"
#include "common/cuda_math.h"
#include "common/image_io.h"
#include "common/logger.h"
#include "common/math_util.h"
#include "common/ply_io.h"
#include "common/point_cloud.h"
#include "common/types.h"
#include "mvs/image_loader.h"
#include "mvs/patchmatch/hierarchical.h"
#include "mvs/patchmatch/image_ops.h"
#include "mvs/patchmatch/params.h"

PatchMatchMvs::PatchMatchMvs(const Config& config, const CameraArray& cameras)
    : config_(config), cameras_(cameras) {
  view_results_.resize(cameras_.NumCameras());
}

PatchMatchMvs::~PatchMatchMvs() = default;

void PatchMatchMvs::Run() {
  int num_cameras = cameras_.NumCameras();
  int num_gpus = config_.num_gpus;

  if (config_.use_mask && config_.mask_dir.empty()) {
    LOG_ERROR("use_mask is enabled but data.mask_dir is empty");
    return;
  }

  LOG_INFO("PatchMatchMvs: processing %d reference views on %d GPU(s)", num_cameras,
           num_gpus);

#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic) num_threads(num_gpus)
#endif
  for (int i = 0; i < num_cameras; i++) {
    int gpu_id = config_.gpu_id;
#ifdef _OPENMP
    gpu_id = config_.gpu_id + omp_get_thread_num() % num_gpus;
#endif
    ProcessReferenceView(i, gpu_id);
  }

  LOG_INFO("PatchMatchMvs: all views processed");
}

void PatchMatchMvs::ProcessReferenceView(int ref_index, int gpu_id) {
  // Declare GPU pointers before try so catch can clean them up
  float* d_refMap = nullptr;
  Line3D* d_lineMap = nullptr;
  float* d_orient2D = nullptr;
  float* d_variance = nullptr;
  float* d_cost = nullptr;
  std::vector<float*> d_neiMaps;
  std::vector<cudaTextureObject_t> h_neiTexObjs;
  unsigned char* d_refMask = nullptr;
  unsigned char** d_neiMaskPtrs = nullptr;
  std::vector<unsigned char*> d_neiMaskBuffers;
  int numnei = 0;

  try {
    cudaError_t set_err = cudaSetDevice(gpu_id);
    if (set_err != cudaSuccess) {
      LOG_ERROR("Failed to set GPU %d: %s", gpu_id, cudaGetErrorString(set_err));
      return;
    }
    cudaGetLastError();  // Clear any prior errors

    unsigned int ref_cam_id = cameras_.GetCamId(ref_index);
    LOG_INFO("Processing reference view %d (cam_id=%u) on GPU %d", ref_index,
             ref_cam_id, gpu_id);

    // 1. Select neighbor views
    std::vector<int> nei_indices = cameras_.SelectNeighborViews(
        ref_index, config_.min_angle, config_.max_angle, config_.num_neighbor_views);
    if (nei_indices.empty()) {
      LOG_WARN("No neighbor views found for ref %d, skipping", ref_index);
      return;
    }
    numnei = static_cast<int>(nei_indices.size());
    LOG_DEBUG("  %d neighbor views selected", numnei);

    // 2. Load reference image
    std::string ref_path = image_loader::FormatPath(config_.image_dir, ref_cam_id);
    Image ref_img = image_loader::LoadGrayImage(ref_path, config_.downsample);
    if (ref_img.Empty()) {
      LOG_ERROR("Failed to load reference image: %s", ref_path.c_str());
      return;
    }

    // 3. Build GpuCamera for reference
    const Camera& ref_cam = cameras_.GetCamera(ref_index);
    GpuCamera ref_gpu = ref_cam.BuildGpuCamera();

    // Undistort reference image if using distorted inputs
    if (config_.distorted_images) {
      image_loader::UndistortImage(ref_img, ref_gpu.K, ref_gpu.dist);
    }

    unsigned int width = ref_img.width;
    unsigned int height = ref_img.height;
    unsigned int npix = width * height;

    // Build relative neighbor cameras (centered at reference)
    std::vector<GpuCamera> nei_gpu_cams(numnei);
    Mat3x3f ref_R_host;
    memcpy(ref_R_host.Ptr(), ref_gpu.R, 9 * sizeof(float));
    Vec3f ref_t_host(ref_gpu.t[0], ref_gpu.t[1], ref_gpu.t[2]);

    // Express neighbor extrinsics relative to the reference (so the reference
    // becomes the identity).
    Mat3x3f ref_R_T = ref_R_host.T();

    for (int k = 0; k < numnei; k++) {
      const Camera& nei_cam = cameras_.GetCamera(nei_indices[k]);
      GpuCamera nei_gpu = nei_cam.BuildGpuCamera();

      // Compute relative extrinsic: nei_cam relative to ref_cam
      Mat3x3f nei_R_host;
      memcpy(nei_R_host.Ptr(), nei_gpu.R, 9 * sizeof(float));
      Vec3f nei_t_host(nei_gpu.t[0], nei_gpu.t[1], nei_gpu.t[2]);

      Mat3x3f rel_R = nei_R_host * ref_R_T;
      Vec3f rel_t = nei_t_host - rel_R * ref_t_host;

      // Copy intrinsic, relative R and t
      memcpy(nei_gpu_cams[k].K, nei_gpu.K, 9 * sizeof(float));
      memcpy(nei_gpu_cams[k].R, rel_R.Ptr(), 9 * sizeof(float));
      nei_gpu_cams[k].t[0] = rel_t.x;
      nei_gpu_cams[k].t[1] = rel_t.y;
      nei_gpu_cams[k].t[2] = rel_t.z;

      // Camera center in relative coords
      Mat3x3f rel_R_T_host = rel_R.T();
      Vec3f rel_center = rel_R_T_host * (Vec3f(0, 0, 0) - rel_t);
      nei_gpu_cams[k].center[0] = rel_center.x;
      nei_gpu_cams[k].center[1] = rel_center.y;
      nei_gpu_cams[k].center[2] = rel_center.z;
    }

    // 4. Set up AlgorithmParams from Config
    AlgorithmParams alg_params;
    alg_params.depthMin = config_.min_depth;
    alg_params.depthMax = config_.max_depth;
    alg_params.iterations = config_.iterations;
    alg_params.hair_alpha = config_.alpha;
    alg_params.hair_orient_gabor_ksize = config_.gabor_kernel_size;
    alg_params.hair_orient_gabor_sigma = config_.gabor_sigma;
    alg_params.hair_orient_gabor_gamma = config_.gabor_gamma;
    alg_params.hair_orient_gabor_lambd = config_.gabor_lambda;
    alg_params.hair_orient_gabor_min_contrast = config_.gabor_min_contrast;
    alg_params.hair_orient_gabor_min_response = config_.gabor_min_response;
    alg_params.hair_orient_rotate_res = config_.gabor_num_orientations;
    alg_params.bReconWithMask = config_.use_mask;
    alg_params.hair_pt_sample_radius = config_.pt_sample_radius;
    alg_params.hair_pt_sample_kappa = config_.pt_sample_kappa;
    alg_params.hair_delta_orient = config_.delta_orient;
    alg_params.hair_delta_depth = config_.delta_depth;
    alg_params.hair_spatial_prop_radius = config_.spatial_prop_radius;
    alg_params.hair_num_view_select = config_.num_view_select;
    alg_params.hair_mask_min_neighbor_views = config_.mask_min_neighbor_views;
    alg_params.hair_gaussian_pyramid = config_.gaussian_pyramid;
    alg_params.box_hsize = config_.patch_size / 2;
    alg_params.box_vsize = config_.patch_size / 2;
    alg_params.cols = width;
    alg_params.rows = height;

    // 5. Allocate GPU memory for reference image
    CUDA_CHECK(cudaMalloc(&d_refMap, npix * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_refMap, ref_img.data, npix * sizeof(float),
                          cudaMemcpyHostToDevice));

    // 6. Allocate neighbor images with pitched memory + texture objects
    d_neiMaps.resize(numnei);
    h_neiTexObjs.resize(numnei);
    std::vector<size_t> nei_pitches(numnei, 0);

    for (int k = 0; k < numnei; k++) {
      unsigned int nei_cam_id = cameras_.GetCamId(nei_indices[k]);
      std::string nei_path = image_loader::FormatPath(config_.image_dir, nei_cam_id);
      Image nei_img = image_loader::LoadGrayImage(nei_path, config_.downsample);
      if (nei_img.Empty()) {
        LOG_WARN("Failed to load neighbor image %s", nei_path.c_str());
        nei_img = Image(width, height, 1);  // blank
      }

      // Undistort neighbor image if using distorted inputs
      if (config_.distorted_images) {
        const Camera& nei_cam = cameras_.GetCamera(nei_indices[k]);
        GpuCamera nei_gpu = nei_cam.BuildGpuCamera();
        image_loader::UndistortImage(nei_img, nei_gpu.K, nei_gpu.dist);
      }

      CUDA_CHECK(cudaMallocPitch(&d_neiMaps[k], &nei_pitches[k],
                                 sizeof(float) * width, height));
      CUDA_CHECK(cudaMemcpy2D(d_neiMaps[k], nei_pitches[k], nei_img.data,
                              sizeof(float) * width, sizeof(float) * width, height,
                              cudaMemcpyHostToDevice));

      // Create texture object
      cudaResourceDesc resDesc;
      memset(&resDesc, 0, sizeof(resDesc));
      resDesc.resType = cudaResourceTypePitch2D;
      resDesc.res.pitch2D.devPtr = d_neiMaps[k];
      resDesc.res.pitch2D.pitchInBytes = nei_pitches[k];
      resDesc.res.pitch2D.width = width;
      resDesc.res.pitch2D.height = height;
      resDesc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

      cudaTextureDesc texDesc;
      memset(&texDesc, 0, sizeof(texDesc));
      texDesc.addressMode[0] = cudaAddressModeBorder;
      texDesc.addressMode[1] = cudaAddressModeBorder;
      texDesc.filterMode = cudaFilterModeLinear;
      texDesc.readMode = cudaReadModeElementType;
      texDesc.normalizedCoords = false;

      CUDA_CHECK(
          cudaCreateTextureObject(&h_neiTexObjs[k], &resDesc, &texDesc, nullptr));
    }

    // 6b. Load masks if use_mask is enabled

    if (config_.use_mask && !config_.mask_dir.empty()) {
      // Reference mask
      std::string ref_mask_path =
          image_loader::FormatPath(config_.mask_dir, ref_cam_id);
      Image ref_mask_img =
          image_loader::LoadMaskImage(ref_mask_path, config_.downsample);
      if (!ref_mask_img.Empty() && config_.distorted_images) {
        image_loader::UndistortImage(ref_mask_img, ref_gpu.K, ref_gpu.dist);
      }
      if (ref_mask_img.Empty() ||
          ref_mask_img.width != static_cast<int>(width) ||
          ref_mask_img.height != static_cast<int>(height)) {
        LOG_ERROR("Reference mask %s is missing or has size %dx%d (expected %ux%u)",
                  ref_mask_path.c_str(), ref_mask_img.width, ref_mask_img.height,
                  width, height);
        throw std::runtime_error("invalid reference mask");
      } else {
        unsigned int mask_npix = width * height;
        std::vector<unsigned char> ref_mask_uchar(mask_npix);
        for (unsigned int j = 0; j < mask_npix; j++) {
          ref_mask_uchar[j] =
              static_cast<unsigned char>(ref_mask_img.data[j] > 0.5f ? 255 : 0);
        }
        CUDA_CHECK(cudaMalloc(&d_refMask, mask_npix * sizeof(unsigned char)));
        CUDA_CHECK(cudaMemcpy(d_refMask, ref_mask_uchar.data(),
                              mask_npix * sizeof(unsigned char),
                              cudaMemcpyHostToDevice));
      }

      // Neighbor masks
      d_neiMaskBuffers.resize(numnei, nullptr);
      std::vector<unsigned char*> h_neiMaskDevPtrs(numnei, nullptr);
      for (int k = 0; k < numnei; k++) {
        unsigned int nei_cam_id = cameras_.GetCamId(nei_indices[k]);
        std::string nei_mask_path =
            image_loader::FormatPath(config_.mask_dir, nei_cam_id);
        Image nei_mask_img =
            image_loader::LoadMaskImage(nei_mask_path, config_.downsample);
        if (!nei_mask_img.Empty() && config_.distorted_images) {
          const Camera& nei_cam = cameras_.GetCamera(nei_indices[k]);
          GpuCamera nei_gpu = nei_cam.BuildGpuCamera();
          image_loader::UndistortImage(nei_mask_img, nei_gpu.K, nei_gpu.dist);
        }
        if (nei_mask_img.Empty() ||
            nei_mask_img.width != static_cast<int>(width) ||
            nei_mask_img.height != static_cast<int>(height)) {
          LOG_WARN("Ignoring neighbor mask %s with size %dx%d (expected %ux%u)",
                   nei_mask_path.c_str(), nei_mask_img.width, nei_mask_img.height,
                   width, height);
          continue;
        } else {
          unsigned int mask_npix = width * height;
          std::vector<unsigned char> nei_mask_uchar(mask_npix);
          for (unsigned int j = 0; j < mask_npix; j++) {
            nei_mask_uchar[j] =
                static_cast<unsigned char>(nei_mask_img.data[j] > 0.5f ? 255 : 0);
          }
          CUDA_CHECK(
              cudaMalloc(&d_neiMaskBuffers[k], mask_npix * sizeof(unsigned char)));
          CUDA_CHECK(cudaMemcpy(d_neiMaskBuffers[k], nei_mask_uchar.data(),
                                mask_npix * sizeof(unsigned char),
                                cudaMemcpyHostToDevice));
          h_neiMaskDevPtrs[k] = d_neiMaskBuffers[k];
        }
      }
      // Device array of device pointers for double indirection in kernel
      CUDA_CHECK(cudaMalloc(&d_neiMaskPtrs, numnei * sizeof(unsigned char*)));
      CUDA_CHECK(cudaMemcpy(d_neiMaskPtrs, h_neiMaskDevPtrs.data(),
                            numnei * sizeof(unsigned char*), cudaMemcpyHostToDevice));
    }

    // 7. Allocate output buffers
    CUDA_CHECK(cudaMalloc(&d_lineMap, npix * sizeof(Line3D)));
    CUDA_CHECK(cudaMalloc(&d_orient2D, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_variance, npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cost, npix * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_lineMap, 0, npix * sizeof(Line3D)));

    // 8. Build pyramid levels
    unsigned int num_levels = config_.num_hierarchy_levels;
    std::vector<unsigned int> level_widths(num_levels);
    std::vector<unsigned int> level_heights(num_levels);
    level_widths[0] = width;
    level_heights[0] = height;
    for (unsigned int l = 1; l < num_levels; l++) {
      level_widths[l] = level_widths[l - 1] / 2;
      level_heights[l] = level_heights[l - 1] / 2;
    }

    Mat3x3f K_scaled;
    memcpy(K_scaled.Ptr(), ref_gpu.K, 9 * sizeof(float));

    // 9. Create and run HierarchicalPatchMatch
    HierarchicalPatchMatch hpm(K_scaled, nei_gpu_cams, alg_params, level_widths,
                               level_heights, num_levels);
    hpm.use_mask = alg_params.bReconWithMask;

    if (!config_.output_dir.empty()) {
      std::string view_out = config_.output_dir + "/view_" + std::to_string(ref_cam_id);
      hpm.SetOutputFolder(view_out);
    }

    hpm.Run(d_refMap, h_neiTexObjs.data(), d_lineMap, d_orient2D, d_variance, d_cost,
            d_refMask, d_neiMaskPtrs);

    // 9b. Save per-view intermediates if requested
    if (config_.save_intermediates && !config_.output_dir.empty()) {
      std::string view_dir = config_.output_dir + "/view_" + std::to_string(ref_cam_id);
      std::filesystem::create_directories(view_dir);

      // Download orient2D, variance, cost from GPU
      std::vector<float> h_orient(npix), h_var(npix), h_costmap(npix);
      CUDA_CHECK(cudaMemcpy(h_orient.data(), d_orient2D, npix * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_var.data(), d_variance, npix * sizeof(float),
                            cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(h_costmap.data(), d_cost, npix * sizeof(float),
                            cudaMemcpyDeviceToHost));

      // Download line3D for depth map
      std::vector<Line3D> h_lines_tmp(npix);
      CUDA_CHECK(cudaMemcpy(h_lines_tmp.data(), d_lineMap, npix * sizeof(Line3D),
                            cudaMemcpyDeviceToHost));
      std::vector<float> h_depth(npix);
      for (unsigned int i = 0; i < npix; i++)
        h_depth[i] = h_lines_tmp[i].p.z;

      image_io::WriteExrFloat(view_dir + "/hair_depth.exr", h_depth.data(), width,
                              height);
      image_io::WriteExrFloat(view_dir + "/hair_orient2D.exr", h_orient.data(), width,
                              height);
      image_io::WriteExrFloat(view_dir + "/hair_variance.exr", h_var.data(), width,
                              height);
      image_io::WriteExrFloat(view_dir + "/hair_cost.exr", h_costmap.data(), width,
                              height);

      // Color-mapped visualizations
      // Orient2D: theta [0,pi] mapped to HSV hue (not auto-normalized)
      std::vector<float> h_orient_norm(npix);
      for (unsigned int i = 0; i < npix; i++)
        h_orient_norm[i] = h_orient[i] / static_cast<float>(M_PI);
      image_io::WriteVisPng(view_dir + "/hair_orient2D_vis.png", h_orient_norm.data(),
                            width, height, false);
      image_io::WriteVisPng(view_dir + "/hair_variance_vis.png", h_var.data(), width,
                            height, true);
      image_io::WriteVisPng(view_dir + "/hair_cost_vis.png", h_costmap.data(), width,
                            height, true, 0.9f);
      image_io::WriteVisPng(view_dir + "/hair_depth_vis.png", h_depth.data(), width,
                            height, true);
      LOG_INFO("  Saved intermediates to %s", view_dir.c_str());
    }

    // 10. Download camera-local line3D map for cross-validation fusion
    std::vector<Line3D> h_lines(npix);
    std::vector<float> h_cost(npix);
    CUDA_CHECK(cudaMemcpy(h_lines.data(), d_lineMap, npix * sizeof(Line3D),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_cost.data(), d_cost, npix * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Pre-filter: zero out lines with invalid depth or high cost
    size_t valid_count = 0;
    for (unsigned int i = 0; i < npix; i++) {
      if (h_lines[i].p.z <= 0 || h_cost[i] > config_.fusion_cost_threshold) {
        h_lines[i].p = make_float3(0, 0, 0);
        h_lines[i].v = make_float3(0, 0, 0);
      } else {
        valid_count++;
      }
    }

    // Transform valid lines from camera-local to WORLD coordinates
    Mat3x3f ref_R_inv = ref_R_host.T();
    Mat4x4f invRt;
    invRt.MakeI();
    for (int r = 0; r < 3; r++)
      for (int c = 0; c < 3; c++)
        invRt(r, c) = ref_R_inv(r, c);
    Vec3f neg_Rt_t = ref_R_inv * (Vec3f(0, 0, 0) - ref_t_host);
    invRt(0, 3) = neg_Rt_t.x;
    invRt(1, 3) = neg_Rt_t.y;
    invRt(2, 3) = neg_Rt_t.z;

    for (unsigned int i = 0; i < npix; i++) {
      if (h_lines[i].p.z <= 0)
        continue;  // skip zeroed lines
      Vec3f p_cam(h_lines[i].p.x, h_lines[i].p.y, h_lines[i].p.z);
      Vec3f p_world = invRt.transform_point(p_cam);
      Vec3f v_cam(h_lines[i].v.x, h_lines[i].v.y, h_lines[i].v.z);
      Vec3f v_world = ref_R_inv * v_cam;
      float len = v_world.Magnitude();
      if (len > 1e-6f)
        v_world = v_world / len;
      h_lines[i].p = make_float3(p_world.x, p_world.y, p_world.z);
      h_lines[i].v = make_float3(v_world.x, v_world.y, v_world.z);
    }

    LOG_INFO("  View %d: %zu valid pixels (of %u)", ref_index, valid_count, npix);

    // Save per-view point cloud (world coords)
    if (config_.save_intermediates && !config_.output_dir.empty()) {
      std::string view_dir = config_.output_dir + "/view_" + std::to_string(ref_cam_id);
      PointCloud pc;
      for (unsigned int i = 0; i < npix; i++) {
        if (h_lines[i].p.z <= 0)
          continue;
        pc.AddPoint(h_lines[i].p.x, h_lines[i].p.y, h_lines[i].p.z, h_lines[i].v.x,
                    h_lines[i].v.y, h_lines[i].v.z, 0);
      }
      ply_io::WritePointCloud(view_dir + "/ptcloud.ply", pc);
    }

    // Store result for cross-validation fusion (line3D now in WORLD coords)
    ViewMvsResult result;
    result.ref_index = ref_index;
    result.width = width;
    result.height = height;
    result.line_map = std::move(h_lines);
    result.nei_indices = nei_indices;

    {
      std::lock_guard<std::mutex> lock(results_mutex_);
      view_results_[ref_index] = std::move(result);
    }

    // 11. Clean up (don't abort on errors during cleanup in multi-GPU)
    cudaFree(d_refMap);
    cudaFree(d_lineMap);
    cudaFree(d_orient2D);
    cudaFree(d_variance);
    cudaFree(d_cost);

    for (int k = 0; k < numnei; k++) {
      cudaDestroyTextureObject(h_neiTexObjs[k]);
      cudaFree(d_neiMaps[k]);
    }

    // Clean up masks
    if (d_refMask)
      cudaFree(d_refMask);
    if (d_neiMaskPtrs)
      cudaFree(d_neiMaskPtrs);
    for (auto& p : d_neiMaskBuffers) {
      if (p)
        cudaFree(p);
    }

    cudaGetLastError();  // Clear any leftover errors

  } catch (const std::exception& e) {
    LOG_ERROR("View %d on GPU %d failed: %s", ref_index, gpu_id, e.what());
    // Best-effort GPU cleanup (ignore errors during cleanup)
    cudaFree(d_refMap);
    cudaFree(d_lineMap);
    cudaFree(d_orient2D);
    cudaFree(d_variance);
    cudaFree(d_cost);
    for (int k = 0; k < numnei; k++) {
      if (k < static_cast<int>(h_neiTexObjs.size()) && h_neiTexObjs[k])
        cudaDestroyTextureObject(h_neiTexObjs[k]);
      if (k < static_cast<int>(d_neiMaps.size()) && d_neiMaps[k])
        cudaFree(d_neiMaps[k]);
    }
    if (d_refMask)
      cudaFree(d_refMask);
    if (d_neiMaskPtrs)
      cudaFree(d_neiMaskPtrs);
    for (auto& p : d_neiMaskBuffers)
      if (p)
        cudaFree(p);
    cudaGetLastError();  // Clear error state
  }
}

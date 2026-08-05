// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include "mvs/patchmatch/hierarchical.h"

#include <algorithm>
#include <cmath>
#include <cstring>

#include "common/cuda_check.h"
#include "common/cuda_math.h"
#include "common/logger.h"
#include "mvs/gabor_orientation.h"
#include "mvs/patchmatch/image_ops.h"

// ---------------------------------------------------------------------------
// Helper: create a cudaTextureObject_t from a pitched float device buffer
// ---------------------------------------------------------------------------
static cudaTextureObject_t CreateFloatTexture(float* d_ptr, size_t pitch_bytes,
                                              unsigned int w, unsigned int h) {
  cudaResourceDesc res_desc;
  memset(&res_desc, 0, sizeof(res_desc));
  res_desc.resType = cudaResourceTypePitch2D;
  res_desc.res.pitch2D.devPtr = d_ptr;
  res_desc.res.pitch2D.pitchInBytes = pitch_bytes;
  res_desc.res.pitch2D.width = w;
  res_desc.res.pitch2D.height = h;
  res_desc.res.pitch2D.desc = cudaCreateChannelDesc<float>();

  cudaTextureDesc tex_desc;
  memset(&tex_desc, 0, sizeof(tex_desc));
  tex_desc.addressMode[0] = cudaAddressModeBorder;
  tex_desc.addressMode[1] = cudaAddressModeBorder;
  tex_desc.filterMode = cudaFilterModeLinear;
  tex_desc.readMode = cudaReadModeElementType;
  tex_desc.normalizedCoords = false;

  cudaTextureObject_t tex = 0;
  CUDA_CHECK(cudaCreateTextureObject(&tex, &res_desc, &tex_desc, nullptr));
  return tex;
}

// ---------------------------------------------------------------------------
// Helper: build PatchMatchParams from AlgorithmParams for a given level
// ---------------------------------------------------------------------------
static PatchMatchParams BuildPMParams(const AlgorithmParams& ap) {
  PatchMatchParams pm;
  pm.depthMin = ap.depthMin;
  pm.depthMax = ap.depthMax;
  pm.maxIt = ap.iterations;
  pm.deltaNormal = ap.deltaNormal;
  pm.deltaDisp = ap.deltaDisp;
  pm.bMaskInput = ap.bReconWithMask;
  pm.box_hsize = ap.box_hsize;
  pm.box_vsize = ap.box_vsize;
  pm.hair_alpha = ap.hair_alpha;
  pm.hair_pt_sample_radius = ap.hair_pt_sample_radius;
  pm.hair_pt_sample_kappa = ap.hair_pt_sample_kappa;
  pm.hair_delta_orient = ap.hair_delta_orient;
  pm.hair_delta_depth = ap.hair_delta_depth;
  pm.hair_spatial_prop_radius = ap.hair_spatial_prop_radius;
  pm.hair_num_view_select = ap.hair_num_view_select;
  pm.hair_mask_min_neighbor_views = ap.hair_mask_min_neighbor_views;
  return pm;
}

// ===========================================================================
// Constructor / Destructor
// ===========================================================================

HierarchicalPatchMatch::HierarchicalPatchMatch(const Mat3x3f& K,
                                               const std::vector<GpuCamera>& nei_cams,
                                               const AlgorithmParams& params,
                                               std::vector<unsigned int>& widths,
                                               std::vector<unsigned int>& heights,
                                               unsigned int numLevels)
    : intrinsic_(K), nei_cams_(nei_cams), params_(params) {
  numnei_ = static_cast<int>(nei_cams.size());
  Allocate(widths, heights, numLevels);
}

HierarchicalPatchMatch::~HierarchicalPatchMatch() {
  Deallocate();
}

// ===========================================================================
// Allocate
// ===========================================================================

void HierarchicalPatchMatch::Allocate(std::vector<unsigned int>& widths,
                                      std::vector<unsigned int>& heights,
                                      unsigned int numLevels) {
  num_levels_ = numLevels;
  unsigned int origW = widths[0];
  unsigned int origH = heights[0];

  widths_ = widths;
  heights_ = heights;

  // Resize all per-level vectors
  ref_gray_.resize(numLevels, nullptr);
  ref_orient_.resize(numLevels, nullptr);
  ref_orient_var_.resize(numLevels, nullptr);
  nei_gray_.resize(numLevels, nullptr);
  nei_orient_.resize(numLevels, nullptr);
  nei_orient_var_.resize(numLevels, nullptr);
  est_line_.resize(numLevels, nullptr);
  cost_color_.resize(numLevels, nullptr);
  cost_orient_.resize(numLevels, nullptr);
  cost_total_.resize(numLevels, nullptr);

  nei_gray_tex_.resize(numLevels, nullptr);
  nei_orient_tex_.resize(numLevels, nullptr);
  nei_orient_var_tex_.resize(numLevels, nullptr);

  pitch_gray_.resize(numLevels, std::vector<size_t>(numnei_, 0));
  pitch_orient_.resize(numLevels, std::vector<size_t>(numnei_, 0));
  pitch_orient_var_.resize(numLevels, std::vector<size_t>(numnei_, 0));

  intrinsic_at_level_.resize(numLevels);
  nei_cams_at_level_.resize(numLevels);
  params_at_level_.resize(numLevels);

  // Create the single PatchMatchEngine at original resolution
  engine_ = new PatchMatchEngine(origW, origH, numnei_);
  {
    float K_arr[9];
    memcpy(K_arr, intrinsic_.Ptr(), 9 * sizeof(float));
    PatchMatchParams pm = BuildPMParams(params_);
    engine_->InitParameters(K_arr, nei_cams_, pm, origW, origH);
  }

  // Per-level allocation
  for (unsigned int i = 0; i < numLevels; i++) {
    unsigned int w = widths_[i];
    unsigned int h = heights_[i];
    unsigned int npix = w * h;

    float scaleW = static_cast<float>(w) / static_cast<float>(origW);
    float scaleH = static_cast<float>(h) / static_cast<float>(origH);

    // --- Scaled intrinsic (half-pixel-aware principal point scaling) ---
    Mat3x3f newK = intrinsic_;
    // fx, skew: scale linearly
    newK(0, 0) *= scaleW;
    newK(0, 1) *= scaleW;
    // fy: scale linearly
    newK(1, 0) *= scaleH;
    newK(1, 1) *= scaleH;
    // Principal point: half-pixel-aware scaling
    newK(0, 2) = (newK(0, 2) + 0.5f) * scaleW - 0.5f;
    newK(1, 2) = (newK(1, 2) + 0.5f) * scaleH - 0.5f;
    intrinsic_at_level_[i] = newK;

    // --- Scaled neighbor cameras (same half-pixel-aware formula) ---
    std::vector<GpuCamera> scaled_cams = nei_cams_;
    for (int k = 0; k < numnei_; k++) {
      // fx, skew
      scaled_cams[k].K[0] *= scaleW;
      scaled_cams[k].K[1] *= scaleW;
      // cx: half-pixel-aware
      scaled_cams[k].K[2] = (scaled_cams[k].K[2] + 0.5f) * scaleW - 0.5f;
      // fy, skew
      scaled_cams[k].K[3] *= scaleH;
      scaled_cams[k].K[4] *= scaleH;
      // cy: half-pixel-aware
      scaled_cams[k].K[5] = (scaled_cams[k].K[5] + 0.5f) * scaleH - 0.5f;
    }
    nei_cams_at_level_[i] = scaled_cams;

    // --- Scaled algorithm params ---
    // Scale factor = sqrt(width_at_mirror_level / origWidth)
    // where mirror_level = numLevels - 1 - i (coarsest level gets scale=1,
    // finest gets smallest scale)
    AlgorithmParams newparams = params_;
    newparams.cols = static_cast<int>(w);
    newparams.rows = static_cast<int>(h);
    float tmpinvscale = sqrtf(static_cast<float>(widths[numLevels - 1 - i]) /
                              static_cast<float>(origW));
    newparams.iterations = std::max(
        4, static_cast<int>(static_cast<float>(params_.iterations) * tmpinvscale));
    newparams.deltaNormal = params_.deltaNormal * tmpinvscale;
    newparams.deltaDisp = params_.deltaDisp * tmpinvscale;
    params_at_level_[i] = newparams;

    // --- Level 0: orient/variance for ref and neighbors ---
    // (ref gray and est_line at level 0 are external pointers set in Run)
    CUDA_CHECK(cudaMalloc(&ref_orient_[i], npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ref_orient_var_[i], npix * sizeof(float)));

    // Neighbor orient/variance — pitched allocations at ALL levels
    nei_orient_[i] = new float*[numnei_];
    nei_orient_var_[i] = new float*[numnei_];
    for (int k = 0; k < numnei_; k++) {
      CUDA_CHECK(cudaMallocPitch(&nei_orient_[i][k], &pitch_orient_[i][k],
                                 sizeof(float) * w, h));
      CUDA_CHECK(cudaMallocPitch(&nei_orient_var_[i][k],
                                 &pitch_orient_var_[i][k], sizeof(float) * w, h));
    }

    // Orient/variance texture object arrays (filled during Run)
    nei_orient_tex_[i] = new cudaTextureObject_t[numnei_];
    nei_orient_var_tex_[i] = new cudaTextureObject_t[numnei_];
    memset(nei_orient_tex_[i], 0, numnei_ * sizeof(cudaTextureObject_t));
    memset(nei_orient_var_tex_[i], 0, numnei_ * sizeof(cudaTextureObject_t));

    // Cost maps — allocated at ALL levels
    CUDA_CHECK(cudaMalloc(&cost_color_[i], npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cost_orient_[i], npix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&cost_total_[i], npix * sizeof(float)));
    CUDA_CHECK(cudaMemset(cost_color_[i], 0, npix * sizeof(float)));
    CUDA_CHECK(cudaMemset(cost_orient_[i], 0, npix * sizeof(float)));
    CUDA_CHECK(cudaMemset(cost_total_[i], 0, npix * sizeof(float)));

    // --- Level > 0: ref gray, neighbor gray (pitched), line3D ---
    if (i > 0) {
      CUDA_CHECK(cudaMalloc(&ref_gray_[i], npix * sizeof(float)));

      nei_gray_[i] = new float*[numnei_];
      nei_gray_tex_[i] = new cudaTextureObject_t[numnei_];
      memset(nei_gray_tex_[i], 0, numnei_ * sizeof(cudaTextureObject_t));
      for (int k = 0; k < numnei_; k++) {
        CUDA_CHECK(cudaMallocPitch(&nei_gray_[i][k], &pitch_gray_[i][k],
                                   sizeof(float) * w, h));
      }

      CUDA_CHECK(cudaMalloc(&est_line_[i], npix * sizeof(Line3D)));
      CUDA_CHECK(cudaMemset(est_line_[i], 0, npix * sizeof(Line3D)));
    }
    // Level 0 ref_gray_, est_line_, nei_gray_tex_ are set in Run()
  }
}

// ===========================================================================
// Deallocate
// ===========================================================================

void HierarchicalPatchMatch::Deallocate() {
  for (unsigned int i = 0; i < num_levels_; i++) {
    // ref orient/variance (all levels)
    if (ref_orient_[i])
      cudaFree(ref_orient_[i]);
    if (ref_orient_var_[i])
      cudaFree(ref_orient_var_[i]);

    // nei orient/variance (all levels)
    if (nei_orient_[i]) {
      for (int k = 0; k < numnei_; k++) {
        if (nei_orient_[i][k])
          cudaFree(nei_orient_[i][k]);
        if (nei_orient_var_[i][k])
          cudaFree(nei_orient_var_[i][k]);
      }
      delete[] nei_orient_[i];
      delete[] nei_orient_var_[i];
    }

    // orient/variance tex arrays (destroy surviving handles first)
    if (nei_orient_tex_[i]) {
      for (int k = 0; k < numnei_; k++)
        if (nei_orient_tex_[i][k])
          cudaDestroyTextureObject(nei_orient_tex_[i][k]);
      delete[] nei_orient_tex_[i];
    }
    if (nei_orient_var_tex_[i]) {
      for (int k = 0; k < numnei_; k++)
        if (nei_orient_var_tex_[i][k])
          cudaDestroyTextureObject(nei_orient_var_tex_[i][k]);
      delete[] nei_orient_var_tex_[i];
    }

    // cost maps (all levels)
    if (cost_color_[i])
      cudaFree(cost_color_[i]);
    if (cost_orient_[i])
      cudaFree(cost_orient_[i]);
    if (cost_total_[i])
      cudaFree(cost_total_[i]);

    // level > 0: ref gray, nei gray (pitched), gray tex, line3D
    if (i > 0) {
      if (ref_gray_[i])
        cudaFree(ref_gray_[i]);

      if (nei_gray_[i]) {
        for (int k = 0; k < numnei_; k++) {
          if (nei_gray_[i][k])
            cudaFree(nei_gray_[i][k]);
        }
        delete[] nei_gray_[i];
      }
      if (nei_gray_tex_[i]) {
        for (int k = 0; k < numnei_; k++)
          if (nei_gray_tex_[i][k])
            cudaDestroyTextureObject(nei_gray_tex_[i][k]);
        delete[] nei_gray_tex_[i];
      }

      if (est_line_[i])
        cudaFree(est_line_[i]);
    }
  }

  delete engine_;
  engine_ = nullptr;
}

// ===========================================================================
// Run
// ===========================================================================

void HierarchicalPatchMatch::Run(float* d_refGrayMap, cudaTextureObject_t* d_neiTexObjs,
                                 Line3D* d_estLineMap, float* d_orient2D,
                                 float* d_variance, float* d_cost,
                                 unsigned char* d_refMask, unsigned char** d_neiMasks) {
  unsigned int numLevels = num_levels_;

  // -----------------------------------------------------------------------
  // Set level-0 external pointers
  // -----------------------------------------------------------------------
  ref_gray_[0] = d_refGrayMap;
  nei_gray_tex_[0] = d_neiTexObjs;  // HOST array from caller
  est_line_[0] = d_estLineMap;

  // -----------------------------------------------------------------------
  // Build multi-level gray images (fine-to-coarse, levels 0..N-2)
  // -----------------------------------------------------------------------
  for (unsigned int i = 0; i < numLevels - 1; i++) {
    unsigned int src_w = widths_[i];
    unsigned int src_h = heights_[i];
    unsigned int dst_w = widths_[i + 1];
    unsigned int dst_h = heights_[i + 1];

    // Reference: downsample with optional Gaussian pre-smoothing
    if (params_.hair_gaussian_pyramid) {
      float* d_smoothed = nullptr;
      CUDA_CHECK(cudaMalloc(&d_smoothed, src_w * src_h * sizeof(float)));
      GaussianSmooth(d_smoothed, ref_gray_[i], src_w, src_h);
      ResampleImage(ref_gray_[i + 1], dst_w, dst_h, d_smoothed, src_w, src_h, nullptr);
      CUDA_CHECK(cudaFree(d_smoothed));
    } else {
      ResampleImage(ref_gray_[i + 1], dst_w, dst_h, ref_gray_[i], src_w, src_h,
                    nullptr);
    }

    // Neighbors: read from texture, write to pitched memory
    for (int j = 0; j < numnei_; j++) {
      if (params_.hair_gaussian_pyramid) {
        // Smooth from texture to temp pitched buffer, then resample to next level
        float* d_smoothed = nullptr;
        size_t smooth_pitch = 0;
        CUDA_CHECK(
            cudaMallocPitch(&d_smoothed, &smooth_pitch, src_w * sizeof(float), src_h));
        GaussianSmoothTexture(d_smoothed, nei_gray_tex_[i][j], src_w, src_h,
                              static_cast<unsigned int>(smooth_pitch / sizeof(float)));
        cudaTextureObject_t smoothTex =
            CreateFloatTexture(d_smoothed, smooth_pitch, src_w, src_h);
        ResampleImageTexture(
            nei_gray_[i + 1][j], dst_w, dst_h,
            static_cast<unsigned int>(pitch_gray_[i + 1][j] / sizeof(float)), smoothTex,
            src_w, src_h);
        CUDA_CHECK(cudaDestroyTextureObject(smoothTex));
        CUDA_CHECK(cudaFree(d_smoothed));
      } else {
        ResampleImageTexture(
            nei_gray_[i + 1][j], dst_w, dst_h,
            static_cast<unsigned int>(pitch_gray_[i + 1][j] / sizeof(float)),
            nei_gray_tex_[i][j], src_w, src_h);
      }

      // Create texture object for the downsampled neighbor at level i+1
      nei_gray_tex_[i + 1][j] =
          CreateFloatTexture(nei_gray_[i + 1][j], pitch_gray_[i + 1][j], dst_w,
                             dst_h);
    }

    // Initialize line3D at level i+1 to zero
    Line3D zero_line;
    zero_line.p = make_float3(0.f, 0.f, 0.f);
    zero_line.v = make_float3(0.f, 0.f, 0.f);
    InitializeLine3D(est_line_[i + 1], zero_line, dst_w, dst_h);
  }

  // -----------------------------------------------------------------------
  // Compute orientation maps for ALL levels
  // -----------------------------------------------------------------------
  GaborParams gabor_params;
  gabor_params.ksize = params_.hair_orient_gabor_ksize;
  gabor_params.sigma = params_.hair_orient_gabor_sigma;
  gabor_params.gamma = params_.hair_orient_gabor_gamma;
  gabor_params.lambd = params_.hair_orient_gabor_lambd;
  gabor_params.min_contrast = params_.hair_orient_gabor_min_contrast;
  gabor_params.min_response = params_.hair_orient_gabor_min_response;
  int rotate_res = params_.hair_orient_rotate_res;

  for (unsigned int i = 0; i < numLevels; i++) {
    unsigned int w = widths_[i];
    unsigned int h = heights_[i];

    // Reference orientation from grayscale (non-pitched)
    ComputeGaborOrientation(ref_orient_[i], ref_orient_var_[i], ref_gray_[i], w, h,
                            rotate_res, gabor_params, nullptr);

    // Neighbor orientations from gray texture objects (pitched output)
    for (int j = 0; j < numnei_; j++) {
      ComputeGaborOrientationTexture(
          nei_orient_[i][j],
          static_cast<unsigned int>(pitch_orient_[i][j] / sizeof(float)),
          nei_orient_var_[i][j],
          static_cast<unsigned int>(pitch_orient_var_[i][j] / sizeof(float)),
          nei_gray_tex_[i][j], w, h, rotate_res, gabor_params, nullptr);

      // Create texture objects for orient and orient variance
      nei_orient_tex_[i][j] =
          CreateFloatTexture(nei_orient_[i][j], pitch_orient_[i][j], w, h);
      nei_orient_var_tex_[i][j] =
          CreateFloatTexture(nei_orient_var_[i][j], pitch_orient_var_[i][j], w, h);
    }
  }

  // -----------------------------------------------------------------------
  // Solve coarse-to-fine
  // -----------------------------------------------------------------------
  LOG_DEBUG("LPMVS PatchMatch: %d levels, starting coarse-to-fine",
            static_cast<int>(numLevels));

  for (int i = static_cast<int>(numLevels) - 1; i >= 0; i--) {
    unsigned int w = widths_[i];
    unsigned int h = heights_[i];
    LOG_DEBUG("  Level %d: %dx%d, numnei=%d", i, w, h, numnei_);

    // Build PatchMatchParams for this level
    PatchMatchParams pm = BuildPMParams(params_at_level_[i]);

    // Only use masks at the finest level (level 0) to avoid resolution mismatch
    if (i > 0) {
      pm.bMaskInput = false;
    }

    // Re-initialize engine for this level's cameras and params
    float K_arr[9];
    memcpy(K_arr, intrinsic_at_level_[i].Ptr(), 9 * sizeof(float));
    engine_->InitParameters(K_arr, nei_cams_at_level_[i], pm, w, h);

    // Set up engine's input/state pointers
    PatchMatchInput& inp = engine_->GetInput();
    PatchMatchState& st = engine_->GetState();

    st.d_line3D = est_line_[i];
    inp.d_refGrayMapFloat = ref_gray_[i];

    // Copy neighbor gray texture object handles to device
    CUDA_CHECK(cudaMemcpy(inp.d_neiTexMapsObj, nei_gray_tex_[i],
                          numnei_ * sizeof(cudaTextureObject_t),
                          cudaMemcpyHostToDevice));

    inp.d_refOrientMapFloat = ref_orient_[i];
    CUDA_CHECK(cudaMemcpy(inp.d_neiOrientTexMapsObj, nei_orient_tex_[i],
                          numnei_ * sizeof(cudaTextureObject_t),
                          cudaMemcpyHostToDevice));

    inp.d_refOrientVarianceMapFloat = ref_orient_var_[i];
    CUDA_CHECK(cudaMemcpy(inp.d_neiOrientVarianceTexMapsObj, nei_orient_var_tex_[i],
                          numnei_ * sizeof(cudaTextureObject_t),
                          cudaMemcpyHostToDevice));

    // Masks (only valid at level 0)
    if (i == 0) {
      inp.d_refMaskMapUchar = d_refMask;
      inp.d_neiMaskMapsUchar = d_neiMasks;
    } else {
      inp.d_refMaskMapUchar = nullptr;
      inp.d_neiMaskMapsUchar = nullptr;
    }

    // Run PatchMatch at this level
    engine_->RunHair(inp, st, pm);

    // Upsample line3D to the next finer level
    if (i > 0) {
      ResampleLine3DMap(est_line_[i - 1], widths_[i - 1], heights_[i - 1], est_line_[i],
                        w, h);

      // Destroy gray texture objects for this level (levels > 0 only)
      for (int j = 0; j < numnei_; j++) {
        if (nei_gray_tex_[i][j]) {
          CUDA_CHECK(cudaDestroyTextureObject(nei_gray_tex_[i][j]));
          nei_gray_tex_[i][j] = 0;
        }
      }
    }

    // Destroy orient/variance texture objects for this level
    for (int j = 0; j < numnei_; j++) {
      if (nei_orient_tex_[i][j]) {
        CUDA_CHECK(cudaDestroyTextureObject(nei_orient_tex_[i][j]));
        nei_orient_tex_[i][j] = 0;
      }
      if (nei_orient_var_tex_[i][j]) {
        CUDA_CHECK(cudaDestroyTextureObject(nei_orient_var_tex_[i][j]));
        nei_orient_var_tex_[i][j] = 0;
      }
    }
  }

  // -----------------------------------------------------------------------
  // Copy finest-level results to caller's output buffers
  // -----------------------------------------------------------------------
  unsigned int w0 = widths_[0];
  unsigned int h0 = heights_[0];
  unsigned int npix0 = w0 * h0;

  // est_line_[0] IS d_estLineMap (same pointer), so no copy needed for line3D.
  CUDA_CHECK(cudaMemcpy(d_orient2D, ref_orient_[0], npix0 * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  CUDA_CHECK(cudaMemcpy(d_variance, ref_orient_var_[0], npix0 * sizeof(float),
                        cudaMemcpyDeviceToDevice));
  // Copy actual cost from solver state (not the zeroed cost_total_[0] buffer)
  CUDA_CHECK(cudaMemcpy(d_cost, engine_->GetState().d_cost_total, npix0 * sizeof(float),
                        cudaMemcpyDeviceToDevice));

  LOG_DEBUG("HierarchicalPatchMatch::Run completed");
}

// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the license found in the
// LICENSE file in the root directory of this source tree.
//
// Reference: "Strand-accurate Multi-view Hair Capture"
//            G. Nam, C. Wu, M.H. Kim, Y. Sheikh (CVPR 2019)

#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>

#include "common/config.h"
#include "common/logger.h"
#include "common/profiler.h"
#include "common/camera_array.h"
#include "common/point_cloud.h"
#include "common/ply_io.h"
#include "common/timer.h"
#include "mvs/patchmatch_mvs.h"
#include "mvs/depth_fusion.h"
#include "strand/meanshift.h"
#include "strand/strand_tracer.h"
#include "strand/strand_cleaner.h"
#include "strand/hair_grower.h"
#include "strand/mesh_generator.h"
#include "strand/strand_io.h"

static void PrintUsage(const char* prog) {
  printf("Usage: %s <command> <config.toml>\n\n", prog);
  printf("Commands:\n");
  printf("  mvs        Line-based PatchMatch MVS (multi-GPU)\n");
  printf("  meanshift  Mean-shift 3D line fusion (CUDA)\n");
  printf("  trace      Forward Euler strand tracing\n");
  printf("  clean      Remove short + outlier strands\n");
  printf("  grow       Multi-view strand-tip growing (CUDA)\n");
  printf("  mesh       Cylinder mesh generation\n");
  printf("  pipeline   Run all stages end-to-end\n");
}

static int RunMvs(const Config& cfg) {
  PROFILE_SCOPE("MVS Pipeline");
  ScopedTimer timer("MVS Pipeline");

  CameraArray cameras;
  if (!cameras.LoadFromJson(cfg.cameras_json))
    return 1;

  if (cfg.downsample != 1.0f) {
    cameras.DownsizeCameras(cfg.downsample);
  }

  PatchMatchMvs mvs(cfg, cameras);
  mvs.Run();

  PointCloud fused = depth_fusion::FusePointClouds(mvs.view_results(), cameras, cfg);

  std::string out_path = cfg.output_dir + "/fused.ply";
  ply_io::WritePointCloud(out_path, fused);

  return 0;
}

static int RunMeanshift(const Config& cfg) {
  PROFILE_SCOPE("Mean-shift");

  PointCloud cloud;
  std::string in_path = cfg.output_dir + "/fused.ply";
  if (!ply_io::ReadPointCloud(in_path, cloud))
    return 1;

  PointCloud filtered = meanshift::RunCuda(cloud, cfg, cfg.gpu_id, cfg.num_gpus);

  std::string out_path = cfg.output_dir + "/meanshift.ply";
  ply_io::WritePointCloud(out_path, filtered);

  return 0;
}

static int RunTrace(const Config& cfg) {
  PROFILE_SCOPE("Strand Tracing");

  PointCloud cloud;
  std::string in_path = cfg.output_dir + "/meanshift.ply";
  if (!ply_io::ReadPointCloud(in_path, cloud))
    return 1;

  auto strands = strand_tracer::Trace(cloud, cfg);

  std::string out_ply = cfg.output_dir + "/strands_raw.ply";
  std::string out_bin = cfg.output_dir + "/strands_raw.bin";
  strand_io::WritePly(out_ply, strands);
  strand_io::WriteBinary(out_bin, strands);

  return 0;
}

static int RunClean(const Config& cfg) {
  PROFILE_SCOPE("Strand Cleaning");

  std::vector<Strand> strands;
  std::string in_path = cfg.output_dir + "/strands_raw.bin";
  if (!strand_io::ReadBinary(in_path, strands))
    return 1;

  auto cleaned = strand_cleaner::Clean(strands, cfg);

  std::string out_ply = cfg.output_dir + "/strands_clean.ply";
  std::string out_bin = cfg.output_dir + "/strands_clean.bin";
  strand_io::WritePly(out_ply, cleaned);
  strand_io::WriteBinary(out_bin, cleaned);

  return 0;
}

static int RunGrow(const Config& cfg) {
  PROFILE_SCOPE("Multi-view Hair Growing");

  std::vector<Strand> strands;
  std::string in_path = cfg.output_dir + "/strands_clean.bin";
  if (!strand_io::ReadBinary(in_path, strands))
    return 1;

  std::vector<Strand> grown;
  if (strands.empty()) {
    // An empty reconstruction is a valid input and does not require camera
    // images. Still emit both expected outputs so later stages remain usable.
    grown = strands;
  } else {
    CameraArray cameras;
    if (!cameras.LoadFromJson(cfg.cameras_json))
      return 1;

    try {
      grown = hair_grower::GrowCuda(strands, cameras, cfg, cfg.gpu_id,
                                    cfg.num_gpus);
    } catch (const std::exception& e) {
      LOG_ERROR("Hair growing failed: %s", e.what());
      return 1;
    }
  }

  std::string out_ply = cfg.output_dir + "/strands_grown.ply";
  std::string out_bin = cfg.output_dir + "/strands_grown.bin";
  const bool ply_ok = strand_io::WritePly(out_ply, grown);
  const bool bin_ok = strand_io::WriteBinary(out_bin, grown);
  return ply_ok && bin_ok ? 0 : 1;
}

static int RunMesh(const Config& cfg) {
  PROFILE_SCOPE("Mesh Generation");

  std::vector<Strand> strands;
  std::string in_path = cfg.output_dir + "/strands_grown.bin";
  if (!strand_io::ReadBinary(in_path, strands))
    return 1;

  std::string out_path = cfg.output_dir + "/hair_mesh.ply";
  mesh_generator::WriteCylinderMesh(out_path, strands, 0.2f, 6);

  return 0;
}

static int RunPipeline(const Config& cfg) {
  ScopedTimer timer("Full Pipeline");

  LOG_INFO("=== Stage 1: MVS ===");
  if (RunMvs(cfg) != 0)
    return 1;

  LOG_INFO("=== Stage 2: Mean-shift ===");
  if (RunMeanshift(cfg) != 0)
    return 1;

  LOG_INFO("=== Stage 3: Strand Tracing ===");
  if (RunTrace(cfg) != 0)
    return 1;

  LOG_INFO("=== Stage 4: Strand Cleaning ===");
  if (RunClean(cfg) != 0)
    return 1;

  LOG_INFO("=== Stage 5: Multi-view Hair Growing ===");
  if (RunGrow(cfg) != 0)
    return 1;

  LOG_INFO("=== Stage 6: Mesh Generation ===");
  if (RunMesh(cfg) != 0)
    return 1;

  LOG_INFO("Pipeline complete.");
  return 0;
}

int main(int argc, char* argv[]) {
  if (argc < 2) {
    PrintUsage(argv[0]);
    return 1;
  }

  if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-h") == 0) {
    PrintUsage(argv[0]);
    return 0;
  }

  if (argc < 3) {
    LOG_ERROR("Missing config file");
    PrintUsage(argv[0]);
    return 1;
  }

  const char* command = argv[1];
  const char* config_path = argv[2];

  Config cfg;
  try {
    cfg = Config::LoadFromToml(config_path);
  } catch (const std::exception& e) {
    fprintf(stderr, "Error: %s\n", e.what());
    return 1;
  }

  Logger::Init(cfg.log_level, cfg.log_file);
  Profiler::Enable(cfg.profile);

  std::filesystem::create_directories(cfg.output_dir);

  int ret = 1;
  if (strcmp(command, "mvs") == 0)
    ret = RunMvs(cfg);
  else if (strcmp(command, "meanshift") == 0)
    ret = RunMeanshift(cfg);
  else if (strcmp(command, "trace") == 0)
    ret = RunTrace(cfg);
  else if (strcmp(command, "clean") == 0)
    ret = RunClean(cfg);
  else if (strcmp(command, "grow") == 0)
    ret = RunGrow(cfg);
  else if (strcmp(command, "mesh") == 0)
    ret = RunMesh(cfg);
  else if (strcmp(command, "pipeline") == 0)
    ret = RunPipeline(cfg);
  else {
    LOG_ERROR("Unknown command: %s", command);
    PrintUsage(argv[0]);
  }

  Profiler::PrintReport();
  return ret;
}

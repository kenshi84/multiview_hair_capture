# Configuration Reference

All parameters with their defaults. Omitted parameters use the default values shown below.

## `[data]`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `cameras_json` | (required) | Path to camera calibration JSON file |
| `image_dir` | (required) | Printf pattern for image paths, e.g., `data/images/%06d.png`. The camera ID (unsigned int) is formatted into the pattern. |
| `mask_dir` | `""` | Printf pattern for hair mask paths. Same naming convention as `image_dir`. Used when either `mvs.use_mask` or `grow.use_mask` is enabled. |
| `output_dir` | (required) | Directory for all output files (fused point cloud, raw/clean/grown strands, and mesh) |
| `downsample` | `1.0` | Image downscale factor applied after loading. `0.5` = half resolution. Intrinsics are scaled accordingly. |
| `distorted_images` | `false` | If `true`, images are undistorted at load time using distortion coefficients from `cameras.json`. Set to `true` when using raw camera images. |

## `[mvs]`

### Depth & Resolution

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_depth` | `800` | Minimum depth search range in mm (along camera optical axis) |
| `max_depth` | `1400` | Maximum depth search range in mm |
| `num_hierarchy_levels` | `4` | Number of coarse-to-fine pyramid levels. Each level is half the resolution of the previous. |
| `gaussian_pyramid` | `false` | If `true`, apply Gaussian blur before each pyramid downsample (proper anti-aliasing). If `false`, downsample directly (preserves thin strand features but may alias). |

### View Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_neighbor_views` | `25` | Number of neighbor views selected per reference view. The closest views (by baseline angle, ascending) within `[min_angle, max_angle]` are picked. |
| `min_angle` | `1.0` | Minimum baseline angle (degrees) between reference and neighbor camera, measured at the centroid of all camera centers. Views closer than this are excluded. |
| `max_angle` | `90.0` | Maximum baseline angle (degrees), measured at the centroid of all camera centers. Views farther than this are excluded. |
| `num_view_select` | `4` | Best-K view selection during cost evaluation. Of all neighbor views, only the K with lowest cost contribute to the final cost. |
| `num_gpus` | `1` | Number of GPUs for MVS, mean-shift, and hair growing. MVS views and independent strand-tip batches are distributed across GPUs. |

### Cost Function

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | `0.1` | Weight for geometric (orientation) cost vs intensity (NCC) cost. Total cost = `(1 - alpha) * orient_cost + alpha * color_cost`. |
| `pt_sample_radius` | `10.0` | Radius (pixels) for sampling points along the projected 2D line in each view. |
| `pt_sample_kappa` | `41` | Number of sample points along the projected 2D line. Must be odd and `<= 41` (compile-time max set by `HAIR_LPMVS_KAPPA` in `cost.cuh`). Even values are rounded up to the next odd internally. |
| `use_mask` | `false` | Enable hair mask filtering. Hypothesis centers must be inside the reference mask, and neighbor views are eligible when the projected center is inside their mask. Support windows may extend beyond mask boundaries. |
| `mask_min_neighbor_views` | `1` | Minimum number of neighbor views whose projected center must be inside a valid mask. Masked-out neighbors are excluded from cost/view selection instead of rejecting the hypothesis globally. |

### Solver

| Parameter | Default | Description |
|-----------|---------|-------------|
| `iterations` | `8` | PatchMatch iterations at the **coarsest** pyramid level. Finer levels use proportionally fewer iterations (scaled by `sqrt(level_width / coarsest_width)`, clamped at min 4). |
| `patch_size` | `21` | Support patch size in pixels. The half-size (`patch_size / 2`) is used as the box radius. |
| `delta_depth` | `30.0` | Max random perturbation for depth (mm) during refinement. Halved every 3 refinement iterations within a pass. Also scaled per pyramid level (smaller at finer levels). |
| `delta_orient` | `0.8` | Max random perturbation added to each component of the unit direction vector, then renormalized. Halved every 3 refinement iterations within a pass. Also scaled per pyramid level (smaller at finer levels). |
| `spatial_prop_radius` | `5.0` | Radius (pixels) for disk-based spatial propagation. Random pixels within this radius are tested as propagation candidates. |

### `[mvs.gabor]`

Gabor filter bank parameters for computing 2D hair orientation fields.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_orientations` | `180` | Number of filter orientations (1-degree resolution at 180). |
| `kernel_size` | `21` | Gabor kernel size in pixels. |
| `sigma` | `1.12` | Gaussian envelope standard deviation. |
| `gamma` | `0.28` | Spatial aspect ratio (sigma_y = sigma / gamma). |
| `lambda` | `3.0` | Wavelength of the cosine factor. |
| `min_contrast` | `0.01` | Minimum local grayscale standard deviation (images are normalized to 0–1); lower-contrast pixels are marked invalid. |
| `min_response` | `0.02` | Minimum normalized quadrature-filter response; weaker responses are marked invalid. |

### `[mvs.fusion]`

Cross-view depth map fusion parameters. After per-view MVS, 3D lines from all views are merged into a single point cloud. A point is kept only if it is consistent with enough neighbor views.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `position_threshold` | `1.0` | Maximum position difference (mm) for two 3D points to be considered consistent. |
| `orientation_threshold` | `10.0` | Maximum direction difference (degrees) for consistency. |
| `min_neighbors` | `4` | Minimum number of consistent neighbor views required to keep a point. |
| `cost_threshold` | `0.2` | Maximum matching cost for a point to be considered valid. |

## `[meanshift]`

GPU mean-shift filtering that fuses the noisy point cloud into thin strand-like curves using bilateral weighting on position and orientation.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `neighbor_radius` | `2.0` | Search radius (mm) for finding neighboring points. Also used as the voxel grid cell size. |
| `min_neighbors` | `10` | Minimum number of neighbors within the radius. Points with fewer neighbors are removed as noise. |
| `sigma_position` | `0.1` | Position bandwidth (mm) for the bilateral weight. Controls how tightly points cluster spatially. |
| `sigma_orientation` | `30.0` | Orientation bandwidth (degrees) for the bilateral weight. Controls sensitivity to direction differences. |
| `convergence` | `0.002` | Convergence threshold (mm). Mean-shift stops when the point moves less than this per iteration. |
| `max_iterations` | `1000` | Maximum mean-shift iterations per point. |

## `[trace]`

Forward Euler strand tracing from the fused point cloud.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_size` | `0.1` | Step size (mm) for tracing along the strand direction. |
| `neighbor_radius` | `0.1` | Search radius (mm) for finding nearby points at each tracing step. Points within this radius are averaged to determine the next position and direction. |
| `angle_threshold` | `30.0` | Maximum direction change (degrees) between consecutive tracing steps. Steps exceeding this terminate the strand. |
| `min_strand_length` | `2.0` | Minimum strand length (mm). Strands shorter than this are discarded during tracing. Also used as the **lookahead distance** for the forward-Euler step, so changing it affects both the discard threshold and the tracing geometry. |

## `[clean]`

Post-processing to remove short strands and spatially isolated outliers.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_length` | `5.0` | Minimum strand length (mm). Strands shorter than this are removed. |
| `outlier_radius` | `10.0` | Radius (mm) for outlier detection. A strand is an outlier if it has too few neighboring strands within this radius. Set to 0 to disable. |
| `outlier_min_neighbors` | `3` | Minimum number of distinct neighboring strands within `outlier_radius` to keep a strand. |

## `[grow]`

Multi-view growing extends both tips of every cleaned strand. At each step it
scores candidate projected directions against the Gabor orientation fields,
triangulates a supported 3D direction, and stops when image support or geometric
continuity is lost. Orientation/variance maps are cached in
`output_dir/.grow_cache/`; stale or missing maps are regenerated automatically.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `step_size` | `0.1` | Distance (mm) advanced by each accepted growth step. |
| `cone_half_angle` | `5.0` | Half-angle (degrees) of the candidate direction cone around the projected tip tangent. |
| `direction_sample_step` | `1.0` | Angular spacing (degrees) between candidate directions inside the cone. |
| `window_width` | `3` | Width (pixels) of each forward directional scoring window. |
| `window_length` | `10` | Forward length (pixels) of each directional scoring window. |
| `max_pixel_angle` | `5.0` | Maximum modulo-180-degree orientation error (degrees) for an image sample to support a candidate. |
| `min_scored_pixels` | `10` | Minimum number of valid, matching pixels required for a view to support a candidate direction. |
| `min_views` | `8` | Minimum number of supporting views required to triangulate and accept a growth step. At least this many readable, compatible views must be available. |
| `max_direction_change` | `45.0` | Maximum 3D direction change (degrees) between consecutive growth steps. |
| `irls_iterations` | `2` | Number of stabilized IRLS refinements used for robust multi-view direction fitting. |
| `max_growth_length` | `100.0` | Per-tip growth safety limit (mm). |
| `use_mask` | `false` | Require mask foreground for individual samples and point-support votes. Mask failures remove that view's vote rather than vetoing the point globally. |
| `min_intensity` | `0.0` | Minimum normalized image intensity for individual samples and point-support votes. `0` disables intensity gating. When masks are also enabled, samples must pass both gates. |

## `[debug]`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_id` | `0` | Starting GPU device index. With `num_gpus = N`, uses GPUs `gpu_id` through `gpu_id + N - 1`. |
| `log_level` | `"info"` | Logging verbosity: `"error"`, `"warn"`, `"info"`, or `"debug"`. |
| `log_file` | `""` | Path to log file. If empty, logs to stdout (info/debug) and stderr (error/warn) only. If set, all levels are also written to the file with timestamps. |
| `save_intermediates` | `false` | If `true`, save per-view orientation maps, variance maps, cost maps, and point clouds to `output_dir/view_<cam_id>/`. |
| `profile` | `true` | If `true`, print timing breakdowns for each pipeline stage. |

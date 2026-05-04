# Strand-accurate Multi-view Hair Capture

Official implementation of the [CVPR 2019 paper](https://openaccess.thecvf.com/content_CVPR_2019/papers/Nam_Strand-Accurate_Multi-View_Hair_Capture_CVPR_2019_paper.pdf):

> **Strand-accurate Multi-view Hair Capture**
> Giljoo Nam, Chenglei Wu, Min H. Kim, Yaser Sheikh
> *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019*

End-to-end pipeline for capturing strand-level hair geometry from multi-view images: PatchMatch MVS with line-based matching, GPU mean-shift filtering, forward Euler strand tracing, and mesh generation. Note: multi-view hair growing (Section 6 of the paper) is not included in this release.

## Build

**Prerequisites:** CUDA 12.0+, CMake 3.18+, Eigen3, OpenMP, GPU with compute capability 8.0+

```bash
git clone --recursive <repo-url>
cd <repo>

# Linux
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release .. && cmake --build . -j$(nproc)
cd ..

# Windows (requires Visual Studio 2022+ and Ninja)
build_windows.bat
```

If you forgot `--recursive`: `git submodule update --init --recursive`

### Example data (Multiface)

Test the pipeline using the [Multiface](https://github.com/facebookresearch/multiface) dataset (up to 150 cameras):

```bash
# Clone the multiface repo alongside this one (needed for camera configs)
git clone https://github.com/facebookresearch/multiface.git ../multiface

# Download + convert (subject 5067077, 146 cameras) — requires: requests
python scripts/prepare_multiface.py --subject_id 5067077
```

## Run

```bash
# Full pipeline (MVS → mean-shift → trace → clean → mesh)
./build/hair_recon pipeline configs/5067077.toml

# Individual stages
./build/hair_recon mvs       configs/5067077.toml
./build/hair_recon meanshift configs/5067077.toml
./build/hair_recon trace     configs/5067077.toml
./build/hair_recon clean     configs/5067077.toml
./build/hair_recon mesh      configs/5067077.toml

# Visualize results — requires: trimesh, pyrender, numpy, pillow
python scripts/visualize.py configs/5067077.toml
```

Set `num_gpus` in the config to distribute MVS and mean-shift across multiple GPUs. See [`configs/README.md`](configs/README.md) for input data format and full parameter reference.

## Dependencies

All bundled as git submodules (except Eigen3 and OpenMP):
[nanoflann](https://github.com/jlblancoc/nanoflann) (KD-tree),
[toml++](https://github.com/marzer/tomlplusplus) (config),
[stb](https://github.com/nothings/stb) (image I/O),
[tinyexr](https://github.com/syoyo/tinyexr) (EXR),
[nlohmann/json](https://github.com/nlohmann/json) (JSON).
See [`third_party/THIRD_PARTY_VERSIONS.md`](third_party/THIRD_PARTY_VERSIONS.md) for pinned versions.

## Citation

```bibtex
@inproceedings{nam2019strand,
  title={Strand-accurate Multi-view Hair Capture},
  author={Nam, Giljoo and Wu, Chenglei and Kim, Min H. and Sheikh, Yaser},
  booktitle={IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2019}
}
```

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

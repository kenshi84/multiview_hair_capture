# Third-Party Library Versions

All third-party libraries are pulled in as Git submodules pinned to the
specific upstream commits below. None of them are modified locally — every
file under each submodule path is exactly what upstream ships at the pinned
revision.

To populate them after a fresh clone:

```bash
git submodule update --init --recursive
```

The `--recursive` is required because `tinyexr` itself has a nested submodule
(`deps/ZFP`).

## Pinned revisions

| Submodule path | Upstream repo | Tag / version | Pinned commit |
|---|---|---|---|
| `third_party/json` | https://github.com/nlohmann/json | `v3.12.0` | `55f93686c01528224f448c19128836e7df245f72` |
| `third_party/nanoflann` | https://github.com/jlblancoc/nanoflann | `v1.9.0` | `92911c0bc382e4b287330219bc720ca2b30b2857` |
| `third_party/stb` | https://github.com/nothings/stb | (no tag — pins `stb_image` v2.30, `stb_image_write` v1.16, `stb_image_resize2` v2.18) | `904aa67e1e2d1dec92959df63e700b166d5c1022` |
| `third_party/tinyexr` | https://github.com/syoyo/tinyexr | post-v1.0.12 master snapshot | `935140f5a6826f6ddc9a5609b4b2057950a6595c` |
| `third_party/tomlplusplus` | https://github.com/marzer/tomlplusplus | post-v3.4.0 master snapshot | `1c8b7466e4946fcc3bf20484c0e1d001202cca5a` |
| `third_party/tinyexr/deps/ZFP` *(nested)* | https://github.com/LLNL/ZFP | n/a | `300e77d12f25d3eaa4ba9461d937fb17a71d45f6` |

## Updating a submodule

To bump one submodule to a new upstream commit:

```bash
cd third_party/<name>
git fetch
git checkout <new-tag-or-sha>
cd ../..
git add third_party/<name>
git commit -m "<name>: bump to <new-tag-or-sha>"
```

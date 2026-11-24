# ffs_infer

Minimal C++/CUDA/TensorRT runtime for Fast-FoundationStereo. It uses the
fast single-engine path with an embedded `FFSGWCVolume` TensorRT plugin.
The model returns a dense **left-referenced disparity map** in input-image
pixel units.

This derivative retains the upstream NVIDIA license in `LICENSE.txt`. The
research release is limited to non-commercial use; use NVIDIA's commercial
model/code offering when deploying commercially.

## Layout

```text
include/ffs_infer.hpp       public API (the only public header)
sec/ffs_infer.cpp           TensorRT runtime and CPU/GPU APIs
sec/ffs_gwc_plugin.cpp      required TensorRT GWC plugin
sec/ffs_kernels.cu          GPU preprocessing/plugin/postprocessing
tools/build_engine.cpp      ONNX -> FP16 TensorRT engine builder
../../../robots/common/models/               ONNX (tracked) and engine (built on first use) of AirSLAM2
ffs.yaml                    runtime configuration
main.cpp                    OpenCV test program
main_test.cpp               batch sequence-to-depth test program
```

## 1. Export the model

The deployment repo intentionally excludes PyTorch and the model training
sources. Export from the adjacent `Fast-FoundationStereo` repository using the
smallest published checkpoint (`20-30-48`):

```bash
cd ../Fast-FoundationStereo
python3 scripts/make_plugin_onnx.py \
  --model_dir weights/20-30-48/model_best_bp2_serialize.pth \
  --save_path <AirSLAM2>/robots/common/models \
  --height 384 --width 512 --valid_iters 4 --max_disp 192
```

The checkpoint is not present in the source repository and must be downloaded
first as described by its README.

## 2. Build

Use the CUDA architecture of the deployment GPU:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DFFS_CUDA_ARCHITECTURES=86
cmake --build build -j
```

Common architecture values: RTX 30 = `86`, Jetson Orin = `87`, RTX 40 = `89`,
A100 = `80`, H100 = `90`.

Both `ffs_demo` and `ffs_main_test` automatically build the engine on first
use if the `engine` file configured in `ffs.yaml` does not exist. Place the
exported ONNX model at the configured `onnx` path first. Relative paths are
resolved against the directory containing `ffs.yaml`. Existing engines are
loaded without rebuilding, even if the ONNX model changes. To rebuild after
changing the model or configuration, remove the old engine first.

The default ONNX path, when omitted from the configuration, is
`fast_foundationstereo_plugin.onnx` beside the engine. Automatic building uses
FP16 with a 4096 MiB workspace limit. Failed builds do not leave a partial
engine at the configured path. Engine files and temporary engine files are
excluded by `.gitignore`.

You can also build the engine explicitly on the target machine (or an
identical GPU/TensorRT system):

```bash
./build/ffs_build_engine \
  ../../../robots/common/models/fast_foundationstereo_plugin.onnx \
  ../../../robots/common/models/fast_foundationstereo.engine
```

`ffs.yaml` must use the same width and height used for ONNX export.

## 3. Test

```bash
./build/ffs_demo ffs.yaml left.png right.png disp_vis.png
```

Inputs must be rectified, undistorted, same-sized stereo BGR images. Do not
swap left and right images.

Process the VisualInertialLeg sequence one stereo pair at a time. With no arguments, the program
uses `ffs.yaml`, the `grass_seq1_slope_flat_20260810` sequence path configured
in `main_test.cpp`, and a maximum depth of 100 meters:

```bash
./build/ffs_main_test
```

Or specify all inputs explicitly:

```bash
./build/ffs_main_test ffs.yaml /path/to/sequence 100
```

It reads matched `cam0/data` and `cam1/data` images, loads `fx` and the stereo
baseline from `camera.yaml`, and writes `depth_ffs/<timestamp>.png`. Each file
is an original-resolution, single-channel **uint16 PNG in centimeters**:
`depth_m = pixel_value / 100.0`. Invalid or out-of-range depth is zero.
The maximum supported depth limit is 655.35 m (default 100 m).
`depth_ffs/metadata.txt` records the encoding. Read PNGs with
`cv::IMREAD_UNCHANGED` / `cv2.IMREAD_UNCHANGED` to preserve all 16 bits.

Each frame also produces `depth_ffs_color/<timestamp>.png`, an 8-bit color
visualization that opens in ordinary image viewers. The TURBO colormap shows
near depths in blue/cool colors, far depths in red/warm colors, and invalid
pixels in black. A fixed logarithmic scale from 0 to `max_depth_m` improves
near-field contrast and keeps colors consistent across frames. This color PNG
is for viewing; use the uint16 PNG in `depth_ffs` for numerical depth.

Each iteration loads one stereo pair, runs synchronous inference, saves its PNG,
and prints that frame's timing before loading the next pair. Images are not
batched or preloaded; only filenames are collected for sorting. Existing NPY
outputs from previous runs are left untouched.

The batch test requires `yaml-cpp` (`sudo apt-get install libyaml-cpp-dev` on
Ubuntu). Its calibration reader accepts JSON or YAML, including `null` values
in unused metadata fields. Required fields are `cam0.k` (9 entries, with `fx`
first), `cam0.width`, `cam0.height`, and `stereo.baseline_m`.

## API

The convenient CPU API accepts BGR host images and returns `std::vector<float>`:

```cpp
ffs_infer::FFSInfer model("ffs.yaml");
std::vector<float> disparity;
model.infer(left.data, right.data, width, height, disparity, row_stride);
```

For the lowest latency, keep camera images and output on the GPU:

```cpp
model.inferDevice(d_left_bgr, d_right_bgr, width, height, d_disparity);
// Queue downstream CUDA work on model.stream(), or synchronize when required.
model.synchronize();
```

Create `FFSInfer` once and reuse it. `inferDevice()` is asynchronous and avoids
host/device image and disparity copies. The class is not thread-safe; use one
instance per concurrent stream.

CUDA runtime and TensorRT runtime are required. OpenCV is needed for the demo
and batch test. When available at compile time, the TensorRT ONNX parser is
linked into the runtime to support automatic engine building and the explicit
engine-builder tool. Builds without the parser can only load existing engines.

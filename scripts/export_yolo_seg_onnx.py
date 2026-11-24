#!/usr/bin/env python3
"""Export the YOLO11n-seg instance segmentation model used by the online dynamic-object masks
(algorithm.yaml visual_odometry.dynamic_mask, off by default) to robots/common/models/yolo11n_seg.onnx.

The repository ships this ONNX (Ultralytics weights, AGPL-3.0); the script re-creates it. Requires the ultralytics package
(pip install ultralytics onnx onnxslim); yolo11n-seg.pt is downloaded on first use. The ONNX has a dynamic input size:
the VO node builds one TensorRT engine per input size (dynamic_mask.input_size) and caches it next to the ONNX.

usage: export_yolo_seg_onnx.py [--weights yolo11n-seg.pt] [--output robots/common/models/yolo11n_seg.onnx]
"""
import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--weights', default='yolo11n-seg.pt')
    ap.add_argument('--output', type=Path, default=ROOT / 'robots/common/models/yolo11n_seg.onnx')
    a = ap.parse_args()
    from ultralytics import YOLO
    exported = Path(YOLO(a.weights).export(format='onnx', dynamic=True, simplify=True, opset=17, imgsz=416, batch=1))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(exported), a.output)
    print(f'wrote {a.output}')


if __name__ == '__main__':
    main()

# Go2 meshes

Visual meshes of the Unitree Go2 used by `../model_go2.xml` (`meshdir="assets"`) and by the RViz demo URDF that
`ros2_packages/airslam/scripts/mjcf_to_urdf.py` generates from that model.

- Source: MuJoCo Menagerie `unitree_go2/assets` (https://github.com/google-deepmind/mujoco_menagerie), downloaded
  2026-11-18. License: `LICENSE` (BSD-3-Clause, Unitree Robotics).
- Change: the `mtllib` / `usemtl` lines were removed (the referenced `.mtl` files are not distributed), so RViz uses
  the URDF colours (taken from the model's materials) instead of a default grey. Geometry is unchanged.

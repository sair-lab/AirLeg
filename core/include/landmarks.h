#ifndef LANDMARKS_H_
#define LANDMARKS_H_

// Selected landmarks of the S3 back-end window (BackendConfig visual_mode landmarks): SelectWindow picks
// <= `max_points` map points observed by the window's nodes (BackendNode::obs) by track length and span in the
// window, one per voxel, keeping the previous selection first. The selected points enter the window as point
// vertices with reprojection edges; the other points are marginalised (visual_marg win_pair).

#include <vector>

#include "fusion_backend.h"

struct WindowSettings {
  int max_points = 100;
  int min_nodes = 3;          // observed by at least this many nodes of the window
  double voxel_size = 0.5;
};

// Selected landmark ids among the observations of nodes [lo, end) (`nodes`), previous selection kept first.
std::vector<long> SelectWindow(const std::vector<BackendNode>& nodes, size_t lo, size_t end, const WindowSettings& s,
                               const std::vector<long>& previous);

#endif  // LANDMARKS_H_

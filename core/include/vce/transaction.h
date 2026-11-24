#pragma once
#include <Eigen/Core>
#include <functional>
#include <string>
#include <exception>

namespace vce {
// A failed solve must restore BOTH state and information, including exceptions.
struct StateUpdate {
  std::function<void()> save, restore, commit;
  std::function<void(const Eigen::VectorXd&)> weights;
  std::function<bool()> optimize;
};
inline bool ApplyCandidate(const Eigen::VectorXd& current, const Eigen::VectorXd& candidate,
                           const StateUpdate& state, std::string& reason) {
  state.save();
  try {
    state.weights(candidate);
    if (state.optimize()) { state.commit(); return true; }
    reason="final-weight optimization failed";
  } catch (const std::exception& e) { reason=std::string("candidate optimization failed: ")+e.what(); }
  state.restore();
  state.weights(current);  // Always reconstruct from immutable base information.
  reason+="; restored last valid state and weights";
  return false;
}
}  // namespace vce

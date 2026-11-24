# anymal_d_simple_description (meshes only)

Meshes of ANYmal D from https://github.com/ANYbotics/anymal_d_simple_description (commit d5bf4dc,
BSD 3-Clause, see LICENSE), for the RViz view robot_viz (robots/anymal/viz.yaml -> model/anymal.urdf). anymal.urdf refers
to them as package://anymal_d_simple_description/meshes/...; without that ROS package installed, robot_viz looks for the
package directory next to the model. Not used by the estimation (the kinematic model is anymal_d_kinematic.urdf).

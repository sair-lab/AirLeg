# ANYbotics anymal_msgs/Contact as in the GrandTour (ANYmal D) recordings. ROS 2 needs lower-case field names: the ROS 1
# frictionCoefficient / restitutionCoefficient are friction_coefficient / restitution_coefficient here; CDR does not
# encode field names, so the recorded messages deserialize unchanged.
uint8 STATE_OPEN=0
uint8 STATE_CLOSED=1
uint8 STATE_SLIPPING=2
std_msgs/Header header
string name
uint8 state
geometry_msgs/Wrench wrench
geometry_msgs/Point position
geometry_msgs/Vector3 normal
float64 friction_coefficient
float64 restitution_coefficient

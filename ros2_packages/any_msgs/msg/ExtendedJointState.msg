# ANYbotics any_msgs/ExtendedJointState, field order and types as in the GrandTour (ANYmal D) recordings, so that the
# recorded CDR messages deserialize unchanged.
std_msgs/Header header
string[] name
float64[] position
float64[] velocity
float64[] acceleration
float64[] effort

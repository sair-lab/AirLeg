# ANYbotics anymal_msgs/AnymalState as in the GrandTour (ANYmal D) recordings (/anymal/state_estimator/anymal_state):
# state estimate of the base and the joint states. The leg odometry reads joints (robots/anymal/leg_odometry.yaml
# robot: leg_inputs).
int8 STATE_ERROR_SENSOR=-3
int8 STATE_ERROR_ESTIMATOR=-2
int8 STATE_ERROR_UNKNOWN=-1
int8 STATE_OK=0
int8 STATE_UNINITIALIZED=1
std_msgs/Header header
int8 state
geometry_msgs/PoseStamped pose
geometry_msgs/TwistStamped twist
any_msgs/ExtendedJointState joints
anymal_msgs/Contact[] contacts
geometry_msgs/TransformStamped[] frame_transforms

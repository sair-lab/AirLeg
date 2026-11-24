# Online leg odometry from the S3 velocity network (airslam online_leg_odom).
std_msgs/Header header          # sensor (low-state) time, not shifted by time_delay
float64[6] twist                # [vx vy vz wx wy wz] of the robot base frame B (URDF base, not the IMU), gyro bias removed
float64[6] variance             # per-axis variance; linear part includes velocity_scale^2
float64[3] gyro_bias            # bias subtracted from the gyro rotated into base axes (rad/s, B)
float32[12] foot_b              # FL,FR,RL,RR foot-sphere centres in the base frame B (m)
float32[4] pressure             # normalised foot pressure, FL,FR,RL,RR
uint8[4] contact                # hysteresis contact state
uint32[4] contact_segment       # increments at every touchdown
uint32 model_version            # increments whenever online adapter weights change

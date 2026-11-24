
#ifndef OPTIMIZATION_3D_TYPES_H_
#define OPTIMIZATION_3D_TYPES_H_

#include <istream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/types/slam3d_addons/types_slam3d_addons.h>

#include "utils.h"
#include "imu.h"

struct Pose3d {
  bool fixed;
  int id_camera;
  Eigen::Vector3d p;
  Eigen::Matrix3d R;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Pose3d() {}
  Pose3d& operator =(Pose3d& other){
		fixed = other.fixed;
		id_camera = other.id_camera;
		p = other.p;
		R = other.R;
		return *this;
	}
};
typedef std::map<int, Pose3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Pose3d>>> MapOfPoses;


struct Position3d{
  bool fixed;
  Eigen::Vector3d p;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Position3d() {}
  Position3d& operator =(Position3d& other){
		fixed = other.fixed;
		p = other.p;
		return *this;
	}
};
typedef std::map<int, Position3d, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Position3d>>> MapOfPoints3d;


struct MonoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left
  Eigen::Vector2d keypoint;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MonoPointConstraint() {}
  MonoPointConstraint& operator =(MonoPointConstraint& other){
		id_pose = other.id_pose;
		id_point = other.id_point;
		id_camera = other.id_camera;
		inlier = other.inlier;
		keypoint = other.keypoint;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<MonoPointConstraint> MonoPointConstraintPtr;
typedef std::vector<MonoPointConstraintPtr> VectorOfMonoPointConstraints;


struct StereoPointConstraint {
  int id_pose;
  int id_point;
  int id_camera;
  bool inlier;

  // x_left, y_left, x_right
  Eigen::Vector3d keypoint;  
  double pixel_sigma;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  StereoPointConstraint() {}
  StereoPointConstraint& operator =(StereoPointConstraint& other){
		id_pose = other.id_pose;
		id_point = other.id_point;
		id_camera = other.id_camera;
		inlier = other.inlier;
		keypoint = other.keypoint;
		pixel_sigma = other.pixel_sigma;
		return *this;
	}
};
typedef std::shared_ptr<StereoPointConstraint> StereoPointConstraintPtr;
typedef std::vector<StereoPointConstraintPtr> VectorOfStereoPointConstraints;


// for imu
struct Velocity{
  bool fixed;
  Eigen::Vector3d velocity;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Velocity() {}
  Velocity& operator =(Velocity& other){
		fixed = other.fixed;
		velocity = other.velocity;
		return *this;
	}
};
typedef std::map<int, Velocity, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Velocity>>> MapOfVelocity;


struct Bias{
  bool fixed;
  Eigen::Vector3d gyr_bias;
  Eigen::Vector3d acc_bias;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  Bias() {}
  Bias& operator =(Bias& other){
		fixed = other.fixed;
		gyr_bias = other.gyr_bias;
		acc_bias = other.acc_bias;
		return *this;
	}
};
typedef std::map<int, Bias, std::less<int>, Eigen::aligned_allocator<std::pair<const int, Bias>>> MapOfBias;


struct ImuConstraint{
  int id_pose1;
  int id_pose2;
  int id_camera1;
  int id_camera2;
  PreinterationPtr preinteration;
  
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuConstraint() {}
  ImuConstraint& operator =(ImuConstraint& other){
		id_pose1 = other.id_pose1;
		id_pose2 = other.id_pose2;
		id_camera1 = other.id_camera1;
		id_camera2 = other.id_camera2;
		preinteration = other.preinteration;
		return *this;
	}
};
typedef std::shared_ptr<ImuConstraint> IMUConstraintPtr;
typedef std::vector<IMUConstraintPtr> VectorOfIMUConstraints;

struct RelativePoseConstraint{
  int id_pose1;
  int id_pose2;
  int id_camera1;
  int id_camera2;

  Eigen::Matrix3d Rc1c2;
  Eigen::Vector3d tc1c2;
  bool robust;
  bool active;
  double information_scale;

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  RelativePoseConstraint(): robust(false), active(true), information_scale(1.0) {}
  RelativePoseConstraint& operator =(RelativePoseConstraint& other){
		id_pose1 = other.id_pose1;
		id_pose2 = other.id_pose2;
		id_camera1 = other.id_camera1;
		id_camera2 = other.id_camera2;
		Rc1c2 = other.Rc1c2;
		tc1c2 = other.tc1c2;
		robust = other.robust;
		active = other.active;
		information_scale = other.information_scale;
		return *this;
	}
};
typedef std::shared_ptr<RelativePoseConstraint> RelativePoseConstraintPtr;
typedef std::vector<RelativePoseConstraintPtr> VectorOfRelativePoseConstraints;

// ---- S3 online fusion inputs (all optional; empty means the original BA) ----
struct LegRelativeConstraint{
  int id_pose1, id_pose2;
  Eigen::Isometry3d T_c1_c2;                 // camera-frame relative pose from leg odometry
  Eigen::Matrix<double, 6, 6> information;   // already divided by the online VCE scale
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::vector<LegRelativeConstraint, Eigen::aligned_allocator<LegRelativeConstraint>> VectorOfLegConstraints;

struct FootSurfaceConstraint{
  int id_pose, id_anchor;
  Eigen::Vector3d foot_b, q_c, n_c;          // foot centre in B; plane point/normal in anchor camera frame
  double radius, sigma;
  double rmse = 0.0;                         // plane fit RMSE (back-end quality gate)
  bool inlier = true;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
typedef std::vector<FootSurfaceConstraint, Eigen::aligned_allocator<FootSurfaceConstraint>> VectorOfFootConstraints;

struct LocalBAExtras{
  Eigen::Isometry3d T_C_B = Eigen::Isometry3d::Identity();
  VectorOfLegConstraints legs;
  double leg_huber = 2.5;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

#endif  // OPTIMIZATION_3D_TYPES_H_

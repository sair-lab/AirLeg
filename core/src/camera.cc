#include <cmath>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <Eigen/Core>
#include <opencv2/core/eigen.hpp>

#include "camera.h"
#include "utils.h"

double Camera::IMU_G_VALUE = 9.81;

Camera::Camera(){
}

Camera::Camera(const std::string& camera_file){
  if(!FileExists(camera_file)){
    throw std::runtime_error("Camera config file doesn't exist: " + camera_file);
  }

  YAML::Node file_node = YAML::LoadFile(camera_file);
  _image_height = file_node["image_height"].as<int>();
  _image_width = file_node["image_width"].as<int>();
  _depth_lower_thr = file_node["depth_lower_thr"].as<double>();
  _depth_upper_thr = file_node["depth_upper_thr"].as<double>();
  _max_y_diff = file_node["max_y_diff"].as<double>();

  cv::Mat K0, K1, D0, D1, R10, t10;
  cv::Mat R0, R1, P0, P1, Q;
  Eigen::Matrix4d Tbc0, Tbc1;

  YAML::Node cam0_node = file_node["cam0"];
  YAML::Node cam1_node = file_node["cam1"];
  ReadCameraNode(cam0_node, K0, D0, Tbc0);
  ReadCameraNode(cam1_node, K1, D1, Tbc1);

  Eigen::Matrix4d Tc1c0 = Tbc1.inverse() * Tbc0;
  _Tbc = Tbc0;
  _Tcb = _Tbc.inverse();

  int distortion_type = file_node["distortion_type"].as<int>();
  if(distortion_type == 0){
    _fx = K0.at<double>(0, 0);
    _fy = K0.at<double>(1, 1);
    _cx = K0.at<double>(0, 2);
    _cy = K0.at<double>(1, 2);
    _fx_inv = 1.0 / _fx;
    _fy_inv = 1.0 / _fy;

    _bf = _fx * std::abs(Tc1c0(0, 3));
    _max_x_diff = _bf / _depth_lower_thr;
    _min_x_diff = _bf / _depth_upper_thr;
  }else{
    Eigen::Matrix3d R10_eigen = Tc1c0.block<3, 3>(0, 0);
    Eigen::Vector3d t10_eigen = Tc1c0.block<3, 1>(0, 3);
    cv::eigen2cv(R10_eigen, R10);
    cv::eigen2cv(t10_eigen, t10);

    cv::Size image_size(_image_width, _image_height);
    if(distortion_type == 1){
      cv::stereoRectify(K0, D0, K1, D1, image_size, R10, t10, R0, R1, P0, P1, Q,
                        cv::CALIB_ZERO_DISPARITY, 0, image_size);

      cv::initUndistortRectifyMap(K0, D0, R0, P0.rowRange(0,3).colRange(0,3), 
          image_size, CV_32F, _mapl1, _mapl2);
      cv::initUndistortRectifyMap(K1, D1, R1, P1.rowRange(0,3).colRange(0,3),
          image_size, CV_32F, _mapr1, _mapr2);
    }else{
      cv::fisheye::stereoRectify(K0, D0.rowRange(0,4), K1, D1.rowRange(0,4), image_size, R10, t10, R0, R1, P0, P1, Q,
                                 cv::CALIB_ZERO_DISPARITY, image_size, 0, 0.8);

      cv::fisheye::initUndistortRectifyMap(K0, D0.rowRange(0,4), R0, P0.rowRange(0,3).colRange(0,3), 
          image_size, CV_32F, _mapl1, _mapl2);
      cv::fisheye::initUndistortRectifyMap(K1, D1.rowRange(0,4), R1, P1.rowRange(0,3).colRange(0,3),
          image_size, CV_32F, _mapr1, _mapr2);
    }

    _bf = std::abs(P1.at<double>(0, 3));
    _max_x_diff = _bf / _depth_lower_thr;
    _min_x_diff = _bf / _depth_upper_thr;
    _fx = P0.at<double>(0, 0);
    _fy = P0.at<double>(1, 1);
    _cx = P0.at<double>(0, 2);
    _cy = P0.at<double>(1, 2);
    _fx_inv = 1.0 / _fx;
    _fy_inv = 1.0 / _fy;
  }

  // IMU
  _use_imu = file_node["use_imu"].as<int>();
  // the noise model is also read for imu_mode > 0 without VIO (back-end IMU factors); unused by the front-end
  const bool imu_noise = _use_imu || (file_node["imu_mode"] && file_node["imu_mode"].as<int>() > 0 && file_node["rate_hz"]);
  if(imu_noise){
    _imu_frequency = file_node["rate_hz"].as<double>();
    _gyr_noise = file_node["gyroscope_noise_density"].as<double>();
    _acc_noise = file_node["accelerometer_noise_density"].as<double>();
    _gyr_walk = file_node["gyroscope_random_walk"].as<double>();
    _acc_walk = file_node["accelerometer_random_walk"].as<double>();
    IMU_G_VALUE = file_node["g_value"].as<double>();

    double frequency_sqrt = std::sqrt(_imu_frequency);
    _gyr_noise = _gyr_noise* frequency_sqrt;
    _acc_noise = _acc_noise * frequency_sqrt;
    _gyr_walk = _gyr_walk / frequency_sqrt;
    _acc_walk = _acc_walk / frequency_sqrt;
  }else{  // not used without IMU, but never pass uninitialised values on
    _imu_frequency = 0.0;
    _gyr_noise = _acc_noise = _gyr_walk = _acc_walk = 0.0;
  }
  _imu_mode = file_node["imu_mode"] ? file_node["imu_mode"].as<int>() : (_use_imu ? 1 : 0);
  if(_imu_mode == 1 && !_use_imu) throw std::runtime_error("imu_mode 1 (VIO) needs use_imu: 1");
  if(_imu_mode > 0){
    if(!file_node["T_imu_cam0"]) throw std::runtime_error("imu_mode > 0 needs T_imu_cam0 (cam0 pose in the IMU frame)");
    Eigen::Matrix4d T_ic;
    for(int i = 0; i < 4; i++) for(int j = 0; j < 4; j++) T_ic(i, j) = file_node["T_imu_cam0"][i][j].as<double>();
    _R_ci = T_ic.block<3, 3>(0, 0).transpose();
    _T_ic = T_ic;
    if(file_node["attitude_kp"]) _attitude_kp = file_node["attitude_kp"].as<double>();
    if(file_node["attitude_ki"]) _attitude_ki = file_node["attitude_ki"].as<double>();
  }
}

Camera& Camera::operator=(const Camera& camera){
  _image_height = camera._image_height;
  _image_width = camera._image_width;
  _use_imu = camera._use_imu;
  _imu_mode = camera._imu_mode;
  _R_ci = camera._R_ci;
  _T_ic = camera._T_ic;
  _attitude_kp = camera._attitude_kp;
  _attitude_ki = camera._attitude_ki;
  _bf = camera._bf;
  _depth_lower_thr = camera._depth_lower_thr;
  _depth_upper_thr = camera._depth_upper_thr;
  _max_x_diff = _bf / _depth_lower_thr;
  _min_x_diff = _bf / _depth_upper_thr;
  _max_y_diff = camera._max_y_diff;
  _fx = camera._fx;
  _fy = camera._fy;
  _cx = camera._cx;
  _cy = camera._cy;
  _fx_inv = camera._fx_inv;
  _fy_inv = camera._fy_inv;
  _mapl1 = camera._mapl1.clone();
  _mapl2 = camera._mapl2.clone();
  _mapr1 = camera._mapr1.clone();
  _mapr2 = camera._mapr2.clone();

  _Tbc = camera._Tbc;
  _Tcb = camera._Tcb;
  _gyr_noise = camera._gyr_noise;
  _acc_noise = camera._acc_noise;
  _gyr_walk = camera._gyr_walk;
  _acc_walk = camera._acc_walk;
  return *this;
}

void Camera::ReadCameraNode(YAML::Node& cam_node, cv::Mat& K, cv::Mat& D, Eigen::Matrix4d& Tbc){
  Eigen::Matrix3d K_eigen;
  Vector5d D_eigen;
  K_eigen << cam_node["intrinsics"][0].as<double>(), 0, cam_node["intrinsics"][2].as<double>(), 0, 
        cam_node["intrinsics"][1].as<double>(), cam_node["intrinsics"][3].as<double>(), 
        0, 0, 1;
  D_eigen << cam_node["distortion_coeffs"][0].as<double>(), cam_node["distortion_coeffs"][1].as<double>(), 
             cam_node["distortion_coeffs"][2].as<double>(), cam_node["distortion_coeffs"][3].as<double>(), 
             cam_node["distortion_coeffs"][4].as<double>();

  for(size_t i = 0; i < 4; i++){
    for(size_t j = 0; j < 4; j++){
      Tbc(i, j) = cam_node["T"][i][j].as<double>();
    }
  }

  int T_type = cam_node["T_type"].as<double>();
  if(T_type){
    Tbc = Tbc.inverse();
  }

  cv::eigen2cv(K_eigen, K);
  cv::eigen2cv(D_eigen, D);
}

void Camera::UndistortImage(cv::Mat& image_left, cv::Mat& image_left_rect){
  if(!_mapl1.empty() && !_mapl2.empty()){
    cv::remap(image_left, image_left_rect, _mapl1, _mapl2, cv::INTER_LINEAR);
  }else{
    image_left_rect = image_left;
  }
}

void Camera::UndistortMask(const cv::Mat& mask, cv::Mat& mask_rect, bool right) const{
  const cv::Mat& map1 = right ? _mapr1 : _mapl1;
  const cv::Mat& map2 = right ? _mapr2 : _mapl2;
  if(mask.empty()){
    mask_rect = cv::Mat();
  }else if(!map1.empty() && !map2.empty()){
    cv::remap(mask, mask_rect, map1, map2, cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));
  }else{
    mask_rect = mask;
  }
}

void Camera::UndistortImage(
    cv::Mat& image_left, cv::Mat& image_right, cv::Mat& image_left_rect, cv::Mat& image_right_rect){
  if(!_mapl1.empty() && !_mapl2.empty()){
    cv::remap(image_left, image_left_rect, _mapl1, _mapl2, cv::INTER_LINEAR);
  }else{
    image_left_rect = image_left;
  }

  if(!_mapr1.empty() && !_mapr2.empty()){
    cv::remap(image_right, image_right_rect, _mapr1, _mapr2, cv::INTER_LINEAR);
  }else{
    image_right_rect = image_right;
  }
}

double Camera::ImageHeight(){
  return _image_height;
}

double Camera::ImageWidth(){
  return _image_width;
}

bool Camera::UseIMU(){
  return _use_imu;
}

double Camera::BF(){
  return _bf;
}

double Camera::Fx(){
  return _fx;
}

double Camera::Fy(){
  return _fy;
}

double Camera::Cx(){
  return _cx;
}

double Camera::Cy(){
  return _cy;
}

double Camera::MaxXDiff(){
  return _max_x_diff;
}

double Camera::MinXDiff(){
  return _min_x_diff;
}

double Camera::MaxYDiff(){
  return _max_y_diff;
}

Eigen::Matrix4d Camera::CameraToBody(){
  return _Tbc;
}

Eigen::Matrix4d Camera::BodyToCamera(){
  return _Tcb;
}

double Camera::GyrNoise(){
  return _gyr_noise;
}

double Camera::AccNoise(){
  return _acc_noise;
}

double Camera::GyrWalk(){
  return _gyr_walk;
}

double Camera::AccWalk(){
  return _acc_walk;
}

void Camera::GetCamerMatrix(cv::Mat& camera_matrix){
  camera_matrix = (cv::Mat_<double>(3, 3) << _fx, 0.0, _cx, 0.0, _fy, _cy, 0.0, 0.0, 1.0);
}

void Camera::GetDistCoeffs(cv::Mat& dist_coeffs){
  dist_coeffs = (cv::Mat_<double>(5, 1) << 0.0, 0.0, 0.0, 0.0, 0.0);
}

bool Camera::BackProjectMono(const Eigen::Vector2d& keypoint, Eigen::Vector3d& output){
  output(0) = (keypoint(0) - _cx) * _fx_inv;
  output(1) = (keypoint(1) - _cy) * _fy_inv;
  output(2) = 1.0;
  return true;
}

bool Camera::BackProjectStereo(const Eigen::Vector3d& keypoint, Eigen::Vector3d& output){
  BackProjectMono(keypoint.head(2), output);
  double d = _bf / (keypoint(0) - keypoint(2));
  output = output * d;
  return true;
}
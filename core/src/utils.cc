
#include "utils.h"
#include <dirent.h>
#include <filesystem>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <numeric> 
#include <algorithm>
#include <cmath>

// (f1 - f2) * (f1 - f2) = f1 * f1 + f2 * f2 - 2 * f1 *f2 = 2 - 2 * f1 * f2 -> [0, 4]
float DescriptorDistance(const Eigen::Matrix<float, 256, 1>& f1, const Eigen::Matrix<float, 256, 1>& f2){
  return 2 * (1.0 - f1.transpose() * f2);
}

std::string DoubleTimeToString(double timestamp_seconds){
  // auto timestamp_nanoseconds = static_cast<long long>(timestamp_seconds * 1e9);
  // std::ostringstream oss;
  // oss << timestamp_nanoseconds;
  // return oss.str();

  std::ostringstream stream;
  stream << std::setprecision(9) << std::fixed << timestamp_seconds;
  return stream.str();
}

double StringTimeToDouble(std::string time_str){
  time_str.erase(std::remove_if(time_str.begin(), time_str.end(), [](char c) { return c == '.'; }), time_str.end());

  double time_double1 = atof(time_str.substr(0, 10).c_str());
  double time_double2 = atof(("0." + time_str.substr(10, time_str.length()+1)).c_str());
  return (time_double1 + time_double2);
}

double ImageNameToTime(const std::string& image_name){
  std::string time_str;
  size_t pos = image_name.find_last_of('.');
  if (pos != std::string::npos) {
    time_str = image_name.substr(0, pos);
  } 
  return StringTimeToDouble(time_str);
}

double CalculateStdDev(const std::vector<double>& data) {
  if (data.empty()) return 0.0;
  double mean = std::accumulate(data.begin(), data.end(), 0.0) / data.size();

  double variance = 0.0;
  for (const auto& value : data) {
    variance += (value - mean) * (value - mean);
  }
  variance /= data.size();
  return std::sqrt(variance);
}

cv::Mat DrawFeatures(const cv::Mat& image,
    const std::vector<cv::KeyPoint>& keypoints){
  cv::Mat color;
  cv::cvtColor(image, color, cv::COLOR_GRAY2RGB);
  for(const auto& keypoint : keypoints){
    cv::circle(color, keypoint.pt, 2, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
  }
  return color;
}

cv::Mat DrawMatches(const cv::Mat& ref_image, const cv::Mat& image, const std::vector<cv::KeyPoint>& ref_kpts, 
    const std::vector<cv::KeyPoint>& kpts, const std::vector<cv::DMatch>& matches){

  cv::Mat merged_image;
  cv::hconcat(ref_image, image, merged_image);
  cv::Mat rgba_image;
  cv::cvtColor(merged_image, rgba_image, cv::COLOR_BGR2BGRA);

  for (size_t i = 0; i < matches.size(); i++) {
    cv::line(rgba_image, cv::Point2f(ref_kpts[matches[i].queryIdx].pt.x + image.cols, ref_kpts[matches[i].queryIdx].pt.y),  
              cv::Point2f(kpts[matches[i].trainIdx].pt.x + ref_image.cols, kpts[matches[i].trainIdx].pt.y), 
              cv::Scalar(0,255,0, 10), 2);    
  }
  cv::Mat result;
  cv::cvtColor(rgba_image, result, cv::COLOR_BGRA2BGR);

  return result;
}

void GetFileNames(std::string path, std::vector<std::string>& filenames){
  DIR *pDir;
  struct dirent* ptr;
  if(!(pDir = opendir(path.c_str()))){
    std::cout << "Folder doesn't Exist!" << std::endl;
    return;
  }
  while((ptr = readdir(pDir))!= 0) {
    if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
      filenames.push_back(ptr->d_name);
    }
  }
  closedir(pDir);
}

bool FileExists(const std::string& file) {
  struct stat file_status;
  if (stat(file.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFREG)) {
    return true;
  }
  return false;
}


bool PathExists(const std::string& path) {
  struct stat file_status;
  if (stat(path.c_str(), &file_status) == 0 &&
      (file_status.st_mode & S_IFDIR)) {
    return true;
  }
  return false;
}

void ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name,
    std::string* path) {
  *path = folder;
  if (path->back() != '/') {
    *path += '/';
  }
  *path = *path + file_name;
}

std::string ConcatenateFolderAndFileName(
    const std::string& folder, const std::string& file_name) {
  std::string path;
  ConcatenateFolderAndFileName(folder, file_name, &path);
  return path;
}

void ClearDir(const std::string& path, const std::string& preserved_name){
  if(path.empty() || path == "/") return;
  std::filesystem::create_directories(path);
  for(const auto& entry : std::filesystem::directory_iterator(path)){
    if(entry.path().filename() == preserved_name) continue;
    std::filesystem::remove_all(entry.path());
  }
}

void ReadTxt(const std::string& file_path, 
    std::vector<std::vector<std::string> >& lines, std::string seq){
  if(!FileExists(file_path)){
    std::cout << "file: " << file_path << " dosen't exist" << std::endl;
    exit(0);
  }
  
  std::ifstream infile(file_path, std::ifstream::in);   
  if(!infile.is_open()){
    std::cout << "open file: " << file_path << " failure" << std::endl;
    exit(0);
  }

  std::string line;
  while (getline(infile, line)){ 
    std::string whitespaces(" \t\f\v\n\r");
    std::size_t found = line.find_last_not_of(whitespaces);
    if (found!=std::string::npos){
      line.erase(found+1);

      std::vector<std::string> line_data;
      while (true){
        int index = line.find(seq);
        std::string sub_str = line.substr(0, index);
        if (!sub_str.empty()){
          line_data.push_back(sub_str);
        }

        line.erase(0, index + seq.size());
        if (index == -1){
          break;
        }
      }
      lines.emplace_back(line_data);
    }
    else{
      line.clear();            // str is all whitespace
    }
  }
}

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<double, Eigen::Matrix4d>>& trajectory){
  std::cout << "Save file to " << file_path << std::endl;
  std::ofstream f;
  f.open(file_path.c_str());
  f << std::fixed;
  std::cout << "trajectory.size = " << trajectory.size() << std::endl;
  for(const auto& pose_pair : trajectory){
    Eigen::Vector3d t = pose_pair.second.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose_pair.second.block<3, 3>(0, 0));

    f << std::setprecision(9) << pose_pair.first << " " 
      << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  f.close();
}

void SaveTumTrajectoryToFile(const std::string file_path, 
    const std::vector<std::pair<std::string, Eigen::Matrix4d>>& trajectory){
  std::cout << "Save file to " << file_path << std::endl;
  std::ofstream f;
  f.open(file_path.c_str());
  f << std::fixed;
  std::cout << "trajectory.size = " << trajectory.size() << std::endl;
  for(const auto& pose_pair : trajectory){
    Eigen::Vector3d t = pose_pair.second.block<3, 1>(0, 3);
    Eigen::Quaterniond q(pose_pair.second.block<3, 3>(0, 0));

    f << pose_pair.first << " " 
      << std::setprecision(9) << t(0) << " " << t(1) << " " << t(2) << " "
      << q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
  }
  f.close();
}

TumTrajectory ReadTumTrajectory(const std::string& path){
  TumTrajectory trajectory;
  std::ifstream f(path);
  double t, x, y, z, qx, qy, qz, qw;
  while(f >> t >> x >> y >> z >> qx >> qy >> qz >> qw){
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.topLeftCorner<3, 3>() = Eigen::Quaterniond(qw, qx, qy, qz).normalized().toRotationMatrix();
    T.topRightCorner<3, 1>() = Eigen::Vector3d(x, y, z);
    trajectory.emplace_back(t, T);
  }
  std::sort(trajectory.begin(), trajectory.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
  return trajectory;
}

const std::pair<double, Eigen::Matrix4d>* NearestPose(const TumTrajectory& trajectory, double t, double tolerance){
  auto it = std::lower_bound(trajectory.begin(), trajectory.end(), t,
                             [](const auto& a, double v){ return a.first < v; });
  const std::pair<double, Eigen::Matrix4d>* best = nullptr;
  double d = tolerance;
  for(auto c : {it, it == trajectory.begin() ? it : std::prev(it)})
    if(c != trajectory.end() && std::abs(c->first - t) < d){ d = std::abs(c->first - t); best = &*c; }
  return best;
}

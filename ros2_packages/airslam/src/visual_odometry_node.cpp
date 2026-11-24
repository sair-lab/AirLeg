#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <geometry_msgs/msg/vector3_stamped.hpp>
#include <lowstate_republisher/msg/leg_odometry.hpp>
#include <lowstate_republisher/msg/low_state_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/compressed_image.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <yaml-cpp/yaml.h>

#include "map_builder.h"
#include "read_configs.h"
#include "node_params.h"
#include "ros2_publisher.h"
#include "utils.h"

namespace {

int64_t ToNs(const builtin_interfaces::msg::Time& stamp) {
  return static_cast<int64_t>(stamp.sec) * 1000000000LL + stamp.nanosec;
}

double ToSec(int64_t ns) {
  return static_cast<double>(ns) * 1e-9;
}

// One camera's image stream as received: sensor_msgs/Image or sensor_msgs/CompressedImage (decoded only when the
// frame is used) with the stamp of its header.
struct StampedImage {
  int64_t ns = 0;
  sensor_msgs::msg::Image::ConstSharedPtr image;
  sensor_msgs::msg::CompressedImage::ConstSharedPtr compressed;
};

// How a camera's messages become the image of the camera yaml (keys of the camera yaml):
//   image_message: image | compressed (empty: compressed when the topic ends with /compressed);
//   image_width x image_height: images of another size are resized to it (INTER_AREA; e.g. GrandTour Alphasense
//   recorded at 1440x1080, processed at 720x540), so the VO chooses its resolution.
struct ImageInput {
  bool compressed = false;
  cv::Size size;
};

cv::Mat ToMono8(const StampedImage& msg, const ImageInput& input) {
  cv::Mat gray;
  if (msg.compressed) {
    const cv::Mat data(1, static_cast<int>(msg.compressed->data.size()), CV_8UC1,
                       const_cast<uint8_t*>(msg.compressed->data.data()));
    gray = cv::imdecode(data, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) throw std::runtime_error("cannot decode compressed image (" + msg.compressed->format + ")");
  } else {
    gray = cv_bridge::toCvCopy(msg.image, "mono8")->image.clone();
  }
  if (input.size.area() > 0 && gray.size() != input.size) {
    cv::Mat resized;
    cv::resize(gray, resized, input.size, 0, 0, cv::INTER_AREA);
    gray = resized;
  }
  return gray;
}

ImageInput ReadImageInput(const YAML::Node& camera_root, const std::string& topic) {
  ImageInput input;
  std::string message = camera_root["image_message"] ? camera_root["image_message"].as<std::string>() : "";
  const std::string suffix = "/compressed";
  if (message.empty()) {
    message = topic.size() >= suffix.size() && topic.compare(topic.size() - suffix.size(), suffix.size(), suffix) == 0
                  ? "compressed" : "image";
  }
  if (message != "image" && message != "compressed") {
    throw std::runtime_error("camera yaml image_message must be image or compressed: " + message);
  }
  input.compressed = message == "compressed";
  if (camera_root["image_width"] && camera_root["image_height"]) {
    input.size = cv::Size(camera_root["image_width"].as<int>(), camera_root["image_height"].as<int>());
  }
  return input;
}


std::string ReadRequiredCameraString(const std::string& camera_config_path,
                                     const std::string& key) {
  if (camera_config_path.empty()) {
    throw std::runtime_error("camera_config_path is empty, cannot read " + key);
  }
  const YAML::Node camera_root = YAML::LoadFile(camera_config_path);
  if (!camera_root[key]) {
    throw std::runtime_error("missing required camera config key: " + key);
  }
  return camera_root[key].as<std::string>();
}

}  // namespace

class VisualOdometryTopicNode : public rclcpp::Node {
public:
  VisualOdometryTopicNode() : Node("visual_odometry") {
    declare_parameter<std::string>("config_path", "");
    declare_parameter<std::string>("model_dir", "");
    declare_parameter<std::string>("camera_config_path", "");
    declare_parameter<std::string>("saving_dir", "");
    declare_parameter<std::string>("left_image_topic", "");
    declare_parameter<std::string>("right_image_topic", "");
    declare_parameter<std::string>("imu_topic", "");
    // IMU message type on imu_topic: "imu" (sensor_msgs/Imu) or "lowstate" (the gyroscope / accelerometer of
    // lowstate_republisher/LowStateStamped, the IMU the leg odometry uses); empty: camera yaml imu_message, else "imu".
    declare_parameter<std::string>("imu_message", "");
    // "true" / "false" overrides visual_odometry.use_imu of the algorithm config; empty keeps it
    declare_parameter<std::string>("use_imu", "");
    declare_parameter<double>("sync_tolerance_sec", 0.01);
    declare_parameter<int>("image_queue_size", 30);
    declare_parameter<int>("imu_queue_size", 2000);
    declare_parameter<double>("imu_max_gap_sec", 0.02);
    // S3: empty keeps visual_odometry.online.mode from the algorithm config.
    declare_parameter<std::string>("online_mode", "");
    // Dynamic-object masks: <dir>/{left,right}/<stamp_ns>.png (non-zero = dynamic); empty disables.
    declare_parameter<std::string>("dynamic_mask_dir", "");
  }

  bool UseLeg() const { return static_cast<bool>(online_); }

  void Initialize() {
    const std::string config_path = get_parameter("config_path").as_string();
    const YAML::Node root = LoadAlgorithmConfig(config_path);
    const std::string model_dir =
        PreferParamOrConfig(get_parameter("model_dir").as_string(), root, "model_dir");
    const std::string camera_config_path =
        PreferParamOrConfig(get_parameter("camera_config_path").as_string(), root, "camera_config_path");
    saving_dir_ = get_parameter("saving_dir").as_string();

    left_topic_ = get_parameter("left_image_topic").as_string();
    if (left_topic_.empty()) {
      left_topic_ = ReadRequiredCameraString(camera_config_path, "left_image_topic");
    }
    right_topic_ = get_parameter("right_image_topic").as_string();
    if (right_topic_.empty()) {
      right_topic_ = ReadRequiredCameraString(camera_config_path, "right_image_topic");
    }
    imu_topic_ = get_parameter("imu_topic").as_string();
    if (imu_topic_.empty()) {
      imu_topic_ = ReadRequiredCameraString(camera_config_path, "imu_topic");
    }
    imu_message_ = get_parameter("imu_message").as_string();
    if (imu_message_.empty() && !camera_config_path.empty()) {
      const YAML::Node camera_root = YAML::LoadFile(camera_config_path);
      if (camera_root["imu_message"]) imu_message_ = camera_root["imu_message"].as<std::string>();
    }
    if (imu_message_.empty()) imu_message_ = "imu";
    if (imu_message_ != "imu" && imu_message_ != "lowstate") {
      throw std::runtime_error("imu_message must be imu or lowstate: " + imu_message_);
    }
    {
      const YAML::Node camera_root = camera_config_path.empty() ? YAML::Node() : YAML::LoadFile(camera_config_path);
      left_input_ = ReadImageInput(camera_root, left_topic_);
      right_input_ = ReadImageInput(camera_root, right_topic_);
    }
    sync_tolerance_ns_ =
        static_cast<int64_t>(get_parameter("sync_tolerance_sec").as_double() * 1e9);
    image_queue_size_ = std::max(1, static_cast<int>(get_parameter("image_queue_size").as_int()));
    imu_queue_size_ = std::max(1, static_cast<int>(get_parameter("imu_queue_size").as_int()));
    imu_max_gap_sec_ = std::max(0.0, get_parameter("imu_max_gap_sec").as_double());
    dynamic_mask_dir_ = get_parameter("dynamic_mask_dir").as_string();
    if (!dynamic_mask_dir_.empty()) {
      if (!std::filesystem::is_directory(std::filesystem::path(dynamic_mask_dir_) / "left")) {
        throw std::runtime_error("dynamic_mask_dir has no left/ directory: " + dynamic_mask_dir_);
      }
      RCLCPP_INFO(get_logger(), "Dynamic-object masks: %s", dynamic_mask_dir_.c_str());
    }

    VisualOdometryConfigs configs(config_path, model_dir);
    if (!camera_config_path.empty()) {
      configs.camera_config_path = camera_config_path;
    }
    if (!saving_dir_.empty()) {
      configs.saving_dir = saving_dir_;
    }
    const std::string use_imu = get_parameter("use_imu").as_string();
    if (use_imu == "true" || use_imu == "false") configs.SetUseImu(use_imu == "true");
    else if (!use_imu.empty()) throw std::runtime_error("use_imu must be true or false: " + use_imu);
    const std::string online_mode = get_parameter("online_mode").as_string();
    if (!online_mode.empty()) configs.online.SetMode(online_mode);
    if (configs.dynamic_mask.enabled && !dynamic_mask_dir_.empty()) {
      throw std::runtime_error("dynamic_mask.enabled (online YOLO masks) and dynamic_mask_dir (offline masks) exclude each other");
    }

    publisher_ = std::make_shared<Ros2Publisher>(configs.ros_publisher_config, shared_from_this());
    map_builder_ = std::make_unique<MapBuilder>(configs, publisher_);
    use_imu_ = map_builder_->UseImuData();   // VIO or the gravity-only attitude filter (imu_mode > 0)

    left_sub_ = SubscribeImage(left_topic_, left_input_, true);
    right_sub_ = SubscribeImage(right_topic_, right_input_, false);
    RCLCPP_INFO(get_logger(), "images: %s (%s), %s (%s) -> %dx%d", left_topic_.c_str(),
                left_input_.compressed ? "compressed" : "raw", right_topic_.c_str(),
                right_input_.compressed ? "compressed" : "raw", left_input_.size.width, left_input_.size.height);

    if (use_imu_ && imu_message_ == "lowstate") {
      // best effort: also matches publishers recorded / played back best effort (the Go2 bags)
      lowstate_imu_sub_ = create_subscription<lowstate_republisher::msg::LowStateStamped>(
          imu_topic_, rclcpp::SensorDataQoS().keep_last(500),
          [this](lowstate_republisher::msg::LowStateStamped::ConstSharedPtr msg) {
            ImuData imu;
            imu.timestamp = ToSec(ToNs(msg->header.stamp));
            imu.gyr = Eigen::Vector3d(msg->gyroscope[0], msg->gyroscope[1], msg->gyroscope[2]);
            imu.acc = Eigen::Vector3d(msg->accelerometer[0], msg->accelerometer[1], msg->accelerometer[2]);
            AddImu(imu);
          });
    } else if (use_imu_) {
      // best effort with a deep queue, like the lowstate IMU: matches reliable and best-effort publishers (bag
      // playback, drivers) and absorbs playback bursts
      imu_sub_ = create_subscription<sensor_msgs::msg::Imu>(
          imu_topic_, rclcpp::SensorDataQoS().keep_last(500),
          [this](sensor_msgs::msg::Imu::ConstSharedPtr msg) { OnImu(msg); });
    }
    if (use_imu_) RCLCPP_INFO(get_logger(), "IMU: %s (%s)", imu_topic_.c_str(), imu_message_.c_str());

    online_ = map_builder_->Online();
    total_bias_ = configs.online.backend.gyro_bias_total;
    if (online_) SubscribeLeg(configs.online.leg_topic);
    if (online_ && configs.online.backend.leg_reliability) SubscribeAccel(configs.online.accel_topic);
    if (online_ && online_->Backend()) PublishBackendPath(configs.ros_publisher_config.path_topic + "/backend");
  }

  // Robot accelerometer for the back-end leg reliability features (vibration).
  void SubscribeAccel(const std::string& topic) {
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    accel_sub_ = create_subscription<lowstate_republisher::msg::LowStateStamped>(
        topic, rclcpp::SensorDataQoS().keep_last(500),
        [this](lowstate_republisher::msg::LowStateStamped::ConstSharedPtr msg) {
          online_->PushAccel(ToSec(ToNs(msg->header.stamp)),
                             Eigen::Vector3d(msg->accelerometer[0], msg->accelerometer[1], msg->accelerometer[2]));
        },
        options);
  }

  // Back-end keyframe path after every back-end solve (supervision of the online LoRA, online.leg_lora).
  void PublishBackendPath(const std::string& topic) {
    backend_path_pub_ = create_publisher<nav_msgs::msg::Path>(topic, 10);
    // backend.gyro_bias_total: the back-end's total gyro bias (newest block, B) for the worker after every solve
    if (total_bias_) backend_bias_pub_ = create_publisher<geometry_msgs::msg::Vector3Stamped>("/airslam/backend/gyro_bias", 10);
    // weak: the back-end's final solve at shutdown may run after this node's members are gone
    std::weak_ptr<rclcpp::Publisher<nav_msgs::msg::Path>> weak = backend_path_pub_;
    std::weak_ptr<rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>> weak_bias = backend_bias_pub_;
    FusionBackend* backend = online_->Backend();
    online_->Backend()->SetPathCallback([weak, weak_bias, backend](const std::vector<std::pair<double, Eigen::Isometry3d>>& traj) {
      auto bias_pub = weak_bias.lock();
      if (bias_pub && !traj.empty() && rclcpp::ok()) {   // called from the solve thread: LatestGyroBias is consistent
        geometry_msgs::msg::Vector3Stamped b;
        b.header.stamp.sec = static_cast<int32_t>(std::floor(traj.back().first));
        b.header.stamp.nanosec = static_cast<uint32_t>((traj.back().first - std::floor(traj.back().first)) * 1e9);
        const Eigen::Vector3d g = backend->LatestGyroBias();
        b.vector.x = g.x(); b.vector.y = g.y(); b.vector.z = g.z();
        bias_pub->publish(b);
      }
      auto pub = weak.lock();
      if (!pub || traj.empty() || !rclcpp::ok() || pub->get_subscription_count() == 0) return;
      nav_msgs::msg::Path path;
      path.header.frame_id = "map";
      for (const auto& p : traj) {
        geometry_msgs::msg::PoseStamped pose;
        pose.header.frame_id = "map";
        pose.header.stamp.sec = static_cast<int32_t>(std::floor(p.first));
        pose.header.stamp.nanosec = static_cast<uint32_t>((p.first - std::floor(p.first)) * 1e9);
        const Eigen::Quaterniond q(p.second.linear());
        pose.pose.position.x = p.second.translation().x();
        pose.pose.position.y = p.second.translation().y();
        pose.pose.position.z = p.second.translation().z();
        pose.pose.orientation.x = q.x(); pose.pose.orientation.y = q.y();
        pose.pose.orientation.z = q.z(); pose.pose.orientation.w = q.w();
        path.poses.push_back(pose);
      }
      path.header.stamp = path.poses.back().header.stamp;
      pub->publish(path);
    });
  }

  // Leg samples go straight into the thread-safe stream from their own callback group, so a
  // blocked image callback never delays them.
  void SubscribeLeg(const std::string& topic) {
    rclcpp::SubscriptionOptions options;
    options.callback_group = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    leg_sub_ = create_subscription<lowstate_republisher::msg::LegOdometry>(
        topic, rclcpp::SensorDataQoS().keep_last(500),
        [this](lowstate_republisher::msg::LegOdometry::ConstSharedPtr msg) {
          LegVelocitySample sample;
          sample.time = ToSec(ToNs(msg->header.stamp));
          FootState foot;
          for (int i = 0; i < 3; ++i) {
            sample.linear[i] = msg->twist[i];
            sample.angular[i] = msg->twist[i + 3];
            sample.gyro_bias[i] = msg->gyro_bias[i];
          }
          // backend.gyro_bias_total: keep the worker's bias in the rate; the back-end estimates the total bias
          if (total_bias_) sample.angular += sample.gyro_bias;
          for (int i = 0; i < 6; ++i) sample.variance[i] = msg->variance[i];
          // a non-finite twist / variance would become a (near) infinitely confident leg factor: drop the sample
          bool finite = std::isfinite(sample.time) && sample.linear.allFinite() && sample.angular.allFinite() &&
                        sample.variance.allFinite();
          for (float v : msg->foot_b) finite = finite && std::isfinite(v);
          if (!finite) {
            if (++dropped_leg_samples_ % 100 == 1)
              RCLCPP_WARN(get_logger(), "dropped %zu non-finite leg odometry samples", dropped_leg_samples_.load());
            return;
          }
          for (int i = 0; i < 6; ++i) sample.variance[i] = std::max(1e-12, sample.variance[i]);
          for (int leg = 0; leg < 4; ++leg) {
            foot.foot_b[leg] = Eigen::Vector3d(msg->foot_b[3 * leg], msg->foot_b[3 * leg + 1], msg->foot_b[3 * leg + 2]);
            foot.contact[leg] = msg->contact[leg];
            foot.segment[leg] = msg->contact_segment[leg];
          }
          online_->Stream().Push(sample, foot);
        },
        options);
    extrinsic_sub_ = create_subscription<geometry_msgs::msg::TransformStamped>(
        topic + "/extrinsic", rclcpp::QoS(1).transient_local().reliable(),
        [this](geometry_msgs::msg::TransformStamped::ConstSharedPtr msg) {
          const auto& t = msg->transform;
          Eigen::Isometry3d T_C_B = Eigen::Isometry3d::Identity();
          T_C_B.linear() = Eigen::Quaterniond(t.rotation.w, t.rotation.x, t.rotation.y, t.rotation.z)
                               .normalized().toRotationMatrix();
          T_C_B.translation() << t.translation.x, t.translation.y, t.translation.z;
          online_->Stream().SetExtrinsic(T_C_B);
          RCLCPP_INFO(get_logger(), "S3 leg extrinsic T_C_B received from the leg node");
        },
        options);
    RCLCPP_INFO(get_logger(), "S3 online fusion: subscribed to %s", topic.c_str());
  }

  ~VisualOdometryTopicNode() override {
    ShutdownMapBuilder();
  }

  void Shutdown() {
    ShutdownMapBuilder();
  }

private:
  // Missing file = no dynamic object in that image.
  cv::Mat LoadMask(const char* side, int64_t stamp_ns, const cv::Size& size) const {
    const std::filesystem::path path =
        std::filesystem::path(dynamic_mask_dir_) / side / (std::to_string(stamp_ns) + ".png");
    if (!std::filesystem::exists(path)) return cv::Mat();
    cv::Mat mask = cv::imread(path.string(), cv::IMREAD_GRAYSCALE);
    if (mask.empty() || mask.size() != size) {
      throw std::runtime_error("bad dynamic mask (unreadable or size mismatch): " + path.string());
    }
    return mask;
  }

  rclcpp::SubscriptionBase::SharedPtr SubscribeImage(const std::string& topic, const ImageInput& input, bool left) {
    if (input.compressed) {
      return create_subscription<sensor_msgs::msg::CompressedImage>(
          topic, 20, [this, left](sensor_msgs::msg::CompressedImage::ConstSharedPtr msg) {
            OnImage(StampedImage{ToNs(msg->header.stamp), nullptr, msg}, left);
          });
    }
    return create_subscription<sensor_msgs::msg::Image>(
        topic, 20, [this, left](sensor_msgs::msg::Image::ConstSharedPtr msg) {
          OnImage(StampedImage{ToNs(msg->header.stamp), msg, nullptr}, left);
        });
  }

  void OnImage(StampedImage msg, bool left) {
    std::lock_guard<std::mutex> lock(mutex_);
    (left ? left_queue_ : right_queue_).push_back(std::move(msg));
    TrimImageQueuesLocked();
    TrySyncLocked();
  }

  void OnImu(const sensor_msgs::msg::Imu::ConstSharedPtr& msg) {
    ImuData imu;
    imu.timestamp = ToSec(ToNs(msg->header.stamp));
    imu.gyr = Eigen::Vector3d(
        msg->angular_velocity.x,
        msg->angular_velocity.y,
        msg->angular_velocity.z);
    imu.acc = Eigen::Vector3d(
        msg->linear_acceleration.x,
        msg->linear_acceleration.y,
        msg->linear_acceleration.z);
    AddImu(imu);
  }

  void AddImu(const ImuData& imu) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++imu_messages_received_;
    if (!imu_queue_.empty() && imu.timestamp <= imu_queue_.back().timestamp) {
      ++imu_out_of_order_;
      return;
    }
    imu_queue_.push_back(imu);
    while (imu_queue_.size() > static_cast<size_t>(imu_queue_size_)) {
      imu_queue_.pop_front();
      ++imu_queue_drops_;
    }
    TrySyncLocked();
  }

  void TrySyncLocked() {
    while (map_builder_ && !left_queue_.empty() && !right_queue_.empty()) {
      const int64_t left_ns = left_queue_.front().ns;
      const int64_t right_ns = right_queue_.front().ns;
      const int64_t dt = left_ns - right_ns;

      if (std::llabs(dt) > sync_tolerance_ns_) {
        if (dt < 0) {
          left_queue_.pop_front();
        } else {
          right_queue_.pop_front();
        }
        continue;
      }

      const double image_time = ToSec(left_ns);
      if (use_imu_ && imu_queue_.empty()) {
        RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "waiting for IMU on %s (%s): no sample received yet",
                             imu_topic_.c_str(), imu_message_.c_str());
      }
      if (use_imu_ &&
          (imu_queue_.empty() || imu_queue_.back().timestamp < image_time)) {
        // Wait for the first IMU measurement on or after the image timestamp.
        // OnImu() calls TrySyncLocked() again when that boundary sample arrives.
        return;
      }

      if (use_imu_ && frame_index_ == 0 &&
          (imu_queue_.size() < 100 || imu_queue_.front().timestamp > image_time - 0.5)) {
        // The first frame needs >= 0.5 s of IMU history (attitude filter / complete first preintegration): the IMU
        // stream may connect after the images, so the image pairs before that are dropped.
        left_queue_.pop_front();
        right_queue_.pop_front();
        ++dropped_for_imu_history_;
        continue;
      }

      auto left = left_queue_.front();
      auto right = right_queue_.front();
      left_queue_.pop_front();
      right_queue_.pop_front();

      InputDataPtr data = std::make_shared<InputData>();
      data->index = frame_index_++;
      data->time = image_time;
      data->image_left = ToMono8(left, left_input_);
      data->image_right = ToMono8(right, right_input_);
      if (!dynamic_mask_dir_.empty()) {
        data->mask_left = LoadMask("left", left_ns, data->image_left.size());
        data->mask_right = LoadMask("right", right_ns, data->image_right.size());
      }

      if (use_imu_) {
        auto upper = std::lower_bound(
            imu_queue_.begin(), imu_queue_.end(), data->time,
            [](const ImuData& imu, double timestamp) {
              return imu.timestamp < timestamp;
            });
        if (upper != imu_queue_.end()) {
          data->batch_imu_data.assign(imu_queue_.begin(), std::next(upper));
        }

        const double interval_start = has_last_image_time_ ? last_image_time_ : data->time;
        double batch_max_gap = 0.0;
        for (size_t i = 1; i < data->batch_imu_data.size(); ++i) {
          batch_max_gap = std::max(
              batch_max_gap,
              data->batch_imu_data[i].timestamp-data->batch_imu_data[i-1].timestamp);
        }
        max_observed_imu_gap_sec_ = std::max(max_observed_imu_gap_sec_, batch_max_gap);
        data->imu_data_valid = data->batch_imu_data.size() >= 2 &&
            data->batch_imu_data.front().timestamp <= interval_start &&
            data->batch_imu_data.back().timestamp >= data->time &&
            batch_max_gap <= imu_max_gap_sec_;

        ++imu_batches_;
        imu_samples_batched_ += data->batch_imu_data.size();
        min_imu_samples_per_batch_ = std::min(
            min_imu_samples_per_batch_, data->batch_imu_data.size());
        max_imu_samples_per_batch_ = std::max(
            max_imu_samples_per_batch_, data->batch_imu_data.size());
        if (!data->imu_data_valid) {
          ++invalid_imu_batches_;
          if (invalid_imu_batches_ == 1 || invalid_imu_batches_ % 100 == 0) {
            RCLCPP_WARN(
                get_logger(),
                "invalid IMU batch at %.9f: interval_start=%.9f, samples=%zu, "
                "first=%.9f, last=%.9f, max_gap=%.6f",
                data->time, interval_start, data->batch_imu_data.size(),
                data->batch_imu_data.empty() ? 0.0 : data->batch_imu_data.front().timestamp,
                data->batch_imu_data.empty() ? 0.0 : data->batch_imu_data.back().timestamp,
                batch_max_gap);
          }
        }

        // Retain the last sample at or before this image. Together with the
        // already queued right-boundary sample, it brackets the next interval.
        while (imu_queue_.size() > 1 &&
               imu_queue_[1].timestamp <= data->time) {
          imu_queue_.pop_front();
        }
        last_image_time_ = data->time;
        has_last_image_time_ = true;
      }

      map_builder_->AddInput(data);
      ++stereo_frames_submitted_;
    }
  }

  void TrimImageQueuesLocked() {
    while (left_queue_.size() > static_cast<size_t>(image_queue_size_)) {
      left_queue_.pop_front();
    }
    while (right_queue_.size() > static_cast<size_t>(image_queue_size_)) {
      right_queue_.pop_front();
    }
  }

  void ShutdownMapBuilder() {
    if (!map_builder_) return;

    if (use_imu_) {
      const size_t min_samples = imu_batches_ == 0 ? 0 : min_imu_samples_per_batch_;
      RCLCPP_INFO(
          get_logger(),
          "IMU batching summary: received=%zu, out_of_order=%zu, queue_drops=%zu, "
          "batches=%zu, invalid_batches=%zu, samples_total=%zu, samples_per_batch=[%zu,%zu], "
          "max_gap=%.6f s, stereo_frames=%zu",
          imu_messages_received_, imu_out_of_order_, imu_queue_drops_, imu_batches_,
          invalid_imu_batches_, imu_samples_batched_, min_samples,
          max_imu_samples_per_batch_, max_observed_imu_gap_sec_, stereo_frames_submitted_);
    }

    if (!saving_dir_.empty()) {
      std::error_code ec;
      std::filesystem::create_directories(saving_dir_, ec);
      if (ec) {
        RCLCPP_ERROR(get_logger(), "failed to create saving_dir %s: %s",
                     saving_dir_.c_str(), ec.message().c_str());
        map_builder_.reset();
        return;
      }
      RCLCPP_INFO(get_logger(), "stopping VO and will save trajectory/map to %s",
                  saving_dir_.c_str());
    } else {
      RCLCPP_WARN(get_logger(), "saving_dir is empty; trajectory/map will not be saved");
    }

    map_builder_->Stop();
    while (!map_builder_->IsStopped()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    if (!saving_dir_.empty()) {
      const std::string trajectory_path =
          ConcatenateFolderAndFileName(saving_dir_, "trajectory_v0.txt");
      RCLCPP_INFO(get_logger(), "saving trajectory and map to %s", saving_dir_.c_str());
      map_builder_->SaveTrajectory(trajectory_path);
      map_builder_->SaveMap(saving_dir_);
    }
    map_builder_.reset();
  }

  std::string left_topic_;
  std::string right_topic_;
  std::string imu_topic_;
  std::string imu_message_;
  std::string saving_dir_;
  std::string dynamic_mask_dir_;
  bool use_imu_ = false;
  size_t dropped_for_imu_history_ = 0;   // image pairs before >= 0.5 s of IMU history (first frame)
  int64_t sync_tolerance_ns_ = 10000000;
  int image_queue_size_ = 30;
  int imu_queue_size_ = 2000;
  double imu_max_gap_sec_ = 0.02;
  size_t frame_index_ = 0;
  bool has_last_image_time_ = false;
  double last_image_time_ = 0.0;
  size_t imu_messages_received_ = 0;
  size_t imu_out_of_order_ = 0;
  size_t imu_queue_drops_ = 0;
  size_t imu_batches_ = 0;
  size_t invalid_imu_batches_ = 0;
  size_t imu_samples_batched_ = 0;
  size_t min_imu_samples_per_batch_ = std::numeric_limits<size_t>::max();
  size_t max_imu_samples_per_batch_ = 0;
  double max_observed_imu_gap_sec_ = 0.0;
  size_t stereo_frames_submitted_ = 0;

  std::mutex mutex_;
  std::deque<StampedImage> left_queue_;
  std::deque<StampedImage> right_queue_;
  ImageInput left_input_;
  ImageInput right_input_;
  std::deque<ImuData> imu_queue_;

  PublisherPtr publisher_;
  std::unique_ptr<MapBuilder> map_builder_;
  rclcpp::SubscriptionBase::SharedPtr left_sub_;
  rclcpp::SubscriptionBase::SharedPtr right_sub_;
  rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_sub_;
  rclcpp::Subscription<lowstate_republisher::msg::LowStateStamped>::SharedPtr lowstate_imu_sub_;
  OnlineFusionPtr online_;
  rclcpp::Subscription<lowstate_republisher::msg::LegOdometry>::SharedPtr leg_sub_;
  rclcpp::Subscription<geometry_msgs::msg::TransformStamped>::SharedPtr extrinsic_sub_;
  rclcpp::Subscription<lowstate_republisher::msg::LowStateStamped>::SharedPtr accel_sub_;
  std::atomic<size_t> dropped_leg_samples_{0};
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr backend_path_pub_;
  rclcpp::Publisher<geometry_msgs::msg::Vector3Stamped>::SharedPtr backend_bias_pub_;
  bool total_bias_ = false;
};

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  auto node = std::make_shared<VisualOdometryTopicNode>();
  node->Initialize();
  if (node->UseLeg()) {
    rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
    executor.add_node(node);
    executor.spin();
  } else {
    rclcpp::spin(node);
  }
  node->Shutdown();
  node.reset();
  rclcpp::shutdown();
  return 0;
}

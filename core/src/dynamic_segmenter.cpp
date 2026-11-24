#include "dynamic_segmenter.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include <opencv2/imgproc.hpp>

#include "cuda_utils.h"
#include "dynamic_segmenter_kernels.h"
#include "trt_engine.h"

namespace {
constexpr int kCoefficients = 32;
constexpr int kCapacity = 4096;   // decoded candidates (score >= confidence) kept on the GPU
int RoundUp32(double v) { return std::max(32, static_cast<int>(std::ceil(v / 32.0)) * 32); }
}  // namespace

struct DynamicSegmenter::Impl {
  DynamicMaskConfig config;
  int width = 0, height = 0;    // image
  int iw = 0, ih = 0;           // network input
  int rw = 0, rh = 0;           // resized image inside the input (top-left, padded right / bottom)
  double scale = 1.0;           // input pixels per image pixel
  int anchors = 0, classes = 0, ph = 0, pw = 0;
  uint64_t class_lo = 0, class_hi = 0;

  lightglue::TRTEngine engine;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  lightglue::CudaStream stream;
  lightglue::CudaBuffer<uint8_t> d_gray;
  lightglue::CudaBuffer<float> d_input, d_out0, d_out1, d_candidates, d_prob;   // d_prob: per-detection ROIs, packed
  lightglue::CudaBuffer<int> d_rois;
  lightglue::CudaBuffer<int> d_count;
  lightglue::PinnedBuffer<uint8_t> h_gray;
  lightglue::PinnedBuffer<float> h_candidates, h_prob;
  lightglue::PinnedBuffer<int> h_rois;
  lightglue::PinnedBuffer<int> h_count;
  cv::Mat resized, kernel;
  bool pending = false;
};

DynamicSegmenter::DynamicSegmenter(const DynamicMaskConfig& config, int image_width, int image_height)
    : impl_(new Impl) {
  Impl& m = *impl_;
  m.config = config;
  m.width = image_width;
  m.height = image_height;
  if (image_width <= 0 || image_height <= 0) throw std::runtime_error("DynamicSegmenter: invalid image size");
  // network input: long side input_size, short side keeps the aspect (multiples of 32, the YOLO stride)
  const double long_side = std::max(image_width, image_height);
  m.scale = std::min(1.0, config.input_size / long_side);
  m.iw = RoundUp32(image_width * m.scale);
  m.ih = RoundUp32(image_height * m.scale);
  m.rw = std::min(m.iw, static_cast<int>(std::lround(image_width * m.scale)));
  m.rh = std::min(m.ih, static_cast<int>(std::lround(image_height * m.scale)));
  for (int c : config.classes) {
    if (c < 0 || c >= 128) throw std::runtime_error("dynamic_mask.classes: class id out of range");
    (c < 64 ? m.class_lo : m.class_hi) |= 1ULL << (c % 64);
  }

  lightglue::EngineConfig engine_config;
  engine_config.onnxPath = config.onnx_file;
  engine_config.tag = std::to_string(m.iw) + "x" + std::to_string(m.ih);   // one engine per input size
  engine_config.fp16 = true;
  lightglue::ProfileDims dims;
  dims.min = dims.opt = dims.max = nvinfer1::Dims4{1, 3, m.ih, m.iw};
  engine_config.profiles["images"] = dims;
  if (!m.engine.build(engine_config)) throw std::runtime_error("DynamicSegmenter: cannot build " + config.onnx_file);
  m.context.reset(m.engine.createContext());
  if (!m.context || !m.context->setInputShape("images", nvinfer1::Dims4{1, 3, m.ih, m.iw}))
    throw std::runtime_error("DynamicSegmenter: cannot set the input shape");
  const nvinfer1::Dims o0 = m.context->getTensorShape("output0"), o1 = m.context->getTensorShape("output1");
  if (o0.nbDims != 3 || o1.nbDims != 4 || o1.d[1] != kCoefficients)
    throw std::runtime_error("DynamicSegmenter: unexpected outputs (YOLO-seg output0 [1, 4+C+32, A], output1 [1, 32, h, w])");
  m.classes = static_cast<int>(o0.d[1]) - 4 - kCoefficients;
  m.anchors = static_cast<int>(o0.d[2]);
  m.ph = static_cast<int>(o1.d[2]);
  m.pw = static_cast<int>(o1.d[3]);

  m.d_gray.resize(static_cast<size_t>(m.rw) * m.rh);
  m.d_input.resize(3ULL * m.iw * m.ih);
  m.d_out0.resize(static_cast<size_t>(o0.d[1]) * m.anchors);
  m.d_out1.resize(static_cast<size_t>(kCoefficients) * m.ph * m.pw);
  m.d_candidates.resize(7ULL * kCapacity);
  m.d_rois.resize(6ULL * config.max_detections);
  m.d_prob.resize(static_cast<size_t>(config.max_detections) * m.ph * m.pw);
  m.d_count.resize(1);
  m.h_gray.reserve(static_cast<size_t>(m.rw) * m.rh);
  m.h_candidates.reserve(7ULL * kCapacity);
  m.h_rois.reserve(6ULL * config.max_detections);
  m.h_prob.reserve(static_cast<size_t>(config.max_detections) * m.ph * m.pw);
  m.h_count.reserve(1);
  m.context->setTensorAddress("images", m.d_input.get());
  m.context->setTensorAddress("output0", m.d_out0.get());
  m.context->setTensorAddress("output1", m.d_out1.get());
  if (config.dilate_px > 0)
    m.kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2 * config.dilate_px + 1, 2 * config.dilate_px + 1));
  std::cout << "[dynamic_mask] YOLO-seg " << config.onnx_file << ": input " << m.iw << "x" << m.ih << " (image "
            << image_width << "x" << image_height << " resized to " << m.rw << "x" << m.rh << "), " << m.classes
            << " classes, " << config.classes.size() << " dynamic, left image only" << std::endl;
}

DynamicSegmenter::~DynamicSegmenter() {
  if (impl_ && impl_->pending) impl_->stream.synchronize();
}

int DynamicSegmenter::InputWidth() const { return impl_->iw; }
int DynamicSegmenter::InputHeight() const { return impl_->ih; }

void DynamicSegmenter::Launch(const cv::Mat& gray) {
  Impl& m = *impl_;
  if (gray.type() != CV_8UC1 || gray.cols != m.width || gray.rows != m.height)
    throw std::runtime_error("DynamicSegmenter: expected a CV_8UC1 image of the configured size");
  if (m.pending) m.stream.synchronize();   // a launched frame that was never collected (skipped frame)
  // resize on the CPU straight into pinned memory (small image), letterbox / normalise on the GPU
  cv::Mat dst(m.rh, m.rw, CV_8UC1, m.h_gray.get());
  cv::resize(gray, dst, dst.size(), 0, 0, cv::INTER_AREA);
  const cudaStream_t s = m.stream.get();
  m.d_gray.copyFromHost(m.h_gray.get(), static_cast<size_t>(m.rw) * m.rh, s);
  yolo_cuda::LaunchLetterbox(m.d_gray.get(), m.rw, m.rh, m.d_input.get(), m.iw, m.ih, s);
  if (!m.context->enqueueV3(s)) throw std::runtime_error("DynamicSegmenter: TensorRT enqueue failed");
  yolo_cuda::LaunchDecode(m.d_out0.get(), m.anchors, m.classes, m.class_lo, m.class_hi, m.config.confidence,
                          m.d_candidates.get(), m.d_count.get(), kCapacity, s);
  m.d_count.copyToHost(m.h_count.get(), 1, s);
  m.pending = true;
}

cv::Mat DynamicSegmenter::Collect() {
  Impl& m = *impl_;
  if (!m.pending) throw std::runtime_error("DynamicSegmenter: Collect() without Launch()");
  const cudaStream_t s = m.stream.get();
  m.stream.synchronize();
  m.pending = false;
  const int found = std::min(m.h_count[0], kCapacity);
  if (found == 0) return cv::Mat();
  m.d_candidates.copyToHost(m.h_candidates.get(), 7ULL * found, s);
  m.stream.synchronize();

  // NMS per class (input pixels), highest score first
  std::vector<int> order(found);
  for (int i = 0; i < found; ++i) order[i] = i;
  const float* c = m.h_candidates.get();
  std::sort(order.begin(), order.end(), [c](int a, int b) { return c[7 * a + 4] > c[7 * b + 4]; });
  std::vector<int> kept;
  for (int i : order) {
    const float* a = c + 7 * i;
    bool suppressed = false;
    for (int j : kept) {
      const float* b = c + 7 * j;
      if (a[5] != b[5]) continue;
      const float iw = std::min(a[2], b[2]) - std::max(a[0], b[0]), ih = std::min(a[3], b[3]) - std::max(a[1], b[1]);
      if (iw <= 0 || ih <= 0) continue;
      const float inter = iw * ih;
      const float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
      if (inter > m.config.iou * uni) { suppressed = true; break; }
    }
    if (!suppressed) kept.push_back(i);
    if (static_cast<int>(kept.size()) >= m.config.max_detections) break;
  }

  // As Ultralytics (retina masks, process_mask_native): each detection's probability is upsampled to the image and
  // then cropped to its box (pixels x1 <= u < x2), threshold 0.5. Here only inside the box: the GPU evaluates the
  // prototype cells the bilinear upsampling of the box reads (box grown by the interpolation footprint), the CPU
  // upsamples them into the box (image pixel u <-> prototype (u + 0.5) scale / 4 - 0.5).
  const int n = static_cast<int>(kept.size());
  const double f = m.scale / 4.0;
  const cv::Rect image(0, 0, m.width, m.height);
  std::vector<cv::Rect> boxes, cells;
  cv::Rect roi;
  int total = 0, used = 0;
  for (int k = 0; k < n; ++k) {
    const float* a = c + 7 * kept[k];
    const cv::Rect box = cv::Rect(cv::Point(static_cast<int>(std::ceil(a[0] / m.scale)), static_cast<int>(std::ceil(a[1] / m.scale))),
                                  cv::Point(static_cast<int>(std::ceil(a[2] / m.scale)), static_cast<int>(std::ceil(a[3] / m.scale)))) & image;
    if (box.area() == 0) continue;
    const int x0 = std::max(0, static_cast<int>(std::floor((box.x + 0.5) * f - 0.5)) - 1);
    const int y0 = std::max(0, static_cast<int>(std::floor((box.y + 0.5) * f - 0.5)) - 1);
    const int x1 = std::min(m.pw, static_cast<int>(std::floor((box.br().x - 0.5) * f - 0.5)) + 2);
    const int y1 = std::min(m.ph, static_cast<int>(std::floor((box.br().y - 0.5) * f - 0.5)) + 2);
    if (x1 <= x0 || y1 <= y0) continue;
    int* q = m.h_rois.get() + 6 * used;
    q[0] = x0; q[1] = y0; q[2] = x1 - x0; q[3] = y1 - y0; q[4] = total; q[5] = static_cast<int>(a[6]);
    total += (x1 - x0) * (y1 - y0);
    boxes.push_back(box); cells.emplace_back(x0, y0, x1 - x0, y1 - y0); ++used;
    roi = roi.area() ? (roi | box) : box;
  }
  if (boxes.empty()) return cv::Mat();
  m.d_rois.copyFromHost(m.h_rois.get(), 6ULL * used, s);
  yolo_cuda::LaunchMaskRois(m.d_out0.get(), m.anchors, m.classes, m.d_out1.get(), m.ph, m.pw, m.d_rois.get(), used,
                            m.d_prob.get(), s);
  m.d_prob.copyToHost(m.h_prob.get(), static_cast<size_t>(total), s);
  m.stream.synchronize();

  cv::Mat mask = cv::Mat::zeros(m.height, m.width, CV_8UC1), up, inside;
  for (int k = 0; k < used; ++k) {
    const cv::Rect& box = boxes[k];
    const cv::Rect& cell = cells[k];
    const cv::Mat prob(cell.height, cell.width, CV_32F, m.h_prob.get() + m.h_rois[6 * k + 4]);
    const cv::Matx23d inverse(f, 0, (box.x + 0.5) * f - 0.5 - cell.x, 0, f, (box.y + 0.5) * f - 0.5 - cell.y);
    cv::warpAffine(prob, up, inverse, box.size(), cv::INTER_LINEAR | cv::WARP_INVERSE_MAP, cv::BORDER_REPLICATE);
    cv::compare(up, 0.5, inside, cv::CMP_GT);
    cv::Mat target = mask(box);
    cv::bitwise_or(target, inside, target);
  }
  if (!m.kernel.empty()) {
    const cv::Rect grown = cv::Rect(roi.x - m.config.dilate_px, roi.y - m.config.dilate_px,
                                    roi.width + 2 * m.config.dilate_px, roi.height + 2 * m.config.dilate_px) & image;
    cv::Mat region = mask(grown);
    cv::dilate(region, region, m.kernel);
  }
  return mask;
}

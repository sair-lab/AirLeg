#pragma once

#include <NvInfer.h>

#include <iostream>

namespace lightglue {

class Logger : public nvinfer1::ILogger {
public:
  explicit Logger(Severity severity = Severity::kWARNING) : reportable_severity_(severity) {}

  void log(Severity severity, const char* msg) noexcept override {
    if (severity > reportable_severity_) return;
    std::cerr << "TRT ";
    switch (severity) {
      case Severity::kINTERNAL_ERROR: std::cerr << "[FATAL] "; break;
      case Severity::kERROR: std::cerr << "[ERROR] "; break;
      case Severity::kWARNING: std::cerr << "[WARN] "; break;
      case Severity::kINFO: std::cerr << "[INFO] "; break;
      case Severity::kVERBOSE: std::cerr << "[VERB] "; break;
    }
    std::cerr << msg << std::endl;
  }

  void setSeverity(Severity severity) { reportable_severity_ = severity; }

private:
  Severity reportable_severity_;
};

inline Logger& getLogger() {
  static Logger logger(nvinfer1::ILogger::Severity::kWARNING);
  return logger;
}

}  // namespace lightglue

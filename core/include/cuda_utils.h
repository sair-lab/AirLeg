#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace lightglue {

#define LG_CUDA_CHECK(call)                                                    \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      throw std::runtime_error(std::string("CUDA error: ") +                  \
                               cudaGetErrorString(err) + " at " + __FILE__ +  \
                               ":" + std::to_string(__LINE__));               \
    }                                                                          \
  } while (0)

class CudaStream {
public:
  CudaStream() { LG_CUDA_CHECK(cudaStreamCreate(&stream_)); }
  ~CudaStream() { if (stream_) cudaStreamDestroy(stream_); }

  CudaStream(const CudaStream&) = delete;
  CudaStream& operator=(const CudaStream&) = delete;

  cudaStream_t get() const { return stream_; }
  void synchronize() const { LG_CUDA_CHECK(cudaStreamSynchronize(stream_)); }

private:
  cudaStream_t stream_ = nullptr;
};

template <typename T>
class CudaBuffer {
public:
  CudaBuffer() = default;
  explicit CudaBuffer(size_t count) { resize(count); }
  ~CudaBuffer() { release(); }

  CudaBuffer(const CudaBuffer&) = delete;
  CudaBuffer& operator=(const CudaBuffer&) = delete;

  void resize(size_t count) {
    if (count <= capacity_) {
      size_ = count;
      return;
    }
    release();
    capacity_ = size_ = count;
    if (capacity_ > 0) LG_CUDA_CHECK(cudaMalloc(&ptr_, capacity_ * sizeof(T)));
  }

  void copyFromHost(const T* host, size_t count, cudaStream_t stream) {
    resize(count);
    if (count > 0) LG_CUDA_CHECK(cudaMemcpyAsync(ptr_, host, count * sizeof(T), cudaMemcpyHostToDevice, stream));
  }

  void copyToHost(T* host, size_t count, cudaStream_t stream) const {
    if (count > size_) throw std::runtime_error("CudaBuffer copyToHost exceeds buffer size");
    if (count > 0) LG_CUDA_CHECK(cudaMemcpyAsync(host, ptr_, count * sizeof(T), cudaMemcpyDeviceToHost, stream));
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  size_t size() const { return size_; }

private:
  void release() {
    if (ptr_) cudaFree(ptr_);
    ptr_ = nullptr;
    size_ = 0;
    capacity_ = 0;
  }

  T* ptr_ = nullptr;
  size_t size_ = 0;
  size_t capacity_ = 0;
};

// Page-locked host memory (asynchronous copies); grows, never shrinks.
template <typename T>
class PinnedBuffer {
public:
  PinnedBuffer() = default;
  ~PinnedBuffer() { if (ptr_) cudaFreeHost(ptr_); }

  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;

  void reserve(size_t count) {
    if (count <= capacity_) return;
    if (ptr_) cudaFreeHost(ptr_);
    ptr_ = nullptr;
    capacity_ = 0;
    LG_CUDA_CHECK(cudaMallocHost(&ptr_, count * sizeof(T)));
    capacity_ = count;
  }

  T* get() { return ptr_; }
  const T* get() const { return ptr_; }
  T& operator[](size_t i) { return ptr_[i]; }
  const T& operator[](size_t i) const { return ptr_[i]; }

private:
  T* ptr_ = nullptr;
  size_t capacity_ = 0;
};

}  // namespace lightglue

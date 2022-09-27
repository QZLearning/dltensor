#pragma once
#include <cassert>
#include <cstddef>
#include <cstdio>
// #include <ostream>
#include <string>
#include <vector>

namespace wnn {

#ifdef _MSC_VER
#define WNN_ALWAYS_INLINE __forceinline
#define WNN_NEVER_INLINE __declspec(noinline)
#else
#define WNN_ALWAYS_INLINE __attribute__((always_inline)) inline
#define WNN_NEVER_INLINE __attribute__((noinline))
#endif

#define DCHECK(x, msg)                                                               \
  if (!(x))                                                                    \
  printf("Check failed: %s, %s", #x, msg) // NOLINT(*)

#define TENSOR_MAX_DIM 6

enum PixelType {
  PIXEL_NONE = 0,
  PIXEL_RGB2BGR = 1,
  PIXEL_BGR2RGB,
};

enum PermuteType {
  PERMUTE_NONE = 0,
  PERMUTE_CHW2HWC,
  PERMUTE_HWC2CHW,
};

typedef enum {
  kDLInt = 0U,
  kDLUInt = 1U,
  kDLFloat = 2U,
  kDLOpaqueHandle = 3U,
  kDLBfloat = 4U,
  kDLComplex = 5U,
} DataTypeCode;

enum DType : int {
  DT_INVALID = 0,
  DT_FLOAT = 1,
  DT_DOUBLE = 2,
  DT_INT32 = 3,
  DT_UINT8 = 4,
  DT_INT16 = 5,
  DT_INT8 = 6,
  DT_STRING = 7,
  DT_COMPLEX64 = 8,
  DT_INT64 = 9,
  DT_BOOL = 10,
  DT_QINT8 = 11,
  DT_QUINT8 = 12,
  DT_QINT32 = 13,
  DT_BFLOAT16 = 14,
  DT_QINT16 = 15,
  DT_QUINT16 = 16,
  DT_UINT16 = 17,
  DT_COMPLEX128 = 18,
  DT_FLOAT16 = 19,
  DT_TFLOAT32 = 20,
  DT_UINT32 = 21,
  DT_FLOAT64 = 22,
  DT_MAX = 32U,
};

enum Device : int {
  kX86 = 0,
  kARM = 1,
  kOPENCL = 2,
  kMETAL = 3,
  kCUDA = 4,
  kDSP = 5,
  kATLAS = 6,
  kHUAWEI_NPU = 7,
  kRK_NPU = 8,
  kAPPLE_NPU = 9,
  kCPU = 10,
};

enum BackendType : int { Backend_CPU = 0, Backend_CoreML, Backend_TensorRT };

inline std::string dtype_to_str(DType dt) {
  switch (dt) {
  case DT_FLOAT: {
    return "float32";
  }
  case DT_FLOAT16: {
    return "float16";
  }
  case DT_BOOL: {
    return "bool";
  }
  case DT_INT8: {
    return "int8";
  }
  default: {
    return "float";
  }
  }
}

///////// Datatype info
struct DataTypeInfo {
  uint8_t code;
  // Number of bits, common choices are 8, 16, 32.
  uint8_t bits;
  // Number of lanes in the type, used for vector types. */
  uint16_t lanes;
  DType dtype;
  std::string name;

  inline DataTypeInfo()
      : code(0), bits(0), lanes(0), name(""), dtype(DT_INVALID) {}

  inline DataTypeInfo(std::string name, DType dtype, uint8_t code, uint8_t bits,
                      uint16_t lanes)
      : name(name), dtype(dtype), code(code), bits(bits), lanes(lanes) {}
  inline int get_nbytes() const { return (bits + 7) / 8; }
  static const DataTypeInfo &from(DType dtype);
  static const DataTypeInfo &from(const std::string &name);
};

class RegisterDataType {
public:
  RegisterDataType(std::string name, DType dtype, uint8_t code, uint8_t bits,
                   uint16_t lanes);
  DataTypeInfo dtype_info;
};

///////// ShapeType
const int SHAPE_TYPE_MAX_NDIM = 6;
class ShapeType {
public:
  inline ShapeType() { m_size = 0; }
  inline ShapeType(size_t size) { m_size = size; }
  inline ShapeType(const int *begin, const int *end) {
    m_size = end - begin;
    m_size = m_size > SHAPE_TYPE_MAX_NDIM ? SHAPE_TYPE_MAX_NDIM : m_size;
    std::copy(begin, begin + m_size, dims);
  }
  inline ShapeType(const ShapeType &shape) {
    m_size = shape.m_size;
    std::copy(shape.begin(), shape.begin() + m_size, dims);
  }
  inline ShapeType &operator=(const ShapeType &shape) {
    m_size = shape.m_size;
    std::copy(shape.begin(), shape.begin() + m_size, dims);
    return *this;
  }
  inline bool operator==(const ShapeType &shape) {
    if (m_size != shape.m_size)
      return false;
    for (size_t i = 0; i < m_size; ++i) {
      if (dims[i] != shape.dims[i])
        return false;
    }
    return true;
  }
  inline ShapeType(const std::vector<int> &shape) {
    m_size =
        shape.size() > SHAPE_TYPE_MAX_NDIM ? SHAPE_TYPE_MAX_NDIM : shape.size();
    std::copy(shape.begin(), shape.begin() + m_size, dims);
  }
  inline ShapeType &operator=(const std::vector<int> &shape) {
    m_size =
        shape.size() > SHAPE_TYPE_MAX_NDIM ? SHAPE_TYPE_MAX_NDIM : shape.size();
    std::copy(shape.begin(), shape.begin() + m_size, dims);
    return *this;
  }
  inline ShapeType(const std::initializer_list<int> &shape) {
    m_size =
        shape.size() > SHAPE_TYPE_MAX_NDIM ? SHAPE_TYPE_MAX_NDIM : shape.size();
    std::copy(shape.begin(), shape.begin() + m_size, dims);
  }
  inline ShapeType &operator=(const std::initializer_list<int> &shape) {
    m_size =
        shape.size() > SHAPE_TYPE_MAX_NDIM ? SHAPE_TYPE_MAX_NDIM : shape.size();
    std::copy(shape.begin(), shape.begin() + m_size, dims);
    return *this;
  }

  inline size_t size() const { return m_size; }
  inline bool empty() const { return m_size == 0; }
  inline int *begin() { return dims; }
  inline int *end() { return dims + m_size; }
  inline const int *begin() const { return dims; }
  inline const int *end() const { return dims + m_size; }
  inline void push_back(int d) {
    if (m_size >= SHAPE_TYPE_MAX_NDIM)
      return;
    dims[m_size++] = d;
  }
  template <typename T>
  inline void insert(const int *it, const T &begin, const T &end) {
    size_t len = end - begin;
    size_t new_size = m_size + len;
    if (new_size > SHAPE_TYPE_MAX_NDIM)
      return;

    int tail[SHAPE_TYPE_MAX_NDIM];
    for (size_t i = 0; i < len; ++i) {
      tail[i] = *(begin + i);
    }

    size_t idx = it - dims;
    for (size_t i = idx; i < m_size; ++i) {
      tail[i - idx + len] = dims[i];
    }
    size_t tail_size = new_size - idx;
    std::copy(tail, tail + tail_size, dims + idx);
    m_size = new_size;
  }

  inline void erase(const int *it) {
    size_t idx = it - dims + 1;
    for (; idx < m_size; ++idx) {
      dims[idx - 1] = dims[idx];
    }
    m_size--;
  }

  inline void resize(size_t size) { m_size = size; }
  inline void reserve(size_t size) {}
  inline void clear() { m_size = 0; }
  inline int getdim(int i) {
    if (i < 0)
      i = size() + i;
    return dims[i];
  }
  inline int getdim(int i) const {
    if (i < 0)
      i = size() + i;
    return dims[i];
  }
  inline std::vector<int> vec() const {
    return std::vector<int>(dims, dims + m_size);
  }
  inline int &operator[](int i) {
    if (i < 0)
      i = size() + i;
    return dims[i];
  }
  inline int operator[](int i) const {
    if (i < 0)
      i = size() + i;
    return dims[i];
  }
  inline operator std::vector<int>() {
    return std::vector<int>(dims, dims + m_size);
  }

private:
  int dims[SHAPE_TYPE_MAX_NDIM];
  size_t m_size;
};

class Allocator {
public:
  virtual ~Allocator() {}

  virtual void *Alloc(size_t size) = 0;
  virtual void Free(void *ptr) = 0;
  virtual void FreeAll() = 0;
};
class Tensor {

public:
  // tensor definition in wnn, same idea as Mat in tnn or ncnn
  Tensor(std::vector<int> shape, void *data, Device dev = kCPU);
  Tensor(std::vector<int> shape, Device dev = kCPU);
  Tensor(Device dev = kCPU);
  Tensor(const Tensor &);
  Tensor &operator=(const Tensor &m);

  ~Tensor();

  template <typename T> operator T *() { return (T *)this->data; };
  template <typename T> operator const T *() const {
    return (const T *)this->data;
  };

  Tensor *astype(DType dtype);
  Tensor *to(Device dev);
  Tensor *half();

  float &index(const std::vector<size_t> &indexes) {
    return ((float *)data)[compute_index(indexes)];
  }

  float &index(std::vector<size_t> &indexes) {
    return ((float *)data)[compute_index(indexes)];
  }

protected:
  std::size_t compute_index(const std::vector<size_t> &indexes);

public:
  float operator[](int i) {
    DCHECK(i <= (int)strides[0], "i out of bounds strides");
    return ((float *)data)[i];
  }
  friend std::ostream &operator<<(std::ostream &stream, Tensor &matrix);
  friend Tensor operator+(Tensor &src, Tensor const &second);
  friend Tensor operator*(Tensor &src, Tensor const &second);
  friend Tensor operator*(Tensor &, float);
  friend Tensor &operator*=(Tensor &, float);
  friend Tensor &operator+=(Tensor &, float);

public:
  // create by outer allocator without ref count
  void Create(void *data, const ShapeType &shape, DType dtype,
              Device device = Device::kCPU);
  // create by specific allocator, use inner allocator if `allocator` is nullptr
  void Create(const ShapeType &shape, DType dtype,
              Allocator *allocator = nullptr, Device device = Device::kCPU);
  void Create(const ShapeType &shape, const std::vector<int> &padding,
              DType dtype, Allocator *allocator = nullptr,
              Device device = Device::kCPU);

  void retain();
  void release();

  void init_strides();
  int set_shape(const ShapeType &shape);
  int set_batch_shape(const ShapeType &shape);
  void set_shape_force(const ShapeType &shape);
  void set_padding(const std::vector<int> &padding);

public:
  size_t get_size() { return strides.size() > 0 ? strides[0] : 0; }
  size_t all_elm_size_;
  Tensor clone();
  void allclose(Tensor &src, double epsilon = 0.0001);
  size_t default_aligned_size_ = 16 * 8;
  void from_pixels(const unsigned char *pixels,
                   std::vector<float> pixel_means = {0.f, 0.f, 0.f},
                   std::vector<float> pixel_stds = {1.f, 1.f, 1.f},
                   PixelType pixel_type = PixelType::PIXEL_BGR2RGB,
                   PermuteType permute_type = PermuteType::PERMUTE_HWC2CHW);
  void to_pixels(unsigned char *pixels) const;
  void from_numpy_bin(std::string bin_file);

public:
  void *data = nullptr;
  // data is pre-allocated if ref count is nullptr
  int *ref_count = nullptr;
  Allocator *allocator = nullptr;

  Device device = Device::kCPU;
  DType dtype = DType::DT_FLOAT;

  ShapeType shape;
  ShapeType strides;

public:
  size_t get_size() const { return strides.size() > 0 ? strides[0] : 0; }
  size_t get_nbytes() const {
    return get_size() * DataTypeInfo::from(dtype).get_nbytes();
  }
  size_t get_ndim() const { return shape.size(); }

public:
  static size_t get_alloc_elems(const ShapeType &shape);
  static size_t get_alloc_elems(const ShapeType &shape,
                                const std::vector<int> &padding);
  static size_t get_alloc_bytes(const ShapeType &shape, DType dtype);
  static size_t get_alloc_bytes(const ShapeType &shape,
                                const std::vector<int> &padding, DType dtype);
};

WNN_ALWAYS_INLINE size_t Tensor::get_alloc_elems(const ShapeType &_shape) {
  size_t elem_size = 1;
  for (size_t i = 0; i < _shape.size(); ++i) {
    elem_size *= _shape[i];
  }
  return elem_size;
}
WNN_ALWAYS_INLINE size_t Tensor::get_alloc_elems(
    const ShapeType &_shape, const std::vector<int> &_padding) {
  size_t elem_size = 1;
  size_t idx = _shape.size();
  while (idx--) {
    elem_size *= _shape[idx];
    elem_size += _padding[idx];
  }
  return elem_size;
}
WNN_ALWAYS_INLINE size_t Tensor::get_alloc_bytes(const ShapeType &_shape,
                                                 DType _dtype) {
  return get_alloc_elems(_shape) * DataTypeInfo::from(_dtype).get_nbytes();
}
WNN_ALWAYS_INLINE size_t Tensor::get_alloc_bytes(
    const ShapeType &_shape, const std::vector<int> &_padding, DType _dtype) {
  return get_alloc_elems(_shape, _padding) *
         DataTypeInfo::from(_dtype).get_nbytes();
}
// Tensor ends

Tensor randn(std::vector<int> dims, size_t alinged_size = 16);
Tensor randn_seed(std::vector<int> dims, size_t alinged_size = 16,
                  unsigned seed = 1024);
Tensor ones(std::vector<int> dims, size_t alinged_size = 16);
Tensor zeros(std::vector<int> dims, size_t alinged_size = 16);
Tensor from_numpy_bin(std::string bin_file, std::vector<int> dims,
                      size_t alinged_size = 16);
bool to_numpy_bin(Tensor &src, std::string save_file);

void allclose(Tensor &src, Tensor &tgt, double epsilon = 0.0001);
void check_nan(Tensor &src);
void argmax(Tensor &src, float &value, size_t &index);
void argmax(Tensor &src, std::vector<float> &values,
            std::vector<size_t> &indexes);

// Some helper functions
inline void PrintShape(ShapeType st) {
  printf("Shape: [");
  for (size_t i = 0; i < st.size(); i++) {
    printf("%d, ", st[i]);
  }
  printf("]\n");
}
void print_tensor_flatten_values(Tensor *t);
inline std::string GetShapeTypeStr(ShapeType st) {
  std::string s = "[";
  for (size_t i = 0; i < st.size(); i++) {
    s += std::to_string(st[i]) + ",";
  }
  s += "]";
  return s;
}
} // namespace wnn
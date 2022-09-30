#include "dltensor/tensor.h"
#include "dltensor/builtin_fp16.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
// #include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#ifdef __aarch64__
#include <arm_neon.h>
#endif

#ifdef USE_TENSORRT
#include <cuda_runtime_api.h>
#endif

namespace wnn {

///// DataTypeInfo
std::vector<DataTypeInfo *> &GetDtype2Info() {
  static std::vector<DataTypeInfo *> dtype2info((int)DT_MAX);
  return dtype2info;
}

std::map<std::string, DataTypeInfo *> &GetName2DtypeInfo() {
  static std::map<std::string, DataTypeInfo *> str2dtype_info;
  return str2dtype_info;
}

RegisterDataType::RegisterDataType(std::string name, DType dtype, uint8_t code,
                                   uint8_t bits, uint16_t lanes)
    : dtype_info(name, dtype, code, bits, lanes) {
  GetDtype2Info()[(int)dtype] = &dtype_info;
  GetName2DtypeInfo()[name] = &dtype_info;
}

static RegisterDataType DLDTYPE_UNKNOWN("unknown", DT_INVALID, 0, 0, 0);
static RegisterDataType DLDTYPE_FLOAT64("float64", DT_FLOAT64, kDLFloat, 64, 1);
static RegisterDataType DLDTYPE_FLOAT32("float32", DT_FLOAT, kDLFloat, 32, 1);
static RegisterDataType DLDTYPE_FLOAT16("float16", DT_FLOAT16, kDLFloat, 16, 1);
static RegisterDataType DLDTYPE_BFLOAT16("bfloat16", DT_BFLOAT16, kDLBfloat, 16,
                                         1);
static RegisterDataType DLDTYPE_INT8("int8", DT_INT8, kDLInt, 8, 1);
static RegisterDataType DLDTYPE_UINT8("uint8", DT_UINT8, kDLUInt, 8, 1);
static RegisterDataType DLDTYPE_INT16("int16", DT_INT16, kDLInt, 16, 1);
static RegisterDataType DLDTYPE_UINT16("uint16", DT_UINT16, kDLUInt, 16, 1);
static RegisterDataType DLDTYPE_INT32("int32", DT_INT32, kDLInt, 32, 1);
static RegisterDataType DLDTYPE_UINT32("uint32", DT_UINT32, kDLUInt, 32, 1);

const DataTypeInfo &DataTypeInfo::from(DType dtype) {
  return *(GetDtype2Info()[(int)dtype]);
}

const DataTypeInfo &DataTypeInfo::from(const std::string &name) {
  return *(GetName2DtypeInfo()[name]);
}

Tensor::Tensor(const Tensor &t) {
  this->data = t.data;
  this->ref_count = t.ref_count;
  this->allocator = t.allocator;
  this->dtype = t.dtype;
  this->device = t.device;
  this->shape = t.shape;
  this->strides = t.strides;
  this->retain();
}

Tensor &Tensor::operator=(const Tensor &t) {
  if (this == &t)
    return *this;
  release();
  this->data = t.data;
  this->ref_count = t.ref_count;
  this->allocator = t.allocator;
  this->dtype = t.dtype;
  this->device = t.device;
  this->shape = t.shape;
  this->strides = t.strides;
  this->retain();
  return *this;
}

Tensor::Tensor(std::vector<int> shape, void *data, Device dev)
    : data(data), device(dev) {
  shape = ShapeType(shape);
  all_elm_size_ = std::accumulate(std::begin(shape), std::end(shape), 1,
                                  std::multiplies<size_t>());
}

Tensor::Tensor(std::vector<int> shape, Device dev) : device(dev) {
  all_elm_size_ = std::accumulate(std::begin(shape), std::end(shape), 1,
                                  std::multiplies<size_t>());
  data = (float *)malloc(sizeof(float) * all_elm_size_);
  shape = ShapeType(shape);
  // data_ = (float *)mm_malloc(sizeof(float) * all_elm_size_, 8 *
  // default_aligned_size_);
}

Tensor::Tensor(Device dev) : device(dev) {}

Tensor::~Tensor() {
  // data_ = nullptr;
  release();
}

void Tensor::Create(void *_data, const ShapeType &_shape, DType _dtype,
                    Device _device) {
  release();

  this->data = _data;
  this->ref_count = nullptr;
  this->allocator = nullptr;

  this->dtype = _dtype;
  this->device = _device;
  this->shape = _shape;
  init_strides();
}

void Tensor::Create(const ShapeType &_shape, DType _dtype,
                    Allocator *_allocator, Device _device) {
  release();
  size_t total_bytes = get_alloc_bytes(_shape, _dtype);
  if (_allocator) {
    this->data = _allocator->Alloc(total_bytes + sizeof(int));
    this->ref_count = (int *)((int8_t *)this->data + total_bytes);
  } else {
    this->data = mm_malloc(total_bytes + sizeof(int), 64);
    this->ref_count = (int *)((int8_t *)this->data + total_bytes);
  }
  *(this->ref_count) = 1;
  this->allocator = _allocator;

  this->dtype = _dtype;
  this->device = _device;
  this->shape = _shape;
  init_strides();
}

void Tensor::Create(const ShapeType &_shape, const std::vector<int> &_padding,
                    DType _dtype, Allocator *_allocator, Device _device) {
  release();

  size_t total_bytes = get_alloc_bytes(_shape, _padding, _dtype);
  if (_allocator) {
    this->data = _allocator->Alloc(total_bytes + sizeof(int));
    this->ref_count = (int *)((int8_t *)this->data + total_bytes);
  } else {
    this->data = mm_malloc(total_bytes + sizeof(int), 64);
    this->ref_count = (int *)((int8_t *)this->data + total_bytes);
  }
  *(this->ref_count) = 1;
  this->allocator = _allocator;

  this->dtype = _dtype;
  this->device = _device;
  this->shape = _shape;
  set_padding(_padding);
}

void Tensor::release() {
  if (this->data == nullptr)
    return;

  if (ref_count != nullptr) {
    (*ref_count)--;
    // printf("Release: %p %p %d\n", this, data, *ref_count);
    if (*ref_count == 0) {
      // delete data and ref_count
      if (allocator) {
        allocator->Free(data);
      } else {
        mm_free(data);
      }
    }
  }

  this->ref_count = nullptr;
  this->data = nullptr;
  this->allocator = nullptr;
  this->dtype = DType::DT_INVALID;
  this->device = Device::kCPU;
  this->shape.clear();
  this->strides.clear();
}

void Tensor::set_shape_force(const ShapeType &_shape) {
  this->shape = _shape;
  this->init_strides();
}

int Tensor::set_shape(const ShapeType &_shape) {
  int neg_idx = -1;
  size_t elem_size = 1;
  for (size_t i = 0; i < _shape.size(); ++i) {
    if (_shape[i] < 0) {
      if (neg_idx >= 0)
        return -1;
      neg_idx = (int)i;
    } else {
      elem_size *= _shape[i];
    }
  }

  if (get_size() % elem_size)
    return -2;

  this->shape = _shape;
  if (neg_idx >= 0) {
    this->shape[neg_idx] = (int)(get_size() / elem_size);
  }

  init_strides();
  return 0;
}

int Tensor::set_batch_shape(const ShapeType &_shape) {
  int elem_size = 1;
  int total_size = (int)get_size();

  strides.resize(_shape.size());
  for (size_t i = _shape.size() - 1; i > 0; --i) {
    elem_size *= _shape[i];
    strides[i] = elem_size;
#ifdef TCNN_DEBUG
    if (_shape[i] <= 0) {
      fprintf(stderr, "SetBatchShape Error: %s to %s\n",
              to_string(this->shape).c_str(), to_string(_shape).c_str());
    }
#endif
  }

  int batch = total_size / elem_size;
  if (batch * elem_size != total_size) {
    init_strides();
    return -2;
  }

  this->shape = _shape;
  this->shape[0] = batch;
  return 0;
}

void Tensor::init_strides() {
  strides.resize(shape.size() + 1);
  int ndim = (int)shape.size();
  int total_stride = 1;
  strides[ndim] = 1;
  while (ndim--) {
    total_stride *= shape[ndim];
    strides[ndim] = total_stride;
  }
}

void Tensor::set_padding(const std::vector<int> &_padding) {
  if (_padding.size() != shape.size()) {
    strides.clear();
    return;
  }
  strides.resize(shape.size());
  int ndim = (int)shape.size();
  int total_stride = 1;
  while (ndim--) {
    total_stride *= shape[ndim];
    total_stride += _padding[ndim];
    strides[ndim] = total_stride;
  }
}

void Tensor::retain() {
  if (ref_count != nullptr) {
    (*ref_count)++;
    // printf("Retain: %p %p %d\n", this, data, *ref_count);
  }
}

std::ostream &operator<<(std::ostream &stream, Tensor &matrix) {
  bool delimiter_loged = false;
  if (matrix.shape.size() == 1) {
    stream << "[";
    auto delimiter_loged0 = false;
    for (int j = 0; j < matrix.shape[0]; ++j) {
      if (j < 6 || j > matrix.shape[0] - 8) {
        stream << std::setw(7) << std::fixed << std::setprecision(4)
               << matrix[j];
        if (j != matrix.shape[0] - 1) {
          stream << ", ";
        }
      } else {
        if (!delimiter_loged0) {
          stream << "..., ";
          delimiter_loged0 = true;
        }
      }
    }
    stream << "]";
  } else if (matrix.shape.size() == 2) {
    stream << "[";
    bool delimiter_loged1 = false;
    for (int i = 0; i < matrix.shape[0]; ++i) {
      if (i < 3 || i > matrix.shape[0] - 4) {
        if (i == 0) {
          stream << "[";
        } else {
          stream << " [";
        }
        auto delimiter_loged0 = false;
        for (int j = 0; j < matrix.shape[1]; ++j) {
          if (j < 4 || j > matrix.shape[1] - 5) {
            stream << std::setw(7) << std::fixed << std::setprecision(4)
                   << matrix[i * matrix.shape[1] + j];
            if (j != matrix.shape[1] - 1) {
              stream << ", ";
            }
          } else {
            if (!delimiter_loged0) {
              stream << "..., ";
              delimiter_loged0 = true;
            }
          }
        }
        if (i == matrix.shape[0] - 1) {
          stream << "]";
        } else {
          stream << "],\n";
        }
      } else {
        if (!delimiter_loged1) {
          stream << "  ...\n";
          delimiter_loged1 = true;
        }
      }
    }
    stream << "]";
  } else if (matrix.shape.size() == 3) {
    stream << "[";
    bool delimiter_loged0 = false;
    for (int i = 0; i < matrix.shape[0]; ++i) {
      if (i < 3 || i > matrix.shape[0] - 4) {
        stream << "[";
        bool delimiter_loged1 = false;
        for (int j = 0; j < matrix.shape[1]; ++j) {
          if (j < 4 || j > matrix.shape[1] - 5) {
            if (j == 0) {
              stream << "[";
            } else {
              stream << "  [";
            }
            bool delimiter_loged0 = false;
            for (int k = 0; k < matrix.shape[2]; ++k) {
              if (k < 5 || k > matrix.shape[2] - 5) {
                stream << std::setw(7) << std::fixed << std::setprecision(4)
                       << matrix[i * matrix.shape[1] * matrix.shape[2] +
                                 j * matrix.shape[2] + k];
                if (k != matrix.shape[2] - 1) {
                  stream << ", ";
                }
              } else {
                if (!delimiter_loged0) {
                  stream << "..., ";
                  delimiter_loged0 = true;
                }
              }
            }
            // if i change, don't add \n
            if (j == matrix.shape[1] - 1) {
              if (i == matrix.shape[0] - 1) {
                stream << "]]";
              } else {
                stream << "]],\n";
              }
            } else {
              stream << "],\n";
            }
          } else {
            if (!delimiter_loged1) {
              stream << "  ...\n";
              delimiter_loged1 = true;
            }
          }
        }
        if (i == matrix.shape[1] - 1) {
          stream << "]";
        } else {
          if (i != matrix.shape[0] - 1) {
            stream << " ";
          }
        }
      } else {
        if (!delimiter_loged0) {
          stream << "  ...\n";
          delimiter_loged0 = true;
        }
      }
    }
    stream << "]";
  } else if (matrix.shape.size() == 4) {
    // 1,2,56,34
    stream << "[";
    bool delimiter_loged0 = false;
    for (int i = 0; i < matrix.shape[1]; ++i) {
      if (i < 3 || i > matrix.shape[1] - 4) {
        if (i == 0) {
          stream << "[";
        } else {
          stream << " [";
        }

        bool delimiter_loged1 = false;
        for (int j = 0; j < matrix.shape[2]; ++j) {

          if (j < 3 || j > matrix.shape[2] - 4) {
            if (j == 0) {
              stream << "[";
            } else {
              stream << "  [";
            }
            bool delimiter_loged0 = false;

            for (int k = 0; k < matrix.shape[3]; ++k) {
              if (k < 4 || k > matrix.shape[3] - 5) {
                stream << std::setw(7) << std::fixed << std::setprecision(4)
                       << matrix[i * matrix.shape[2] * matrix.shape[3] +
                                 j * matrix.shape[3] + k];
                if (k != matrix.shape[3] - 1) {
                  stream << ", ";
                }
              } else {
                if (!delimiter_loged0) {
                  stream << "..., ";
                  delimiter_loged0 = true;
                }
              }
            }
            // if i change, don't add \n
            if (j == matrix.shape[2] - 1) {
              stream << "]";
            } else {
              stream << "],\n";
            }
          } else {
            if (!delimiter_loged1) {
              stream << "  ...\n";
              delimiter_loged1 = true;
            }
          }
        }

        if (i == matrix.shape[1] - 1) {
          stream << "]";
        } else {
          stream << "],\n\n";
        }
      } else {
        if (!delimiter_loged0) {
          stream << "  ...\n";
          delimiter_loged0 = true;
        }
      }
    }
    stream << "]";
  } else {
    for (size_t i = 0; i < matrix.get_size(); ++i) {
      if (i < 40 || i > matrix.get_size() - 10) {
        stream << std::setw(7) << std::fixed << std::setprecision(4)
               << matrix[i] << ", ";
        if ((i + 1) % 8 == 0) {
          stream << "\n";
        }
      } else if (!delimiter_loged) {
        stream << "..........., ";
        delimiter_loged = true;
      }
    }
  }
  stream << "\ntotal size: " << matrix.get_size() << ", shape: [";
  for (auto i : matrix.shape) {
    stream << i << ",";
  }
  stream << "]\n";
  return stream;
}

Tensor operator+(Tensor &src, Tensor const &second) {
  DCHECK(src.shape == second.shape,
        "tensor shapes are not same can not perform calculation!");
  return src;
}

Tensor operator*(Tensor &src, Tensor const &second) {
  DCHECK(src.shape == second.shape,
        "tensor shapes are not same can not perform calculation!");
  return src;
}

Tensor operator*(Tensor &src, float a) {
  float *ptr = (float *)src.data;
  auto src_clone = src.clone();
  float *ptr_dst = (float *)src_clone.data;
  for (size_t i = 0; i < src.get_size(); ++i) {
    ptr_dst[i] = ptr[i] * a;
  }
  return src_clone;
}

Tensor &operator*=(Tensor &src, float a) {
  float *ptr = (float *)src.data;
  for (size_t i = 0; i < src.get_size(); ++i) {
    ptr[i] = ptr[i] * a;
  }
  return src;
}

Tensor &operator+=(Tensor &src, float a) {
  float *ptr = (float *)src.data;
  for (size_t i = 0; i < src.get_size(); ++i) {
    ptr[i] = ptr[i] + a;
  }
  return src;
}

std::size_t Tensor::compute_index(const std::vector<size_t> &indexes) {
  size_t index = 0;
  size_t mul = 1;
  for (int i = shape.size(); i != 0; --i) {
    DCHECK((int)indexes[i - 1] < shape[i - 1], "dims shapes not right!");
    index += indexes[i - 1] * mul;
    mul *= shape[i - 1];
  }
  DCHECK(index < get_size(), "index calculated out of bounds");
  return index;
}

void Tensor::from_numpy_bin(std::string bin_file) {
  FILE *f = fopen(bin_file.c_str(), "rb");
  fseek(f, 0, SEEK_SET);

  if (f == NULL) {
    fprintf(stderr, "Couldn't load file\n");
  }
  for (size_t i = 0; i < get_size(); ++i) {
    float item = 0.0;
    fread(&item, sizeof(float), 1, f);
    ((float *)data)[i] = item;
  }
  fclose(f);
}

void Tensor::from_pixels(const unsigned char *pixels,
                         std::vector<float> pixel_means,
                         std::vector<float> pixel_stds, PixelType pixel_type,
                         PermuteType permute_type) {

  size_t batch, channel, h, w;
  if (shape.size() == 3) {
    batch = 1;
    channel = shape[0];
    h = shape[1];
    w = shape[2];
  } else if (shape.size() == 4) {
    batch = shape[0];
    channel = shape[1];
    h = shape[2];
    w = shape[3];
  } else {
   printf("[err] dims not correct, must be 3 or 4!\n");
  }

  DCHECK(batch == 1, "currently from_pixels only support one batch!");
  DCHECK(permute_type == PermuteType::PERMUTE_HWC2CHW,
        "currently from_pixels only support one PERMUTE_HWC2CHW!");

  auto stride = w * channel;
  const int wgap = stride - w * 3;
  if (wgap == 0) {
    w = w * h;
    h = 1;
  }

  // float *ptr_start = (float *)dst.get_data();

  float *ptr0 = (float *)data;
  float *ptr1 = ptr0 + h * w;
  float *ptr2 = ptr0 + 2 * h * w;

  for (size_t y = 0; y < h; y++) {
#if __ARM_NEON
    int nn = w >> 3;
    int remain = w - (nn << 3);
#else
    int remain = w;
#endif // __ARM_NEON

#if __ARM_NEON
#if __aarch64__
    for (; nn > 0; nn--) {
      uint8x8x3_t _rgb = vld3_u8(pixels);
      uint16x8_t _r16 = vmovl_u8(_rgb.val[0]);
      uint16x8_t _g16 = vmovl_u8(_rgb.val[1]);
      uint16x8_t _b16 = vmovl_u8(_rgb.val[2]);

      float32x4_t _rlow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_r16)));
      float32x4_t _rhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_r16)));
      float32x4_t _glow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_g16)));
      float32x4_t _ghigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_g16)));
      float32x4_t _blow = vcvtq_f32_u32(vmovl_u16(vget_low_u16(_b16)));
      float32x4_t _bhigh = vcvtq_f32_u32(vmovl_u16(vget_high_u16(_b16)));

      // normalization, but we always treat means stds in RGB order
      float32x4_t mean_vecr = vdupq_n_f32(pixel_means[0]);
      float32x4_t std_vecr = vdupq_n_f32(pixel_stds[0]);
      float32x4_t mean_vecg = vdupq_n_f32(pixel_means[1]);
      float32x4_t std_vecg = vdupq_n_f32(pixel_stds[1]);
      float32x4_t mean_vecb = vdupq_n_f32(pixel_means[2]);
      float32x4_t std_vecb = vdupq_n_f32(pixel_stds[2]);

      // printf("%f %f %f %f \n", mean_vecr[0], mean_vecr[1], mean_vecr[2],
      // mean_vecr[3]); printf("%f %f %f %f \n", std_vecr[0], std_vecr[1],
      // std_vecr[2], std_vecr[3]);

      if (pixel_type == PIXEL_BGR2RGB) {
        // now the input actually BGR
        _rlow = vsubq_f32(_rlow, mean_vecb);
        _rlow = vdivq_f32(_rlow, std_vecb);
        _rhigh = vsubq_f32(_rhigh, mean_vecb);
        _rhigh = vdivq_f32(_rhigh, std_vecb);

        _glow = vsubq_f32(_glow, mean_vecg);
        _glow = vdivq_f32(_glow, std_vecg);
        _ghigh = vsubq_f32(_ghigh, mean_vecg);
        _ghigh = vdivq_f32(_ghigh, std_vecg);

        _blow = vsubq_f32(_blow, mean_vecr);
        _blow = vdivq_f32(_blow, std_vecr);
        _bhigh = vsubq_f32(_bhigh, mean_vecr);
        _bhigh = vdivq_f32(_bhigh, std_vecr);

        vst1q_f32(ptr2, _rlow);
        vst1q_f32(ptr2 + 4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1 + 4, _ghigh);
        vst1q_f32(ptr0, _blow);
        vst1q_f32(ptr0 + 4, _bhigh);
      } else {
        // don't swap channel
        _rlow = vsubq_f32(_rlow, mean_vecr);
        _rlow = vdivq_f32(_rlow, std_vecr);
        _rhigh = vsubq_f32(_rhigh, mean_vecr);
        _rhigh = vdivq_f32(_rhigh, std_vecr);

        _glow = vsubq_f32(_glow, mean_vecg);
        _glow = vdivq_f32(_glow, std_vecg);
        _ghigh = vsubq_f32(_ghigh, mean_vecg);
        _ghigh = vdivq_f32(_ghigh, std_vecg);

        _blow = vsubq_f32(_blow, mean_vecb);
        _blow = vdivq_f32(_blow, std_vecb);
        _bhigh = vsubq_f32(_bhigh, mean_vecb);
        _bhigh = vdivq_f32(_bhigh, std_vecb);

        vst1q_f32(ptr0, _rlow);
        vst1q_f32(ptr0 + 4, _rhigh);
        vst1q_f32(ptr1, _glow);
        vst1q_f32(ptr1 + 4, _ghigh);
        vst1q_f32(ptr2, _blow);
        vst1q_f32(ptr2 + 4, _bhigh);
      }

      pixels += 3 * 8;
      ptr0 += 8;
      ptr1 += 8;
      ptr2 += 8;
    }
#else
    if (nn > 0) {
      asm volatile("0:                             \n"
                   "pld        [%1, #256]          \n"
                   "vld3.u8    {d0-d2}, [%1]!      \n"
                   "vmovl.u8   q8, d0              \n"
                   "vmovl.u8   q9, d1              \n"
                   "vmovl.u8   q10, d2             \n"
                   "vmovl.u16  q0, d16             \n"
                   "vmovl.u16  q1, d17             \n"
                   "vmovl.u16  q2, d18             \n"
                   "vmovl.u16  q3, d19             \n"
                   "vmovl.u16  q8, d20             \n"
                   "vmovl.u16  q9, d21             \n"
                   "vcvt.f32.u32   q0, q0          \n"
                   "vcvt.f32.u32   q1, q1          \n"
                   "vcvt.f32.u32   q2, q2          \n"
                   "vcvt.f32.u32   q3, q3          \n"
                   "vcvt.f32.u32   q8, q8          \n"
                   "subs       %0, #1              \n"
                   "vst1.f32   {d0-d3}, [%4]!      \n"
                   "vcvt.f32.u32   q9, q9          \n"
                   "vst1.f32   {d4-d7}, [%3]!      \n"
                   "vst1.f32   {d16-d19}, [%2]!    \n"
                   "bne        0b                  \n"
                   : "=r"(nn),   // %0
                     "=r"(rgb),  // %1
                     "=r"(ptr0), // %2
                     "=r"(ptr1), // %3
                     "=r"(ptr2)  // %4
                   : "0"(nn), "1"(rgb), "2"(ptr0), "3"(ptr1), "4"(ptr2)
                   : "cc", "memory", "q0", "q1", "q2", "q3", "q8", "q9", "q10");
    }
#endif // __aarch64__
#endif // __ARM_NEON
    for (; remain > 0; remain--) {
      if (pixel_type == PIXEL_BGR2RGB) {
        *ptr0 = (pixels[2] - pixel_means[0]) / pixel_stds[0];
        *ptr1 = (pixels[1] - pixel_means[1]) / pixel_stds[1];
        *ptr2 = (pixels[0] - pixel_means[2]) / pixel_stds[2];
      } else {
        *ptr0 = (pixels[0] - pixel_means[0]) / pixel_stds[0];
        *ptr1 = (pixels[1] - pixel_means[1]) / pixel_stds[1];
        *ptr2 = (pixels[2] - pixel_means[2]) / pixel_stds[2];
      }

      pixels += 3;
      ptr0++;
      ptr1++;
      ptr2++;
    }

    pixels += wgap;
  }
}

void Tensor::to_pixels(unsigned char *pixels) const {}

Tensor Tensor::clone() {
  auto n_t = Tensor(shape, kCPU);
  memcpy(n_t.data, data, all_elm_size_ * sizeof(float));
  return n_t;
}

void Tensor::allclose(Tensor &src, double epsilon) {
  if (!(shape == src.shape)) {
    printf("dims must same for tensor compare close!\n");
  } else {
    float diff_sum = 0.f;
    float max_diff = 0.f;
    int max_at = 0;
    for (size_t i = 0; i < src.get_size(); ++i) {
      float diff_i = src[i] - ((float *)data)[i];
      diff_sum += diff_i;
      auto max_diff_new = std::max(max_diff, diff_i);
      if (max_diff_new != max_diff) {
        // find a new max one
        max_at = i;
      }
      max_diff = max_diff_new;
    }
    diff_sum = abs(diff_sum) / src.get_size();
    if (diff_sum > epsilon) {
      printf("allclose check failed! diff: %f, max_diff: %f at index: %d \n",
             diff_sum, max_diff, max_at);
    } else {
      printf("tensors are equal at diff: %f \n", diff_sum);
    }
  }
}

Tensor *Tensor::astype(DType t_) {
  // convert dtype here
  if ((dtype == DType::DT_FLOAT) && (t_ == DType::DT_FLOAT16)) {
    // cast data to fp16
    void *data_tmp =
        mm_malloc(get_size() * sizeof(uint16_t), default_aligned_size_);
    std::vector<uint16_t> quantizedFp16Weight;
    quantizedFp16Weight.resize(get_size());
    std::transform((float *)data, (float *)data + get_size(),
                   quantizedFp16Weight.begin(), [](float w) {
                     w = fmaxf(w, -65504.0f);
                     w = fminf(w, 65504.0f);
                     return f2h_ieee(w);
                   });
    memcpy(data_tmp, quantizedFp16Weight.data(), sizeof(uint16_t) * get_size());
    data = data_tmp;
    dtype = DType::DT_FLOAT16;
  }
  return this;
}

Tensor *Tensor::half() {
  // convert dtype here
  if (dtype == DType::DT_FLOAT) {
    this->astype(DType::DT_FLOAT16);
  }
  return this;
}

Tensor *Tensor::to(Device dev) {
// copy tensor data to cuda
#ifdef USE_TENSORRT
  if (dev_ == kCPU && dev == kCUDA) {
    // upload cpu data to cuda
  } else if (dev_ == kCUDA && dev == kCPU) {
    // download gpu data to cpu
  }
#endif
  return this;
}

/////// Helper functions
Tensor randn(std::vector<int> dims, size_t alinged_size) {
  srand((unsigned)time(0));
  float *data;
  size_t all_elm_size = std::accumulate(std::begin(dims), std::end(dims), 1,
                                        std::multiplies<size_t>());

  // all using aligned mem to support AVX 256
  data = (float *)mm_malloc(sizeof(float) * all_elm_size, 8 * alinged_size);
  //   data = (float *)malloc(sizeof(float) * all_elm_size);
  for (size_t i = 0; i < all_elm_size; ++i) {
    auto f = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    data[i] = f;
  }
  // auto t = Tensor(dims, data);
  Tensor t;
  ShapeType sh(dims);
  t.Create(sh, DT_FLOAT);
  t.data = data;
  return t;
}

Tensor randn_seed(std::vector<int> dims, size_t alinged_size, unsigned seed) {
  srand(seed);
  float *data;
  size_t all_elm_size = std::accumulate(std::begin(dims), std::end(dims), 1,
                                        std::multiplies<size_t>());

  // all using aligned mem to support AVX 256
  data = (float *)mm_malloc(sizeof(float) * all_elm_size, 8 * alinged_size);
  //   data = (float *)malloc(sizeof(float) * all_elm_size);
  for (size_t i = 0; i < all_elm_size; ++i) {
    auto f = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    data[i] = f;
  }
  auto t = Tensor(dims, data);
  return t;
}

Tensor ones(std::vector<int> dims, size_t alinged_size) {
  float *data;
  size_t all_elm_size = std::accumulate(std::begin(dims), std::end(dims), 1,
                                        std::multiplies<size_t>());
  //   data = (float *)malloc(sizeof(float) * all_elm_size);
  data = (float *)mm_malloc(sizeof(float) * all_elm_size, 8 * alinged_size);
  for (size_t i = 0; i < all_elm_size; ++i) {
    data[i] = 1.f;
  }
  Tensor t;
  ShapeType sh(dims);
  t.Create(sh, DT_FLOAT);
  t.data = data;
  return t;
}

Tensor zeros(std::vector<int> dims, size_t alinged_size) {
  float *data;
  size_t all_elm_size = std::accumulate(std::begin(dims), std::end(dims), 1,
                                        std::multiplies<size_t>());
  data = (float *)std::calloc(all_elm_size, sizeof(float));
  Tensor t;
  ShapeType sh(dims);
  t.Create(sh, DT_FLOAT);
  t.data = data;
  return t;
}

Tensor from_numpy_bin(std::string bin_file, std::vector<int> dims,
                      size_t alinged_size) {
  FILE *f = fopen(bin_file.c_str(), "rb");
  fseek(f, 0, SEEK_SET);
  if (f == NULL) {
    fprintf(stderr, "Couldn't load file\n");
  }
  size_t all_elm_size = std::accumulate(std::begin(dims), std::end(dims), 1,
                                        std::multiplies<size_t>());
  float *data;
  data = (float *)mm_malloc(sizeof(float) * all_elm_size, 8 * alinged_size);

  for (size_t i = 0; i < all_elm_size; ++i) {
    float item = 0.0;
    fread(&item, sizeof(float), 1, f);
    data[i] = item;
  }
  fclose(f);
  Tensor t;
  ShapeType sh(dims);
  t.Create(sh, DT_FLOAT);
  t.data = data;
  return t;
}

bool to_numpy_bin(Tensor &src, std::string save_file) {
  std::ofstream out(save_file, std::ios_base::binary);
  if (out.good()) {
    for (size_t i = 0; i < src.get_size(); ++i) {
      float a = src[i];
      out.write((char *)&a, sizeof(float));
    }
    out.close();
    return true;
  }
  return false;
}

void allclose(Tensor &src, Tensor &tgt, double epsilon) {
  if (!(tgt.shape == src.shape)) {
    printf("dims must same for tensor compare close!");
  } else {
    float diff_sum = 0.f;
    float max_diff = 0.f;
    int max_at = 0;
    for (size_t i = 0; i < src.get_size(); ++i) {
      float diff_i = src[i] - tgt[i];
      diff_sum += diff_i;
      auto max_diff_new = std::max(max_diff, diff_i);
      if (max_diff_new != max_diff) {
        // find a new max one
        max_at = i;
      }
      max_diff = max_diff_new;
    }
    diff_sum = abs(diff_sum) / src.get_size();
    if (diff_sum > epsilon) {
      printf("allclose check failed! diff: %f, max_diff: %f at index: %d \n",
             diff_sum, max_diff, max_at);
    } else {
      printf("tensors are equal at diff: %f \n", diff_sum);
    }
  }
}

void check_nan(Tensor &src) {
  int k = 0;
  for (size_t i = 0; i < src.get_size(); ++i) {
    float a = src[i];
    if (std::isnan(a)) {
      printf("find Nan at: %lu, ", i);
      k++;
    }
  }
  if (k > 0) {
    printf("all find %d Nan values out of: %lu \n", k, src.get_size());
  }
}

void argmax(Tensor &src, float &value, size_t &index) {
  DCHECK(src.shape.size() == 1, "argmax only support tensor dim == 1.");
}
void argmax(Tensor &src, std::vector<float> &values,
            std::vector<size_t> &indexes) {
  DCHECK(src.shape.size() == 2, "argmax only support tensor dim == 2.");
  float *ptr = (float *)src.data;
  for (int i = 0; i < src.shape[0]; ++i) {
    auto start = ptr + i * src.shape[1];
    std::vector<float> my_vector{start, start + src.shape[1]};
    auto value = std::max_element(my_vector.begin(), my_vector.end());
    size_t max_index_i = std::max_element(my_vector.begin(), my_vector.end()) -
                         my_vector.begin();
    values.push_back(*value);
    indexes.push_back(i * src.shape[1] + max_index_i);

    // std::sort(start, start + src.shape[1],
    //           [](std::pair<int, float> a, std::pair<int, float> b) {
    //             return a.second > b.second;
    //           });
  }
}

void print_tensor_flatten_values(Tensor *t) {
  PrintShape(t->shape);
  if (!t->shape.empty() && t->data != nullptr) {
    int b = t->shape[0];
    // printf("all batch: %d\n", b);
    for (int i = 0; i < b; ++i) {
      printf("batch%d: ", i);
      for (int j = 0; j < t->strides[1]; ++j) {
        if (j < 5 || j > t->strides[1] - 4) {
          if (j == t->strides[1] - 1) {
            printf("%f", (*t)[i * t->strides[1] + j]);
          } else {
            printf("%f, ", (*t)[i * t->strides[1] + j]);
          }
        } else if (j == 6) {
          printf("...., ");
        }
      }
      printf("\n");
    }
  }
}

} // namespace wnn
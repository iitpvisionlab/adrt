#pragma once
#include <stdlib.h>  // abort

#include <cstdint>
#include <cstring>  // std::memcpy

#ifdef __GNUC__
#define A_LIKELY(cond) (__builtin_expect(!!(cond), 1))
#define A_UNLIKELY(cond) (__builtin_expect(!!(cond), 0))
#else
#define A_LIKELY(cond) (cond)
#define A_UNLIKELY(cond) (cond)
#endif

#if !defined(NDEBUG)
#define A_NEVER(cond) \
  if (cond) {         \
    abort();          \
  }
#else
#if defined(__GNUC__)
#define A_NEVER(cond)        \
  if (cond) {                \
    __builtin_unreachable(); \
  }
#elif defined(_MSC_VER)
#define A_NEVER(cond) \
  if (cond) {         \
    __assume(0);      \
  }
#else
#define A_NEVER(cond)
#endif
#endif

#if defined(__cplusplus) || defined(_MSC_VER)
#define A_RESTRICT __restrict
#else
#define A_RESTRICT
#endif

namespace adrt {
enum class Sign : int_fast8_t {
  Negative = -1,
  Positive = 1,
};

template <typename Scalar>
struct Tensor2DTyped;

struct Tensor2D {
  using stride_t = int_fast32_t;
  int32_t height;
  int32_t width;
  stride_t stride;
  uint8_t *data;

  template <typename Scalar>
  Tensor2DTyped<Scalar> const &as() const {
    return reinterpret_cast<Tensor2DTyped<Scalar> const &>(*this);
  }
  Tensor2D(int32_t height, int32_t width, stride_t stride, uint8_t *data)
      : height{height}, width{width}, stride{stride}, data{data} {}
};

template <typename Scalar>
struct Tensor2DTyped: Tensor2D {};
//
// this is for first version, next version should just carry (begin, end)
//
static inline Tensor2D slice_no_checks(Tensor2D const &tensor, int begin,
                                       int end) {
  return {
      /*height = */ end - begin,
      /*width = */ tensor.width,
      /*stride = */ tensor.stride,
      /*data = */ tensor.data + begin * tensor.stride,
  };
}

template <typename Scalar>
static inline Scalar *A_LINE(Tensor2DTyped<Scalar> const &tensor,
                             int_fast32_t n) {
  A_NEVER(n < 0 || n >= tensor.height);
  return reinterpret_cast<Scalar *>(tensor.data + tensor.stride * (n));
}

static inline void copy_tensor(Tensor2D const &dst, Tensor2D const &src,
                               size_t scalar_size) {
  A_NEVER(dst.height != src.height || dst.width != src.width);
  uint8_t *line_dst = dst.data;
  uint8_t const *line_src = src.data;
  size_t const line_length = src.width * scalar_size;
  for (int_fast32_t y = 0; y < src.height; ++y) {
    std::memcpy(line_dst, line_src, line_length);
    line_dst += dst.stride;
    line_src += src.stride;
  }
}

}  // namespace adrt

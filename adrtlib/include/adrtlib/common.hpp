#pragma once
#include <stdlib.h>  // abort

#include <cstdint>

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

#ifdef __cplusplus
#define A_RESTRICT
#else
#define A_RESTRICT restrict
#endif

namespace adrt {
enum class Sign : int_fast8_t {
  Negative = -1,
  Positive = 1,
};

struct Tensor2D {
  int32_t height;
  int32_t width;
  int_fast32_t stride;
  uint8_t *data;
};

template <typename Scalar>
struct Tensor2DTyped: Tensor2D {};
//
// this is for first version, next version should just carry (begin, end)
//
static inline Tensor2D slice_no_checks(Tensor2D const &tensor, int begin,
                                       int end) {
  return (adrt::Tensor2D){
      .height = end - begin,
      .width = tensor.width,
      .stride = tensor.stride,
      .data = tensor.data + begin * tensor.stride,
  };
}

template <typename Scalar>
static inline Scalar *A_LINE(Tensor2DTyped<Scalar> const &tensor,
                             int_fast32_t n) {
  A_NEVER(n < 0 || n >= tensor.height);
  return reinterpret_cast<Scalar *>(tensor.data + tensor.stride * (n));
}

}  // namespace adrt

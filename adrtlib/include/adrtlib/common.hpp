#pragma once
#include <cstdint>

namespace adrt {
enum class Sign : int_fast8_t {
  Negative = -1,
  Positive = 1,
};

typedef struct {
  int32_t height;
  int32_t width;
  int_fast32_t stride;
  uint8_t *data;
} Tensor2D;

}  // namespace adrt

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

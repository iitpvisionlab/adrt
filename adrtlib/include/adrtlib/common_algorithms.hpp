#pragma once
#include <cstring>  // std::memcpy

#include "common.hpp"

namespace adrt {

template <typename Scalar>
static inline void add(Scalar *A_RESTRICT dst, Scalar const *A_RESTRICT src0,
                       Scalar const *A_RESTRICT src1, int const width) {
  A_NEVER(width < 0);
  for (int i = 0; i != width; ++i) {
    dst[i] = src0[i] + src1[i];
  }
}

template <typename Scalar>
static inline void add_with_2nd_shifted(Scalar *A_RESTRICT dst,
                                        Scalar const *A_RESTRICT src0,
                                        Scalar const *A_RESTRICT src1,
                                        int const width, int const shift) {
  A_NEVER(width <= 0 || shift > width);
  int const split = width - shift;
  add(dst, src0, src1 + split, shift);
  add(dst + shift, src0 + shift, src1, split);
}

template <typename Scalar>
static inline void rotate(Scalar *A_RESTRICT dst, Scalar *A_RESTRICT src,
                          int width, int rotation) {
  A_NEVER(width < 0 || rotation >= width);
  int const split = width - rotation;
  std::memcpy(dst, src + split, rotation * sizeof(Scalar));
  std::memcpy(dst + rotation, src, split * sizeof(Scalar));
}

template <typename Scalar>
static inline void ProcessPair(Scalar *A_RESTRICT line0,
                               Scalar *A_RESTRICT line1,
                               Scalar *A_RESTRICT buffer, int width, Sign sign,
                               int shift1) {
  A_NEVER(width <= 0 || shift1 < 0);
  rotate(buffer, line1, width, shift1);
  add(line1, line0, buffer, width);

  if (sign == Sign::Positive) {
    // do shift -1
    add(line0, line0, buffer + 1, width - 1);
    add(line0 + width - 1, line0 + width - 1, buffer, 1);
  } else {
    // do shift +1
    add(line0, line0, buffer + width - 1, 1);
    add(line0 + 1, line0 + 1, buffer, width - 1);
  }
}

template <typename Scalar>
static inline void ProcessLineAndSaveT(Scalar *A_RESTRICT line0,
                                       Scalar const *A_RESTRICT line1,
                                       int const width, int const shift) {
  add_with_2nd_shifted(line0, line0, line1, width, shift);
}

template <typename Scalar>
static inline void ProcessLineWithoutSavingT(Scalar const *A_RESTRICT line0,
                                             Scalar *A_RESTRICT line1,
                                             Scalar *A_RESTRICT buffer,
                                             int const width, int const shift) {
  rotate(buffer, line1, width, shift);
  add(line1, line0, buffer, width);
}

static inline int apply_sign(Sign sign, int value, int width) {
  return (sign == Sign::Positive || value == 0) ? (value) : (width - value);
}

static inline double round05(double value) {
  A_NEVER(value < 0.0);
  double const int_value = static_cast<double>(static_cast<int>(value));
  return value - int_value > 0.5 ? (int_value + 1.0) : int_value;
};

static inline uint32_t div_by_pow2(uint32_t n) {
  if ((n & (n - 1)) == 0) {
    return n >> 1;
  }
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  return (n >> 1) + 1;
}

}  // namespace adrt
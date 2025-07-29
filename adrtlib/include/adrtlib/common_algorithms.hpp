#pragma once
#include <cstring>  // memcpy

#include "common.hpp"

namespace adrt {
static inline void add(float *A_RESTRICT dst, float const *A_RESTRICT src0,
                       float const *A_RESTRICT src1, int const width) {
  A_NEVER(width < 0);
  for (int i = 0; i != width; ++i) {
    dst[i] = src0[i] + src1[i];
  }
}

static inline void add_with_2nd_shifted(float *A_RESTRICT dst,
                                        float const *A_RESTRICT src0,
                                        float const *A_RESTRICT src1,
                                        int const width, int const shift) {
  A_NEVER(width <= 0 || shift > width);
  int const split = width - shift;
  add(dst, src0, src1 + split, shift);
  add(dst + shift, src0 + shift, src1, split);
}

static inline void rotate(float *A_RESTRICT dst, float *A_RESTRICT src,
                          int width, int rotation) {
  A_NEVER(width < 0 || rotation >= width);
  int const split = width - rotation;
  memcpy(dst, src + split, rotation * sizeof(float));
  memcpy(dst + rotation, src, split * sizeof(float));
}

static inline void ProcessPair(float *A_RESTRICT line0, float *A_RESTRICT line1,
                               float *A_RESTRICT buffer, int width, Sign sign,
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

static inline void ProcessLineAndSaveT(float *A_RESTRICT line0,
                                       float const *A_RESTRICT line1,
                                       int const width, int const shift) {
  add_with_2nd_shifted(line0, line0, line1, width, shift);
}

static inline void ProcessLineWithoutSavingT(float const *A_RESTRICT line0,
                                             float *A_RESTRICT line1,
                                             float *A_RESTRICT buffer,
                                             int const width, int const shift) {
  rotate(buffer, line1, width, shift);
  add(line1, line0, buffer, width);
}

static inline int apply_sign(Sign sign, int value, int width) {
  return (sign == Sign::Positive || value == 0) ? (value) : (width - value);
}
}  // namespace adrt
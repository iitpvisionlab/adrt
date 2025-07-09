#include <immintrin.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>  //abort
#include <string.h>
#include <unistd.h>

#include "common.hpp"

#define A_LINE(tensor, n) (float *)((tensor)->data + (tensor)->stride * (n))

//
// this is for first version, next version should just carry (begin, end)
//
namespace adrt {
static Tensor2D slice_no_checks(Tensor2D const *tensor, int begin, int end) {
  return (adrt::Tensor2D){
      .height = end - begin,
      .width = tensor->width,
      .stride = tensor->stride,
      .data = tensor->data + begin * tensor->stride,
  };
}

// XXX: remove dst argument?
static inline void add(float *A_RESTRICT dst, float const *A_RESTRICT src0,
                       float const *A_RESTRICT src1, int const width) {
  A_NEVER(width <= 0);
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

static int fht2ids_core(int const h, Sign sign, int K[], int const K_T[],
                        int const K_B[], float buffer[], Tensor2D const *I_T,
                        Tensor2D const *I_B) {
  int t_B, t_T, k_T, k_B, t;
  int const width = I_T->width;
  if (h % 2 == 0) {
    for (t = 0; t < h; t += 2) {
      t_B = t_T = t / 2;
      k_T = K_T[t_T];
      k_B = K_B[t_B];
      ProcessPair(A_LINE(I_T, k_T), A_LINE(I_B, k_B), buffer, width, sign,
                  apply_sign(sign, (t - t_B + 1), width));
      K[t] = k_T;
      K[t + 1] = I_T->height + k_B;
    }
  } else {
    int const t_L_3deg_plus_one = round((double)h / 4);
    int const num_t_to_preprocess = h - 2 * t_L_3deg_plus_one;
    t_T = h / 2 - 1;
    t_B = h - h / 2 - 1;
    for (t = h - 1; t != h - 1 - num_t_to_preprocess; --t) {
      k_T = K_T[t_T];
      k_B = K_B[t_B];
      int const shift = apply_sign(sign, t - t_B, width);
      float *l_t = A_LINE(I_T, k_T);
      float *l_b = A_LINE(I_B, k_B);
      if (t % 2 == 0) {
        ProcessLineWithoutSavingT(l_t, l_b, buffer, width, shift);
        K[t] = I_T->height + k_B;
        t_B -= 1;
      } else {
        ProcessLineAndSaveT(l_t, l_b, width, shift);
        K[t] = k_T;
        t_T -= 1;
      }
    }
    for (t = 0; t != 2 * t_L_3deg_plus_one; t += 2) {
      t_B = t_T = t / 2;
      k_T = K_T[t_T];
      k_B = K_B[t_B];
      ProcessPair(A_LINE(I_T, k_T), A_LINE(I_B, k_B), buffer, width, sign,
                  apply_sign(sign, (t - t_B + 1), width));
      K[t] = k_T;
      K[t + 1] = I_T->height + k_B;
    }
  }
  return 0;
}

void fht2ids_recursive(Tensor2D const *src, Sign sign, int swaps[],
                       int swaps_buffer[], float line_buffer[]) {
  int const height = src->height;
  if A_UNLIKELY (height <= 1) {
    return;
  }
  int const h_T = height / 2;
  Tensor2D I_T = slice_no_checks(src, 0, h_T);
  Tensor2D I_B = slice_no_checks(src, h_T, src->height);

  memcpy(swaps_buffer, swaps, height * sizeof(*swaps_buffer));

  if (I_T.height > 1) {
    fht2ids_recursive(&I_T, sign, swaps, swaps_buffer, line_buffer);
  }
  if (I_B.height > 1) {
    fht2ids_recursive(&I_B, sign, swaps + h_T, swaps_buffer + h_T, line_buffer);
  }
  memcpy(swaps_buffer, swaps, height * sizeof(*swaps_buffer));
  fht2ids_core(height, sign, swaps, swaps_buffer + 0, swaps_buffer + h_T,
               line_buffer, &I_T, &I_B);
}
}  // namespace adrt

#include <gtest/gtest.h>

#include <adrtlib/adrtlib.hpp>

template <size_t N>
void check_equal(float const (&a)[N], float const (&b)[N]) {
  for (int i = 0; i != N; ++i) {
    ASSERT_FLOAT_EQ(a[i], b[i]) << "Arrays differ at index " << i;
  }
}

template <size_t N>
void check_equal(int const (&a)[N], int const (&b)[N]) {
  for (int i = 0; i != N; ++i) {
    ASSERT_EQ(a[i], b[i]) << "Arrays differ at index " << i;
  }
}

template <size_t N>
void ref_shift(float (&dst)[N], float (&src)[N], int shift) {
  for (int i = 0; i != N; ++i) {
    dst[i] = src[(-shift + N + i) % N];
  }
}

static void print_tensor(adrt::Tensor2D const &tensor) {
  for (int y = 0; y != tensor.height; ++y) {
    float const *line = (float *)(tensor.data + y * tensor.stride);
    for (int x = 0; x != tensor.width; ++x) {
      printf("%.0f ", line[x]);
    }
    printf("\n");
  }
  printf("\n");
}

template <typename F>
void test_adrt_inplace(F func, adrt::Sign sing, float data[], float const ref[],
                       int width, int height) {
  adrt::Tensor2D tensor{.height = height,
                        .width = width,
                        .stride = static_cast<ssize_t>(sizeof(*data)) *
                                  static_cast<ssize_t>(width),
                        .data = reinterpret_cast<uint8_t *>(data)};
  func(tensor, sing);
  for (size_t y = 0; y != height; ++y) {
    for (size_t x = 0; x != width; ++x) {
      size_t const idx = y * width + x;
      ASSERT_FLOAT_EQ(ref[idx], data[idx])
          << "Arrays differ at " << y << ", " << x;
    }
  }
}

class libadrt_test: public testing::TestWithParam<int> {};

TEST(libadrt_test, rotate) {
  float src[] = {0, 1, 2, 3, 4, 5, 6};
  float exp[] = {-1, -1, -1, -1, -1, -1, -1};
  adrt::rotate(exp + 1, src + 1, sizeof(src) / sizeof(*src) - 2, 2);
  float const ref[] = {-1, 4, 5, 1, 2, 3, -1};
  check_equal(ref, exp);
}

TEST(libadrt_test, ProcessPair_2_2) {
  float line0[] = {1, 3};
  float line1[] = {5, 40};
  float buffer[] = {-1, -1};
  adrt::ProcessPair(line0, line1, buffer, sizeof(line0) / sizeof(*line0),
                    adrt::Sign::Positive, 1);
  float exp0[] = {6, 43};
  float exp1[] = {41, 8};
  check_equal(exp0, line0);
  check_equal(exp1, line1);
}

TEST(libadrt_test, ProcessPair_2_4_pos) {
  float line0[] = {1, 3, 2, 4};
  float line1[] = {5, 0, 1, 7};
  float buffer[] = {-1, -1, -1, -1};
  adrt::ProcessPair(line0, line1, buffer, sizeof(line0) / sizeof(*line0),
                    adrt::Sign::Positive, 1);
  float exp0[] = {6, 3, 3, 11};
  float exp1[] = {8, 8, 2, 5};
  check_equal(exp0, line0);
  check_equal(exp1, line1);
}

TEST(libadrt, fht2ids_2_2) {
  float data[] = {1.0f, 3.0f, 5.0f, 40.0f};
  int swaps[] = {0, 0};
  float const ref[] = {6.0f, 43.0f, 41.0f, 8.0f};

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        int swaps_buffer[] = {-1, -1};
        float line_buffer[] = {-1.0f, -1.0f};
        adrt::fht2ids_recursive(&tensor, sign, swaps, swaps_buffer,
                                line_buffer);
      },
      adrt::Sign::Positive, data, ref, 2, 2);

  int const ref_swaps[] = {0, 1};
  check_equal(swaps, ref_swaps);
}

TEST(libadrt, fht2ids_3_3) {
  float data[] = {1, 5, 8, 3, 7, 2, 4, 9, 6};
  float const ref[] = {8, 21, 16, 9, 12, 24, 12, 14, 19};
  adrt::Tensor2D src{.height = 3,
                     .width = 3,
                     .stride = 3 * sizeof(float),
                     .data = (uint8_t *)data};
  int swaps[] = {0, 0, 0};
  int swaps_buffer[] = {-1, -1, -1};
  float line_buffer[] = {-1.0f, -1.0f, -1.0f};
  adrt::fht2ids_recursive(&src, adrt::Sign::Positive, swaps, swaps_buffer,
                          line_buffer);
  check_equal(ref, data);
}

TEST(libadrt, fht2ids_4_4_pos) {
  float data[] = {
      1.0f, 3.0f, 2.0f, 4.0f,   // 0
      5.0f, 0.0f, 1.0f, 7.0f,   // 1
      2.0f, 6.0f, 5.0f, 3.0f,   // 2
      9.0f, 4.0f, 8.0f, 40.0f,  // 3
  };
  float const ref[] = {
      17, 13, 16, 54,  // 0
      19, 50, 17, 14,  // 2
      49, 14, 13, 24,  // 1
      17, 19, 44, 20,  // 3
  };

  int swaps[] = {0, 0, 0, 0};
  int swaps_buffer[] = {-1, -1, -1, -1};
  float line_buffer[] = {-1.0f, -1.0f, -1.0f, -1.0f};

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        adrt::fht2ids_recursive(&tensor, sign, swaps, swaps_buffer,
                                line_buffer);
      },
      adrt::Sign::Positive, data, ref, 4, 4);
  int const ref_swaps[] = {0, 2, 1, 3};
  check_equal(swaps, ref_swaps);
}

TEST(libadrt, fht2ids_4_4_neg) {
  float data[] = {
      1, 3, 2, 4,   // 0
      5, 0, 1, 7,   // 1
      2, 6, 5, 3,   // 2
      9, 4, 8, 40,  // 3
  };
  float const ref[] = {
      17, 13, 16, 54,  // 0
      15, 49, 21, 15,  // 2
      16, 16, 46, 22,  // 1
      46, 16, 15, 23,  // 3
  };

  int swaps[] = {0, 0, 0, 0};
  int swaps_buffer[] = {-1, -1, -1, -1};
  float line_buffer[] = {-1.0f, -1.0f, -1.0f, -1.0f};

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        adrt::fht2ids_recursive(&tensor, sign, swaps, swaps_buffer,
                                line_buffer);
        // print_tensor(tensor);
      },
      adrt::Sign::Negative, data, ref, 4, 4);
  int const ref_swaps[] = {0, 2, 1, 3};
  check_equal(swaps, ref_swaps);
}

TEST(libadrt, fht2ids_5_5_pos) {
  float data[] = {
      8,  2,  6,  16, 20,  // 0
      13, 9,  1,  24, 7,   // 1
      23, 3,  4,  11, 18,  // 2
      15, 25, 5,  22, 21,  // 3
      12, 19, 10, 17, 14,  // 4
  };

  float const ref[] = {
      71, 58, 26, 90, 80,  // 0
      41, 72, 73, 47, 92,  // 2
      74, 61, 54, 59, 77,  // 3
      78, 69, 37, 88, 53,  // 1
      50, 65, 76, 49, 85,  // 4
  };

  int swaps[] = {0, 0, 0, 0, 0};
  int swaps_buffer[] = {-1, -1, -1, -1, -1};
  float line_buffer[] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        adrt::fht2ids_recursive(&tensor, sign, swaps, swaps_buffer,
                                line_buffer);
        // print_tensor(tensor);
      },
      adrt::Sign::Positive, data, ref, 5, 5);
  int const ref_swaps[] = {0, 2, 3, 1, 4};
  check_equal(swaps, ref_swaps);
}

TEST(libadrt, fht2ids_5_5_neg) {
  float data[] = {
      8,  2,  6,  16, 20,  // 0
      13, 9,  1,  24, 7,   // 1
      23, 3,  4,  11, 18,  // 2
      15, 25, 5,  22, 21,  // 3
      12, 19, 10, 17, 14,  // 4
  };

  float const ref[] = {
      71, 58, 26, 90, 80,  // 0
      60, 49, 75, 90, 51,  // 2
      68, 30, 57, 93, 77,  // 3
      39, 54, 53, 85, 94,  // 1
      57, 47, 82, 81, 58,  // 4
  };

  int swaps[] = {0, 0, 0, 0, 0};
  int swaps_buffer[] = {-1, -1, -1, -1, -1};
  float line_buffer[] = {-1.0f, -1.0f, -1.0f, -1.0f, -1.0f};

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        adrt::fht2ids_recursive(&tensor, sign, swaps, swaps_buffer,
                                line_buffer);
        // print_tensor(tensor);
      },
      adrt::Sign::Negative, data, ref, 5, 5);
  int const ref_swaps[] = {0, 2, 3, 1, 4};
  check_equal(swaps, ref_swaps);
}

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

struct ADRTTestCase {
  std::string description;
  adrt::Sign sign;
  int const size;
  float const *const data;
  float const *const ref;
  int const *const ref_swaps;
  std::function<void(adrt::Tensor2D const *, adrt::Sign, int *, int *, float *)>
      adrt_function;
};

::std::string PrintToString(const ADRTTestCase &v) { return v.description; }

template <typename T>
::std::string testing::PrintToString(const ADRTTestCase &value) {
  return value.description;
}

class ADRTLibTest: public testing::TestWithParam<ADRTTestCase> {};

std::vector<ADRTTestCase> GenerateTestCases() {
  /*
    2x2
  */
  static float const data_2x2[] = {1.0f, 3.0f, 5.0f, 40.0f};
  static int const ref_2x2_swaps[] = {0, 1};
  static float const ref_2x2[] = {6.0f, 43.0f, 41.0f, 8.0f};

  /*
    3x3
  */
  static float const data_3x3[] = {1, 5, 8, 3, 7, 2, 4, 9, 6};
  static float const ref_3x3_pos[] = {8, 21, 16, 9, 12, 24, 12, 14, 19};
  static float const ref_3x3_neg[] = {8, 21, 16, 17, 13, 15, 14, 11, 20};
  static int const ref_3x3_swaps[] = {0, 1, 2};

  /*
    4x4
  */
  static float const data_4x4[] = {
      1.0f, 3.0f, 2.0f, 4.0f,   // 0
      5.0f, 0.0f, 1.0f, 7.0f,   // 1
      2.0f, 6.0f, 5.0f, 3.0f,   // 2
      9.0f, 4.0f, 8.0f, 40.0f,  // 3
  };
  static float const ref_4x4_pos[] = {
      17, 13, 16, 54,  // 0
      19, 50, 17, 14,  // 2
      49, 14, 13, 24,  // 1
      17, 19, 44, 20,  // 3
  };
  static float const ref_4x4_neg[] = {
      17, 13, 16, 54,  // 0
      15, 49, 21, 15,  // 2
      16, 16, 46, 22,  // 1
      46, 16, 15, 23,  // 3
  };
  static int const ref_4x4_swaps[] = {0, 2, 1, 3};
  /*
    5x5
  */
  static float const data_5x5[] = {
      8,  2,  6,  16, 20,  // 0
      13, 9,  1,  24, 7,   // 1
      23, 3,  4,  11, 18,  // 2
      15, 25, 5,  22, 21,  // 3
      12, 19, 10, 17, 14,  // 4
  };

  static float const ref_5x5_pos[] = {
      71, 58, 26, 90, 80,  // 0
      41, 72, 73, 47, 92,  // 2
      74, 61, 54, 59, 77,  // 3
      78, 69, 37, 88, 53,  // 1
      50, 65, 76, 49, 85,  // 4
  };

  static float const ref_5x5_neg[] = {
      71, 58, 26, 90, 80,  // 0
      60, 49, 75, 90, 51,  // 2
      68, 30, 57, 93, 77,  // 3
      39, 54, 53, 85, 94,  // 1
      57, 47, 82, 81, 58,  // 4
  };
  static int const ref_5x5_swaps[] = {0, 2, 3, 1, 4};

  return {
      /* 2x2 */
      {"fht2ids_recursive_Positive_2x2", adrt::Sign::Positive, 2, data_2x2,
       ref_2x2, ref_2x2_swaps, adrt::fht2ids_recursive},
      {"fht2ids_recursive_Negative_2x2", adrt::Sign::Negative, 2, data_2x2,
       ref_2x2, ref_2x2_swaps, adrt::fht2ids_recursive},
      {"fht2ids_non_recursive_Positive_2x2", adrt::Sign::Positive, 2, data_2x2,
       ref_2x2, ref_2x2_swaps, adrt::fht2ids_non_recursive},
      {"fht2ids_non_recursive_Negative_2x2", adrt::Sign::Negative, 2, data_2x2,
       ref_2x2, ref_2x2_swaps, adrt::fht2ids_non_recursive},

      /* 3x3 */
      {"fht2ids_recursive_Positive_3x3", adrt::Sign::Positive, 3, data_3x3,
       ref_3x3_pos, ref_3x3_swaps, adrt::fht2ids_recursive},
      {"fht2ids_recursive_Negative_3x3", adrt::Sign::Negative, 3, data_3x3,
       ref_3x3_neg, ref_3x3_swaps, adrt::fht2ids_recursive},
      {"fht2ids_non_recursive_Positive_3x3", adrt::Sign::Positive, 3, data_3x3,
       ref_3x3_pos, ref_3x3_swaps, adrt::fht2ids_non_recursive},
      {"fht2ids_non_recursive_Negative_3x3", adrt::Sign::Negative, 3, data_3x3,
       ref_3x3_neg, ref_3x3_swaps, adrt::fht2ids_non_recursive},

      /* 4x4 */
      {"fht2ids_recursive_Positive_4x4", adrt::Sign::Positive, 4, data_4x4,
       ref_4x4_pos, ref_4x4_swaps, adrt::fht2ids_recursive},
      {"fht2ids_recursive_Negative_4x4", adrt::Sign::Negative, 4, data_4x4,
       ref_4x4_neg, ref_4x4_swaps, adrt::fht2ids_recursive},
      {"fht2ids_non_recursive_Positive_4x4", adrt::Sign::Positive, 4, data_4x4,
       ref_4x4_pos, ref_4x4_swaps, adrt::fht2ids_non_recursive},
      {"fht2ids_non_recursive_Negative_4x4", adrt::Sign::Negative, 4, data_4x4,
       ref_4x4_neg, ref_4x4_swaps, adrt::fht2ids_non_recursive},

      /* 5x5 */
      {"fht2ids_recursive_Positive_5x5", adrt::Sign::Positive, 5, data_5x5,
       ref_5x5_pos, ref_5x5_swaps, adrt::fht2ids_recursive},
      {"fht2ids_recursive_Negative_5x5", adrt::Sign::Negative, 5, data_5x5,
       ref_5x5_neg, ref_5x5_swaps, adrt::fht2ids_recursive},
      {"fht2ids_non_recursive_Positive_5x5", adrt::Sign::Positive, 5, data_5x5,
       ref_5x5_pos, ref_5x5_swaps, adrt::fht2ids_non_recursive},
      {"fht2ids_non_recursive_Negative_5x5", adrt::Sign::Negative, 5, data_5x5,
       ref_5x5_neg, ref_5x5_swaps, adrt::fht2ids_non_recursive},
  };
}

TEST_P(ADRTLibTest, fht2ids_test) {
  auto const test_case = GetParam();
  std::vector<float> data{test_case.data,
                          test_case.data + test_case.size * test_case.size};
  std::vector<float> const ref{test_case.ref,
                               test_case.ref + test_case.size * test_case.size};

  std::vector<int> swaps(test_case.size, 0);
  std::vector<int> swaps_buffer(test_case.size, -1);
  std::vector<float> line_buffer(test_case.size, -1.0f);

  test_adrt_inplace(
      [&](adrt::Tensor2D const &tensor, adrt::Sign sign) {
        test_case.adrt_function(&tensor, sign, swaps.data(),
                                swaps_buffer.data(), line_buffer.data());
      },
      test_case.sign, data.data(), ref.data(), test_case.size, test_case.size);
  std::vector<int> const ref_swaps{test_case.ref_swaps,
                                   test_case.ref_swaps + test_case.size};
  EXPECT_EQ(swaps, ref_swaps);
}

INSTANTIATE_TEST_SUITE_P(ADRTLibTestRenameMe, ADRTLibTest,
                         testing::ValuesIn(GenerateTestCases()),
                         testing::PrintToStringParamName());

TEST(adrtlib_test, rotate) {
  float src[] = {0, 1, 2, 3, 4, 5, 6};
  float exp[] = {-1, -1, -1, -1, -1, -1, -1};
  adrt::rotate(exp + 1, src + 1, sizeof(src) / sizeof(*src) - 2, 2);
  float const ref[] = {-1, 4, 5, 1, 2, 3, -1};
  check_equal(ref, exp);
}

TEST(adrtlib_test, ProcessPair_2_2_pos) {
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

TEST(adrtlib_test, ProcessPair_2_4_pos) {
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

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

using ADRTFunction = std::function<void(adrt::Tensor2D const &, adrt::Sign,
                                        int *, int *, float *)>;

struct ADRTTestCase {
  ADRTTestCase(std::string description, adrt::Sign sign, int const size,
               float const *const data, float const *const ref,
               int const *const ref_swaps, ADRTFunction adrt_function)
      : description{description},
        sign{sign},
        size{size},
        data{data},
        ref{ref},
        ref_swaps{ref_swaps},
        adrt_function{adrt_function} {}
  std::string description;
  adrt::Sign sign;
  int const size;
  float const *const data;
  float const *const ref;
  int const *const ref_swaps;
  ADRTFunction adrt_function;  // fht2ids_non_recursive or fht2ids_recursive
};

::std::string PrintToString(const ADRTTestCase &v) { return v.description; }

template <typename T>
::std::string testing::PrintToString(const ADRTTestCase &value) {
  return value.description;
}

enum class FunctionType {
  fht2ds,
  fht2dt,
};

/*
  2x2
*/
static float const data_2x2[] = {1.0f, 3.0f, 5.0f, 40.0f};
static float const ref_2x2[] = {6.0f, 43.0f, 41.0f, 8.0f};
static int const ref_2x2_swaps[] = {0, 1};

/*
  3x3
*/
static float const data_3x3[] = {1, 5, 8, 3, 7, 2, 4, 9, 6};
static float const ref_3x3_pos_ds[] = {8, 21, 16, 9, 12, 24, 12, 14, 19};
static float const ref_3x3_neg_ds[] = {8, 21, 16, 17, 13, 15, 14, 11, 20};
static int const ref_3x3_swaps_ds[] = {0, 1, 2};

static float const ref_3x3_pos_dt[] = {8, 21, 16, 12, 14, 19, 10, 16, 19};
static float const ref_3x3_neg_dt[] = {8, 21, 16, 14, 11, 20, 13, 18, 14};
static int const ref_3x3_swaps_dt[] = {0, 2, 1};
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

static float const ref_5x5_pos_ds[] = {
    71, 58, 26, 90, 80,  // 0
    41, 72, 73, 47, 92,  // 2
    74, 61, 54, 59, 77,  // 3
    78, 69, 37, 88, 53,  // 1
    50, 65, 76, 49, 85,  // 4
};

static float const ref_5x5_pos_dt[] = {
    71, 58, 26, 90, 80,  // 0
    65, 76, 47, 58, 79,  // 2
    74, 61, 54, 59, 77,  // 4
    50, 65, 76, 49, 85,  // 1
    77, 63, 47, 68, 70,  // 3
};

static float const ref_5x5_neg_ds[] = {
    71, 58, 26, 90, 80,  // 0
    60, 49, 75, 90, 51,  // 2
    68, 30, 57, 93, 77,  // 3
    39, 54, 53, 85, 94,  // 1
    57, 47, 82, 81, 58,  // 4
};

static float const ref_5x5_neg_dt[] = {
    71, 58, 26, 90, 80,  // 0
    42, 43, 74, 75, 91,  // 2
    68, 30, 57, 93, 77,  // 4
    57, 47, 82, 81, 58,  // 1
    59, 37, 54, 91, 84,  // 3
};

static int const ref_5x5_swaps_ds[] = {0, 2, 3, 1, 4};
static int const ref_5x5_swaps_dt[] = {0, 2, 4, 1, 3};

/*
  7x7
*/

static float const data_7x7[] = {
    20, 12, 31, 28, 46, 29, 33,  // 0
    10, 26, 36, 13, 14, 42, 30,  // 1
    22, 25, 2,  32, 45, 16, 11,  // 2
    1,  37, 48, 27, 35, 5,  19,  // 3
    17, 47, 7,  38, 41, 4,  15,  // 4
    23, 49, 34, 40, 39, 3,  6,   // 5
    18, 43, 21, 44, 9,  8,  24,  // 6
};

static float const ref_7x7_pos_ds[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    106, 92,  207, 213, 211, 248, 148,  //  3
    200, 162, 114, 119, 162, 237, 231,  //  1
    116, 122, 245, 183, 254, 211, 94,   //  5
    190, 79,  112, 167, 253, 199, 225,  //  4
    118, 89,  130, 191, 238, 208, 251,  //  2
    216, 140, 185, 117, 129, 217, 221,  //  6
};

static float const ref_7x7_pos_dt[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    190, 79,  112, 167, 253, 199, 225,  //  4
    89,  125, 243, 175, 218, 275, 100,  //  2
    216, 140, 185, 117, 129, 217, 221,  //  6
    98,  158, 256, 162, 262, 181, 108,  //  1
    168, 150, 110, 134, 233, 189, 241,  //  5
    76,  90,  193, 205, 203, 250, 208,  //  3
};

static float const ref_7x7_neg_ds[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    210, 189, 189, 174, 143, 145, 175,  //  3
    163, 146, 133, 160, 217, 203, 203,  //  1
    228, 173, 218, 197, 125, 151, 133,  //  5
    206, 165, 142, 131, 206, 188, 187,  //  4
    210, 163, 163, 126, 179, 180, 204,  //  2
    143, 143, 175, 164, 225, 174, 201,  //  6
};

static float const ref_7x7_neg_dt[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    206, 165, 142, 131, 206, 188, 187,  //  4
    209, 173, 213, 209, 112, 140, 169,  //  2
    143, 143, 175, 164, 225, 174, 201,  //  6
    192, 162, 239, 189, 155, 137, 151,  //  1
    204, 145, 139, 173, 210, 196, 158,  //  5
    177, 175, 214, 152, 120, 184, 203,  //  3
};

static int const ref_7x7_swaps_ds[] = {0, 3, 1, 5, 4, 2, 6};
static int const ref_7x7_swaps_dt[] = {0, 4, 2, 6, 1, 5, 3};

struct Ref {
  float const *data;      // data for input
  float const *ref_data;  // reference values for output data
  int const *ref_swaps;   // reference values for output swaps
 private:
  Ref(float const *data, float const *ref_data, int const *ref_swaps)
      : data{data}, ref_data{ref_data}, ref_swaps{ref_swaps} {}

 public:
  static Ref get(int size, adrt::Sign sign, FunctionType func) {
    switch (size) {
      case 2:
        return Ref{data_2x2, ref_2x2, ref_2x2_swaps};
      case 3:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{
                data_3x3,
                func == FunctionType::fht2ds ? ref_3x3_pos_ds : ref_3x3_pos_dt,
                func == FunctionType::fht2ds ? ref_3x3_swaps_ds
                                             : ref_3x3_swaps_dt};
          case adrt::Sign::Negative:
            return Ref{
                data_3x3,
                func == FunctionType::fht2ds ? ref_3x3_neg_ds : ref_3x3_neg_dt,
                func == FunctionType::fht2ds ? ref_3x3_swaps_ds
                                             : ref_3x3_swaps_dt};
        }
      case 4:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{data_4x4, ref_4x4_pos, ref_4x4_swaps};
          case adrt::Sign::Negative:
            return Ref{data_4x4, ref_4x4_neg, ref_4x4_swaps};
        }
      case 5:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{
                data_5x5,
                func == FunctionType::fht2ds ? ref_5x5_pos_ds : ref_5x5_pos_dt,
                func == FunctionType::fht2ds ? ref_5x5_swaps_ds
                                             : ref_5x5_swaps_dt};
          case adrt::Sign::Negative:
            return Ref{
                data_5x5,
                func == FunctionType::fht2ds ? ref_5x5_neg_ds : ref_5x5_neg_dt,
                func == FunctionType::fht2ds ? ref_5x5_swaps_ds
                                             : ref_5x5_swaps_dt};
        }
      case 7:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{
                data_7x7,
                func == FunctionType::fht2ds ? ref_7x7_pos_ds : ref_7x7_pos_dt,
                func == FunctionType::fht2ds ? ref_7x7_swaps_ds
                                             : ref_7x7_swaps_dt};
          case adrt::Sign::Negative:
            return Ref{
                data_7x7,
                func == FunctionType::fht2ds ? ref_7x7_neg_ds : ref_7x7_neg_dt,
                func == FunctionType::fht2ds ? ref_7x7_swaps_ds
                                             : ref_7x7_swaps_dt};
        }

      default:
        assert(0);
        return {nullptr, nullptr, nullptr};
    }
  }
};

static std::vector<ADRTTestCase> GenerateTestFHT2DSCases() {
  using SignPair = std::pair<adrt::Sign, std::string>;
  std::array<SignPair, 2> const sign_pairs = {
      SignPair(adrt::Sign::Positive, "Positive"),
      SignPair(adrt::Sign::Negative, "Negative")};

  using FunctionPair = std::tuple<ADRTFunction, std::string, FunctionType>;
  std::array<FunctionPair, 4> const function_pairs = {
      FunctionPair(adrt::fht2ids_recursive, "fht2ids_recursive",
                   FunctionType::fht2ds),
      FunctionPair(adrt::fht2ids_non_recursive, "fht2ids_non_recursive",
                   FunctionType::fht2ds),
      FunctionPair(
          [](adrt::Tensor2D const &src, adrt::Sign sign, int swaps[],
             int swaps_buffer[], float line_buffer[]) {
            std::vector<int> t_B_to_check;
            std::vector<int> t_T_to_check;
            std::vector<bool> t_processed;
            std::vector<adrt::OutDegree> out_degrees(src.height);

            adrt::fht2idt_recursive(src, sign, swaps, swaps_buffer, line_buffer,
                                    out_degrees.data(), t_B_to_check,
                                    t_T_to_check, t_processed);
          },
          "fht2idt_recursive", FunctionType::fht2dt),
      FunctionPair(
          [](adrt::Tensor2D const &src, adrt::Sign sign, int swaps[],
             int swaps_buffer[], float line_buffer[]) {
            std::vector<int> t_B_to_check;
            std::vector<int> t_T_to_check;
            std::vector<bool> t_processed;
            std::vector<adrt::OutDegree> out_degrees(src.height);

            adrt::fht2idt_non_recursive(
                src, sign, swaps, swaps_buffer, line_buffer, out_degrees.data(),
                t_B_to_check, t_T_to_check, t_processed);
          },
          "fht2idt_non_recursive", FunctionType::fht2dt)};

  using SizePair = std::pair<int, std::string>;
  std::array<SizePair, 5> const size_pairs = {
      SizePair(2, "2x2"), SizePair(3, "3x3"), SizePair(4, "4x4"),
      SizePair(5, "5x5"), SizePair(7, "7x7")};

  std::vector<ADRTTestCase> out;
  for (auto [adrt_function, adrt_function_str, func_type] : function_pairs) {
    for (auto [sign, sign_str] : sign_pairs) {
      for (auto [size, size_str] : size_pairs) {
        Ref const ref = Ref::get(size, sign, func_type);
        out.emplace_back(adrt_function_str + "_" + sign_str + "_" + size_str,
                         sign, size, ref.data, ref.ref_data, ref.ref_swaps,
                         adrt_function);
      }
    }
  }

  return out;
}

class fht2ids_test: public testing::TestWithParam<ADRTTestCase> {};

TEST_P(fht2ids_test, suite) {
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
        test_case.adrt_function(tensor, sign, swaps.data(), swaps_buffer.data(),
                                line_buffer.data());
      },
      test_case.sign, data.data(), ref.data(), test_case.size, test_case.size);
  std::vector<int> const ref_swaps{test_case.ref_swaps,
                                   test_case.ref_swaps + test_case.size};
  EXPECT_EQ(swaps, ref_swaps);
}

INSTANTIATE_TEST_SUITE_P(ADRTLib, fht2ids_test,
                         testing::ValuesIn(GenerateTestFHT2DSCases()),
                         testing::PrintToStringParamName());

TEST(ADRTLib, rotate) {
  float src[] = {0, 1, 2, 3, 4, 5, 6};
  float exp[] = {-1, -1, -1, -1, -1, -1, -1};
  adrt::rotate(exp + 1, src + 1, sizeof(src) / sizeof(*src) - 2, 2);
  float const ref[] = {-1, 4, 5, 1, 2, 3, -1};
  check_equal(ref, exp);
}

TEST(ADRTLib, ProcessPair_2_2_pos) {
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

TEST(ADRTLib, ProcessPair_2_4_pos) {
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

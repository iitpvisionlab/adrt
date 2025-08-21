#include <gtest/gtest.h>

#include <adrtlib/adrtlib.hpp>
#include <array>

template <size_t N>
static void check_equal(float const (&a)[N], float const (&b)[N]) {
  for (int i = 0; i != N; ++i) {
    ASSERT_FLOAT_EQ(a[i], b[i]) << "Arrays differ at index " << i;
  }
}

template <size_t N>
static void check_equal(int const (&a)[N], int const (&b)[N]) {
  for (int i = 0; i != N; ++i) {
    ASSERT_EQ(a[i], b[i]) << "Arrays differ at index " << i;
  }
}

template <size_t N>
static void ref_shift(float (&dst)[N], float (&src)[N], int shift) {
  for (int i = 0; i != N; ++i) {
    dst[i] = src[(-shift + N + i) % N];
  }
}

// static void print_tensor(adrt::Tensor2D const &tensor) {
//   for (int y = 0; y != tensor.height; ++y) {
//     float const *line = (float *)(tensor.data + y * tensor.stride);
//     for (int x = 0; x != tensor.width; ++x) {
//       printf("%.0f ", line[x]);
//     }
//     printf("\n");
//   }
//   printf("\n");
// }

enum class IsInplace {
  Yes,
  No,
};

using ADRTTestFunction =
    std::function<void(adrt::Tensor2DTyped<float> const & /*dst*/,
                       adrt::Tensor2DTyped<float> const & /*src*/, adrt::Sign)>;

struct ADRTTestCase {
  ADRTTestCase(std::string description, adrt::Sign sign, int const size,
               float const *const data, float const *const ref,
               ADRTTestFunction adrt_function, IsInplace is_inplace)
      : description{description},
        sign{sign},
        size{size},
        data{data},
        ref{ref},
        adrt_function{adrt_function},
        is_inplace{is_inplace} {}
  std::string description;
  adrt::Sign sign;
  int const size;
  float const *const data;
  float const *const ref;
  ADRTTestFunction adrt_function;
  IsInplace is_inplace;
};

::std::string PrintToString(const ADRTTestCase &v) { return v.description; }

enum class FunctionType {
  fht2ds,
  fht2dt,
};

/*
  2x2
*/
static float const data_2x2[] = {1.0f, 3.0f, 5.0f, 40.0f};
static float const ref_2x2[] = {6.0f, 43.0f, 41.0f, 8.0f};

/*
  3x3
*/
static float const data_3x3[] = {1, 5, 8, 3, 7, 2, 4, 9, 6};
static float const ref_3x3_pos_ds[] = {8, 21, 16, 9, 12, 24, 12, 14, 19};
static float const ref_3x3_neg_ds[] = {8, 21, 16, 17, 13, 15, 14, 11, 20};

static float const ref_3x3_pos_dt[] = {8, 21, 16, 10, 16, 19, 12, 14, 19};
static float const ref_3x3_neg_dt[] = {8, 21, 16, 13, 18, 14, 14, 11, 20};
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
    49, 14, 13, 24,  // 1
    19, 50, 17, 14,  // 2
    17, 19, 44, 20,  // 3
};
static float const ref_4x4_neg[] = {
    17, 13, 16, 54,  // 0
    16, 16, 46, 22,  // 1
    15, 49, 21, 15,  // 2
    46, 16, 15, 23,  // 3
};
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
    74, 61, 54, 59, 77,  // 3
    78, 69, 37, 88, 53,  // 1
    41, 72, 73, 47, 92,  // 2
    50, 65, 76, 49, 85,  // 4
};

static float const ref_5x5_pos_dt[] = {
    71, 58, 26, 90, 80,  // 0
    74, 61, 54, 59, 77,  // 1
    77, 63, 47, 68, 70,  // 2
    65, 76, 47, 58, 79,  // 3
    50, 65, 76, 49, 85,  // 4
};

static float const ref_5x5_neg_ds[] = {
    71, 58, 26, 90, 80,  // 0
    68, 30, 57, 93, 77,  // 1
    39, 54, 53, 85, 94,  // 2
    60, 49, 75, 90, 51,  // 3
    57, 47, 82, 81, 58,  // 4
};

static float const ref_5x5_neg_dt[] = {
    71, 58, 26, 90, 80,  // 0
    68, 30, 57, 93, 77,  // 1
    59, 37, 54, 91, 84,  // 2
    42, 43, 74, 75, 91,  // 3
    57, 47, 82, 81, 58,  // 4
};

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
    116, 122, 245, 183, 254, 211, 94,   //  1
    106, 92,  207, 213, 211, 248, 148,  //  2
    118, 89,  130, 191, 238, 208, 251,  //  3
    190, 79,  112, 167, 253, 199, 225,  //  4
    200, 162, 114, 119, 162, 237, 231,  //  5
    216, 140, 185, 117, 129, 217, 221,  //  6
};

static float const ref_7x7_pos_dt[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    98,  158, 256, 162, 262, 181, 108,  //  1
    89,  125, 243, 175, 218, 275, 100,  //  2
    76,  90,  193, 205, 203, 250, 208,  //  3
    190, 79,  112, 167, 253, 199, 225,  //  4
    168, 150, 110, 134, 233, 189, 241,  //  5
    216, 140, 185, 117, 129, 217, 221,  //  6
};

static float const ref_7x7_neg_ds[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    228, 173, 218, 197, 125, 151, 133,  //  1
    210, 189, 189, 174, 143, 145, 175,  //  2
    210, 163, 163, 126, 179, 180, 204,  //  3
    206, 165, 142, 131, 206, 188, 187,  //  4
    163, 146, 133, 160, 217, 203, 203,  //  5
    143, 143, 175, 164, 225, 174, 201,  //  6
};

static float const ref_7x7_neg_dt[] = {
    111, 239, 179, 222, 229, 107, 138,  //  0
    192, 162, 239, 189, 155, 137, 151,  //  1
    209, 173, 213, 209, 112, 140, 169,  //  2
    177, 175, 214, 152, 120, 184, 203,  //  3
    206, 165, 142, 131, 206, 188, 187,  //  4
    204, 145, 139, 173, 210, 196, 158,  //  5
    143, 143, 175, 164, 225, 174, 201,  //  6
};

struct Ref {
  float const *data;      // data for input
  float const *ref_data;  // reference values for output data
 private:
  Ref(float const *data, float const *ref_data)
      : data{data}, ref_data{ref_data} {}

 public:
  static Ref get(int size, adrt::Sign sign, FunctionType func) {
    switch (size) {
      case 2:
        return Ref{data_2x2, ref_2x2};
      case 3:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{data_3x3, func == FunctionType::fht2ds ? ref_3x3_pos_ds
                                                              : ref_3x3_pos_dt};
          case adrt::Sign::Negative:
            return Ref{data_3x3, func == FunctionType::fht2ds ? ref_3x3_neg_ds
                                                              : ref_3x3_neg_dt};
        }
      case 4:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{data_4x4, ref_4x4_pos};
          case adrt::Sign::Negative:
            return Ref{data_4x4, ref_4x4_neg};
        }
      case 5:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{data_5x5, func == FunctionType::fht2ds ? ref_5x5_pos_ds
                                                              : ref_5x5_pos_dt};
          case adrt::Sign::Negative:
            return Ref{data_5x5, func == FunctionType::fht2ds ? ref_5x5_neg_ds
                                                              : ref_5x5_neg_dt};
        }
      case 7:
        switch (sign) {
          case adrt::Sign::Positive:
            return Ref{data_7x7, func == FunctionType::fht2ds ? ref_7x7_pos_ds
                                                              : ref_7x7_pos_dt};
          case adrt::Sign::Negative:
            return Ref{data_7x7, func == FunctionType::fht2ds ? ref_7x7_neg_ds
                                                              : ref_7x7_neg_dt};
        }

      default:
        assert(0);
        return {nullptr, nullptr};
    }
  }
};

static void unswap_tensor(adrt::Tensor2DTyped<float> const &dst,
                          adrt::Tensor2DTyped<float> const &src,
                          int const swaps[]) {
  size_t const size{src.width * sizeof(float)};
  for (size_t idx{}; idx != static_cast<size_t>(src.height); ++idx) {
    std::memcpy(adrt::A_LINE(dst, idx), adrt::A_LINE(src, swaps[idx]), size);
  }
}

static std::vector<ADRTTestCase> GenerateTestFHT2DSCases() {
  using SignPair = std::pair<adrt::Sign, std::string>;
  std::array<SignPair, 2> const sign_pairs{
      SignPair(adrt::Sign::Positive, "Positive"),
      SignPair(adrt::Sign::Negative, "Negative")};

  using FunctionPair =
      std::tuple<ADRTTestFunction, std::string, FunctionType, IsInplace>;
  std::array<FunctionPair, 8> const function_pairs{
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            auto ids_recursive = adrt::ids_recursive<float>::create(src);
            ids_recursive(src, sign);
            unswap_tensor(dst, src, ids_recursive.swaps.get());
          },
          "fht2ids_recursive", FunctionType::fht2ds, IsInplace::Yes),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            auto ids_non_recursive =
                adrt::ids_non_recursive<float>::create(src);
            ids_non_recursive(src, sign);
            unswap_tensor(dst, src, ids_non_recursive.swaps.get());
          },
          "fht2ids_non_recursive", FunctionType::fht2ds, IsInplace::Yes),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            auto idt_recursive = adrt::idt_recursive<float>::create(src);
            idt_recursive(src, sign);
            unswap_tensor(dst, src, idt_recursive.swaps.get());
          },
          "fht2idt_recursive", FunctionType::fht2dt, IsInplace::Yes),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            auto idt_non_recursive =
                adrt::idt_non_recursive<float>::create(src);
            idt_non_recursive(src, sign);
            unswap_tensor(dst, src, idt_non_recursive.swaps.get());
          },
          "fht2idt_non_recursive", FunctionType::fht2dt, IsInplace::Yes),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            adrt::d<float>::create(src).ds_recursive(dst, src, sign);
          },
          "fht2ds_recursive", FunctionType::fht2ds, IsInplace::No),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            adrt::d<float>::create(src).ds_non_recursive(dst, src, sign);
          },
          "fht2ds_non_recursive", FunctionType::fht2ds, IsInplace::No),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            adrt::d<float>::create(src).dt_recursive(dst, src, sign);
          },
          "fht2dt_recursive", FunctionType::fht2dt, IsInplace::No),
      FunctionPair(
          [](adrt::Tensor2DTyped<float> const &dst,
             adrt::Tensor2DTyped<float> const &src, adrt::Sign sign) {
            adrt::d<float>::create(src).dt_non_recursive(dst, src, sign);
          },
          "fht2dt_non_recursive", FunctionType::fht2dt, IsInplace::No)};

  using SizePair = std::pair<int, std::string>;
  std::array<SizePair, 5> const size_pairs = {
      SizePair(2, "2x2"), SizePair(3, "3x3"), SizePair(4, "4x4"),
      SizePair(5, "5x5"), SizePair(7, "7x7")};

  std::vector<ADRTTestCase> out;
  for (auto [adrt_function, adrt_function_str, func_type, is_inplace] :
       function_pairs) {
    for (auto [sign, sign_str] : sign_pairs) {
      for (auto [size, size_str] : size_pairs) {
        Ref const ref{Ref::get(size, sign, func_type)};
        out.emplace_back(adrt_function_str + "_" + sign_str + "_" + size_str,
                         sign, size, ref.data, ref.ref_data, adrt_function,
                         is_inplace);
      }
    }
  }

  return out;
}

class fht2ids_test: public testing::TestWithParam<ADRTTestCase> {};

TEST_P(fht2ids_test, suite) {
  auto const test_case = GetParam();
  auto const size = test_case.size * test_case.size;
  std::vector<float> data{test_case.data, test_case.data + size};
  std::vector<float> const ref{test_case.ref, test_case.ref + size};
  std::vector<float> data_dst(size, -1.0f);

  decltype(adrt::Tensor2D::height) height{test_case.size};
  decltype(adrt::Tensor2D::width) width{test_case.size};
  adrt::Tensor2D::stride_t const stride{
      static_cast<adrt::Tensor2D::stride_t>(sizeof(float)) *
      static_cast<adrt::Tensor2D::stride_t>(width)};
  adrt::Tensor2D const tensor_src{
      /* height = */ height,
      /* width = */ width,
      /* stride = */ stride,
      /* data = */ reinterpret_cast<uint8_t *>(data.data())};
  adrt::Tensor2D const tensor_dst{
      /* height = */ height,
      /* width = */ width,
      /* stride = */ stride,
      /* data = */ reinterpret_cast<uint8_t *>(data_dst.data())};

  test_case.adrt_function(tensor_dst.as<float>(), tensor_src.as<float>(),
                          test_case.sign);
  ASSERT_EQ(ref, data_dst);
  if (test_case.is_inplace == IsInplace::No) {
    // check input wasn't modified
    std::vector<float> const ref_ref{
        test_case.ref, test_case.ref + test_case.size * test_case.size};
    ASSERT_EQ(ref, ref_ref);
  }
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

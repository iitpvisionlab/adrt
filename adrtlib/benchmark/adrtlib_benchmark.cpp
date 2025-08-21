#include <benchmark/benchmark.h>

#include <adrtlib/adrtlib.hpp>
#include <memory>

enum class IsRecursive { Yes, No };

static void BM_fht2ids(benchmark::State &state, IsRecursive is_recursive) {
  int const height = state.range(0);
  int const width = height;
  std::unique_ptr<float[]> src{new float[height * width]{}};
  for (int idx = 0; idx != height * width; ++idx) {
    src.get()[idx] = idx;
  }

  adrt::Tensor2D const tensor{
      height, width,
      static_cast<adrt::Tensor2D::stride_t>(width * sizeof(float)),
      reinterpret_cast<uint8_t *>(src.get())};
  adrt::Sign const sign = adrt::Sign::Positive;

  if (is_recursive == IsRecursive::No) {
    auto ids_non_recursive =
        adrt::ids_non_recursive<float>::create(tensor.as<float>());
    for (auto _ : state) {
      ids_non_recursive(tensor.as<float>(), sign);
    }
  } else {
    auto ids_recursive = adrt::ids_recursive<float>::create(tensor.as<float>());
    for (auto _ : state) {
      ids_recursive(tensor.as<float>(), sign);
    }
  }

  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(height * width * sizeof(float)));
}

static void BM_fht2ds(benchmark::State &state, IsRecursive is_recursive) {
  int const height = state.range(0);
  int const width = height;
  std::unique_ptr<float[]> src_data{new float[height * width]};
  std::unique_ptr<float[]> dst_data{new float[height * width]{}};
  for (int idx = 0; idx != height * width; ++idx) {
    src_data.get()[idx] = idx;
  }
  std::unique_ptr<float[]> line_buffer{new float[width]};
  std::unique_ptr<int[]> swaps{new int[height]};
  std::unique_ptr<int[]> swaps_buffer{new int[height]};

  adrt::Tensor2D const src{
      height, width,
      static_cast<adrt::Tensor2D::stride_t>(width * sizeof(float)),
      reinterpret_cast<uint8_t *>(src_data.get())};
  adrt::Tensor2D const dst{
      height, width,
      static_cast<adrt::Tensor2D::stride_t>(width * sizeof(float)),
      reinterpret_cast<uint8_t *>(dst_data.get())};
  adrt::Sign const sign = adrt::Sign::Positive;

  auto const d_core = adrt::d<float>::create(src.as<float>());

  if (is_recursive == IsRecursive::No) {
    for (auto _ : state) {
      d_core.ds_non_recursive(dst.as<float>(), src.as<float>(), sign);
    }
  } else {
    for (auto _ : state) {
      d_core.ds_recursive(dst.as<float>(), src.as<float>(), sign);
    }
  }

  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(height * width * sizeof(float)));
}

static void BM_fht2idt(benchmark::State &state, IsRecursive is_recursive) {
  int const height = state.range(0);
  int const width = height;
  std::unique_ptr<float[]> src{new float[height * width]{}};
  for (int idx = 0; idx != height * width; ++idx) {
    src.get()[idx] = idx;
  }
  adrt::Tensor2D const tensor{
      height, width,
      static_cast<adrt::Tensor2D::stride_t>(width * sizeof(float)),
      reinterpret_cast<uint8_t *>(src.get())};
  adrt::Sign const sign = adrt::Sign::Positive;

  if (is_recursive == IsRecursive::Yes) {
    auto idt_recursive = adrt::idt_recursive<float>::create(tensor.as<float>());
    for (auto _ : state) {
      idt_recursive(tensor.as<float>(), sign);
    }
  } else {
    auto idt_non_recursive =
        adrt::idt_non_recursive<float>::create(tensor.as<float>());
    for (auto _ : state) {
      idt_non_recursive(tensor.as<float>(), sign);
    }
  }
  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(height * width * sizeof(float)));
}

#define TEST_ARG ->DenseRange(16, 4096, 16)

// Register the function as a benchmark
BENCHMARK_CAPTURE(BM_fht2ids, recursive, IsRecursive::Yes) TEST_ARG;

BENCHMARK_CAPTURE(BM_fht2ids, non_recursive, IsRecursive::No) TEST_ARG;

BENCHMARK_CAPTURE(BM_fht2idt, recursive, IsRecursive::Yes) TEST_ARG;

BENCHMARK_CAPTURE(BM_fht2idt, non_recursive, IsRecursive::No) TEST_ARG;

BENCHMARK_CAPTURE(BM_fht2ds, recursive, IsRecursive::Yes) TEST_ARG;

BENCHMARK_CAPTURE(BM_fht2ds, non_recursive, IsRecursive::No) TEST_ARG;

BENCHMARK_MAIN();

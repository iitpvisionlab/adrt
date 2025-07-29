#include <benchmark/benchmark.h>

#include <adrtlib/adrtlib.hpp>
#include <memory>

static void BM_fht2ids_recursive(benchmark::State &state) {
  int const height = state.range(0);
  int const width = height;
  std::unique_ptr<float[]> src{new float[height * width]{}};
  for (int idx = 0; idx != height * width; ++idx) {
    src.get()[idx] == idx;
  }
  std::unique_ptr<float[]> line_buffer{new float[width]};
  std::unique_ptr<int[]> swaps{new int[height]};
  std::unique_ptr<int[]> swaps_buffer{new int[height]};

  adrt::Tensor2D const tensor{height, width,
                              static_cast<int_fast32_t>(width * sizeof(float)),
                              reinterpret_cast<uint8_t *>(src.get())};
  adrt::Sign const sign = adrt::Sign::Positive;

  for (auto _ : state) {
    memset(swaps.get(), 0, sizeof(int) * width);
    adrt::fht2ids_recursive(tensor, sign, swaps.get(), swaps_buffer.get(),
                            line_buffer.get());
  }

  state.SetBytesProcessed(int64_t(state.iterations()) *
                          int64_t(height * width * sizeof(float)));
}
// Register the function as a benchmark
BENCHMARK(BM_fht2ids_recursive)
    ->Arg(1 << 3)
    ->Arg(1 << 4)
    ->Arg(1 << 5)
    ->Arg(1 << 6)
    ->Arg(1 << 7)
    ->Arg(1 << 8)
    ->Arg(1 << 9)
    ->Arg(1 << 10)
    ->Arg(1 << 11)
    ->Arg(1 << 12);

BENCHMARK_MAIN();

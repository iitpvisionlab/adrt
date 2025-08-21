#include <cmath>  // round
#include <vector>

#include "common_algorithms.hpp"
#include "non_recursive.hpp"

namespace adrt {

template <typename Scalar>
static inline void fht2ids_core(int const h, Sign sign, int K[],
                                int const K_T[], int const K_B[],
                                Scalar buffer[],
                                Tensor2DTyped<Scalar> const &I_T,
                                Tensor2DTyped<Scalar> const &I_B) {
  A_NEVER(h < 2);
  int t_B, t_T, k_T, k_B, t;
  int const width = I_T.width;
  if (h % 2 == 0) {
    for (t = 0; t < h; t += 2) {
      t_B = t_T = t / 2;
      k_T = K_T[t_T];
      k_B = K_B[t_B];
      ProcessPair(A_LINE(I_T, k_T), A_LINE(I_B, k_B), buffer, width, sign,
                  apply_sign(sign, (t - t_B + 1), width));
      K[t] = k_T;
      K[t + 1] = I_T.height + k_B;
    }
  } else {
    int const t_L_3deg_plus_one = round(static_cast<double>(h) / 4.0);
    int const num_t_to_preprocess = h - 2 * t_L_3deg_plus_one;
    t_T = h / 2 - 1;
    t_B = h - h / 2 - 1;
    for (t = h - 1; t != h - 1 - num_t_to_preprocess; --t) {
      k_T = K_T[t_T];
      k_B = K_B[t_B];
      int const shift = apply_sign(sign, t - t_B, width);
      Scalar *l_t = A_LINE(I_T, k_T);
      Scalar *l_b = A_LINE(I_B, k_B);
      if (t % 2 == 0) {
        ProcessLineWithoutSavingT(l_t, l_b, buffer, width, shift);
        K[t] = I_T.height + k_B;
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
      K[t + 1] = I_T.height + k_B;
    }
  }
}

template <typename Scalar>
void _fht2ids_recursive(Tensor2DTyped<Scalar> const &src, Sign sign,
                        int swaps[], int swaps_buffer[], Scalar line_buffer[]) {
  auto const height = src.height;
  if A_UNLIKELY (height <= 1) {
    return;
  }
  std::memset(swaps, 0, height * sizeof(int));
  auto const h_T = height / 2;
  Tensor2D const I_T{slice_no_checks(src, 0, h_T)};
  Tensor2D const I_B{slice_no_checks(src, h_T, src.height)};

  std::memcpy(swaps_buffer, swaps, height * sizeof(swaps_buffer[0]));

  if (I_T.height > 1) {
    _fht2ids_recursive(I_T.as<Scalar>(), sign, swaps, swaps_buffer,
                       line_buffer);
  }
  if (I_B.height > 1) {
    _fht2ids_recursive(I_B.as<Scalar>(), sign, swaps + h_T, swaps_buffer + h_T,
                       line_buffer);
  }
  std::memcpy(swaps_buffer, swaps, height * sizeof(swaps_buffer[0]));
  fht2ids_core(height, sign, swaps, swaps_buffer + 0, swaps_buffer + h_T,
               line_buffer, I_T.as<Scalar>(), I_B.as<Scalar>());
}

template <typename Scalar>
void _fht2ids_non_recursive(Tensor2DTyped<Scalar> const &src, Sign sign,
                            int swaps[], int swaps_buffer[],
                            Scalar line_buffer[],
                            std::vector<ADRTTask> const &tasks) {
  auto const height = src.height;
  if A_UNLIKELY (height <= 1) {
    return;
  }
  std::memset(swaps, 0, height * sizeof(int));

  for (ADRTTask const &task : tasks) {
    A_NEVER(task.size < 2);
    Tensor2D const I_T{slice_no_checks(src, task.start, task.mid)};
    Tensor2D const I_B{slice_no_checks(src, task.mid, task.stop)};
    int *cur_swaps_buffer = swaps_buffer + task.start;
    int *cur_swaps = swaps + task.start;
    std::memcpy(cur_swaps_buffer, cur_swaps,
                task.size * sizeof(swaps_buffer[0]));
    fht2ids_core(task.size, sign, cur_swaps, cur_swaps_buffer,
                 swaps_buffer + task.mid, line_buffer, I_T.as<Scalar>(),
                 I_B.as<Scalar>());
  }
}

template <typename Scalar>
class ids_recursive {
  std::unique_ptr<Scalar[]> line_buffer;
  std::unique_ptr<int[]> swaps_buffer;
  ids_recursive(std::unique_ptr<Scalar[]> &&line_buffer,
                std::unique_ptr<int[]> &&swaps,
                std::unique_ptr<int[]> &&swaps_buffer)
      : line_buffer{std::move(line_buffer)},
        swaps_buffer{std::move(swaps_buffer)},
        swaps{std::move(swaps)} {}

 public:
  std::unique_ptr<int[]> swaps;
  static ids_recursive<Scalar> create(Tensor2DTyped<Scalar> const &prototype) {
    std::unique_ptr<Scalar[]> line_buffer{new Scalar[prototype.width]};
    std::unique_ptr<int[]> swaps{new int[prototype.height]};
    std::unique_ptr<int[]> swaps_buffer{new int[prototype.height]};

    return ids_recursive<Scalar>{std::move(line_buffer),
                                 std::move(swaps_buffer), std::move(swaps)};
  }

  void operator()(Tensor2DTyped<Scalar> const &src, Sign sign) const {
    _fht2ids_recursive(src, sign, this->swaps.get(), this->swaps_buffer.get(),
                       this->line_buffer.get());
  }
};

template <typename Scalar>
class ids_non_recursive {
  std::unique_ptr<Scalar[]> line_buffer;
  std::unique_ptr<int[]> swaps_buffer;
  std::vector<ADRTTask> tasks;
  ids_non_recursive(std::unique_ptr<Scalar[]> &&line_buffer,
                    std::unique_ptr<int[]> &&swaps,
                    std::unique_ptr<int[]> &&swaps_buffer,
                    std::vector<ADRTTask> &&tasks)
      : line_buffer{std::move(line_buffer)},
        swaps_buffer{std::move(swaps_buffer)},
        swaps{std::move(swaps)},
        tasks{std::move(tasks)} {}

 public:
  std::unique_ptr<int[]> swaps;
  static ids_non_recursive<Scalar> create(
      Tensor2DTyped<Scalar> const &prototype) {
    std::unique_ptr<Scalar[]> line_buffer{new Scalar[prototype.width]};
    std::unique_ptr<int[]> swaps{new int[prototype.height]};
    std::unique_ptr<int[]> swaps_buffer{new int[prototype.height]};
    std::vector<ADRTTask> tasks;
    adrt::non_recursive(
        prototype.height,
        [&](ADRTTask const &task) { tasks.emplace_back(task); },
        [](auto val) { return val / 2; });
    return ids_non_recursive<Scalar>{std::move(line_buffer),
                                     std::move(swaps_buffer), std::move(swaps),
                                     std::move(tasks)};
  }

  void operator()(Tensor2DTyped<Scalar> const &src, Sign sign) const {
    _fht2ids_non_recursive(src, sign, this->swaps.get(),
                           this->swaps_buffer.get(), this->line_buffer.get(),
                           this->tasks);
  }
};

template <typename Scalar>
using fht2ids_recursive = ids_recursive<Scalar>;

template <typename Scalar>
using fht2ids_non_recursive = ids_non_recursive<Scalar>;

}  // namespace adrt

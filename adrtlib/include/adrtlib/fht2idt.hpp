#pragma once
#include <algorithm>
#include <numeric>  // std::iota
#include <vector>

#include "common_algorithms.hpp"
#include "non_recursive.hpp"

namespace adrt {

struct OutDegree {
  int32_t v;  // degree
  int32_t a;  // start
  int32_t b;  // end

  OutDegree() : v{0}, a{0x7FFFFFFF}, b{-0x8000000} {}

  void add(int32_t t) {
    A_NEVER(t < 0);
    this->v += 1;
    this->a = std::min(this->a, t);
    this->b = std::max(this->b, t);
  }

  void sub() {
    A_NEVER(this->v <= 0);
    this->v -= 1;
  }
};

// v_T and v_B must be reset before this point
static inline void outteg(OutDegree* v_T, OutDegree* v_B, int height, int h_T,
                          double k_T, double k_B) {
  for (int32_t t{}; t != height; ++t) {
    int32_t t_T = round05(k_T * t);
    int32_t t_B = round05(k_B * t);
    v_T[t_T].add(t);
    v_B[t_B].add(t);
  }
}

template <typename Scalar>
static inline void fht2idt_core(
    int const h, Sign sign, int K[], int const K_T[], int const K_B[],
    Scalar buffer[], Tensor2DTyped<Scalar> const& I_T,
    Tensor2DTyped<Scalar> const& I_B, OutDegree* out_degrees,
    std::vector<int>& t_B_to_check, std::vector<int>& t_T_to_check,
    std::vector<bool>& t_processed) {
  A_NEVER(h < 2);
  auto const h_T = I_T.height;
  auto const h_B = I_B.height;
  auto const width = I_B.width;
  t_T_to_check.resize(h_T);
  std::iota(t_T_to_check.begin(), t_T_to_check.end(), 0);
  t_processed.resize(h);
  fill(t_processed.begin(), t_processed.end(), false);
  int32_t t_B_prev = -1;
  double const k_T = static_cast<double>(h_T - 1) / static_cast<double>(h - 1);
  double const k_B = static_cast<double>(h_B - 1) / static_cast<double>(h - 1);

  OutDegree* v_T = out_degrees + 0;
  OutDegree* v_B = out_degrees + h_T;
  for (int32_t i{}; i != h; ++i) {
    out_degrees[i] = OutDegree();
  }

  outteg(v_T, v_B, h, h_T, k_T, k_B);

  while (!t_T_to_check.empty()) {
    t_B_to_check.clear();
    t_B_prev = -1;

    for (auto const t_T : t_T_to_check) {
      if (v_T[t_T].v == 1) {
        int32_t const start_T = v_T[t_T].a;
        int32_t const end_T = v_T[t_T].b;
        int32_t const t = !t_processed[end_T] ? end_T : start_T;
        int32_t const t_B = round05(k_B * t);
        int const k_T = K_T[t_T];
        int const k_B = K_B[t_B];
        int const shift = apply_sign(sign, t - t_B, width);
        ProcessLineAndSaveT(A_LINE(I_T, k_T), A_LINE(I_B, k_B), width, shift);
        K[t] = k_T;
        t_processed[t] = true;

        if (t_B != t_B_prev) {
          t_B_to_check.emplace_back(t_B);
          t_B_prev = t_B;
        }
        v_T[t_T].sub();
        v_B[t_B].sub();
      }
    }
    if (t_B_to_check.empty()) {
      break;
    }
    t_T_to_check.clear();

    for (auto const t_B : t_B_to_check) {
      if (v_B[t_B].v == 1) {
        int32_t const start_B = v_B[t_B].a;
        int32_t const end_B = v_B[t_B].b;
        int32_t const t = !t_processed[start_B] ? start_B : end_B;
        int32_t const t_T = round05(k_T * t);
        int const k_T = K_T[t_T];
        int const k_B = K_B[t_B];
        int const shift = apply_sign(sign, t - t_B, width);
        ProcessLineWithoutSavingT(A_LINE(I_T, k_T), A_LINE(I_B, k_B), buffer,
                                  width, shift);
        K[t] = h_T + k_B;

        t_processed[t] = true;

        t_T_to_check.emplace_back(t_T);
        v_T[t_T].sub();
        v_B[t_B].sub();
      }
    }
  }

  int32_t t{};
  for (auto const is_processed : t_processed) {
    if (!is_processed) {
      int32_t const t_T = round05(k_T * t);
      int32_t const t_B = round05(k_B * t);

      int32_t const k_T = K_T[t_T];
      int32_t const k_B = K_B[t_B];
      ProcessPair(A_LINE(I_T, k_T), A_LINE(I_B, k_B), buffer, width, sign,
                  apply_sign(sign, (t - t_B + 1), width));

      K[t] = k_T;
      K[t + 1] = h_T + k_B;
      t_processed[t] = true;
      t_processed[t + 1] = true;
    }
    ++t;
  }
}

template <typename Scalar>
void fht2idt_recursive(Tensor2DTyped<Scalar> const& src, Sign sign, int swaps[],
                       int swaps_buffer[], Scalar line_buffer[],
                       OutDegree out_degrees[], std::vector<int>& t_B_to_check,
                       std::vector<int>& t_T_to_check,
                       std::vector<bool>& t_processed) {
  auto const height = src.height;
  if A_UNLIKELY (height <= 1) {
    return;
  }
  auto const h_T = div_by_pow2(height);
  Tensor2D const I_T{slice_no_checks(src, 0, h_T)};
  Tensor2D const I_B{slice_no_checks(src, h_T, src.height)};

  if (I_T.height > 1) {
    fht2idt_recursive(I_T.as<Scalar>(), sign, swaps, swaps_buffer, line_buffer,
                      out_degrees, t_B_to_check, t_T_to_check, t_processed);
  }
  if (I_B.height > 1) {
    fht2idt_recursive(I_B.as<Scalar>(), sign, swaps + h_T, swaps_buffer + h_T,
                      line_buffer, out_degrees, t_B_to_check, t_T_to_check,
                      t_processed);
  }
  memcpy(swaps_buffer, swaps, height * sizeof(swaps_buffer[0]));
  fht2idt_core(height, sign, swaps, swaps_buffer + 0, swaps_buffer + h_T,
               line_buffer, I_T.as<Scalar>(), I_B.as<Scalar>(), out_degrees,
               t_B_to_check, t_T_to_check, t_processed

  );
}

template <typename Scalar>
void fht2idt_non_recursive(Tensor2DTyped<Scalar> const& src, Sign sign,
                           int swaps[], int swaps_buffer[],
                           Scalar line_buffer[], OutDegree out_degrees[],
                           std::vector<int>& t_B_to_check,
                           std::vector<int>& t_T_to_check,
                           std::vector<bool>& t_processed) {
  auto const height = src.height;
  if A_UNLIKELY (height <= 1) {
    return;
  }

  non_recursive(
      height,
      [&](ADRTTask const& task) {
        A_NEVER(task.size < 2);
        Tensor2D const I_T{slice_no_checks(src, task.start, task.mid)};
        Tensor2D const I_B{slice_no_checks(src, task.mid, task.stop)};
        int* cur_swaps_buffer = swaps_buffer + task.start;
        int* cur_swaps = swaps + task.start;
        memcpy(cur_swaps_buffer, cur_swaps,
               task.size * sizeof(swaps_buffer[0]));
        fht2idt_core(task.size, sign, cur_swaps, cur_swaps_buffer,
                     swaps_buffer + task.mid, line_buffer, I_T.as<Scalar>(),
                     I_B.as<Scalar>(), out_degrees, t_B_to_check, t_T_to_check,
                     t_processed);
      },
      [](int val) {
        return static_cast<int>(div_by_pow2(static_cast<uint32_t>(val)));
      });
}

template <typename Scalar>
class idt {
  std::unique_ptr<int[]> swaps_buffer;
  std::unique_ptr<Scalar[]> line_buffer;
  std::unique_ptr<OutDegree[]> out_degrees;
  std::vector<int> t_B_to_check;
  std::vector<int> t_T_to_check;
  std::vector<bool> t_processed;

 public:
  std::unique_ptr<int[]> swaps;
  idt(std::unique_ptr<int[]>&& swaps, std::unique_ptr<int[]>&& swaps_buffer,
      std::unique_ptr<Scalar[]>&& line_buffer,
      std::unique_ptr<OutDegree[]>&& out_degrees)
      : swaps_buffer{std::move(swaps_buffer)},
        line_buffer{std::move(line_buffer)},
        out_degrees{std::move(out_degrees)},
        swaps{std::move(swaps)} {}
  static idt<Scalar> create(Tensor2DTyped<Scalar> const& prototype) {
    std::unique_ptr<int[]> swaps(new int[prototype.height]);
    std::unique_ptr<int[]> swaps_buffer(new int[prototype.height]);
    std::unique_ptr<Scalar[]> line_buffer(new Scalar[prototype.height]);
    std::unique_ptr<adrt::OutDegree[]> out_degrees(
        new adrt::OutDegree[prototype.height]);
    return idt(std::move(swaps), std::move(swaps_buffer),
               std::move(line_buffer), std::move(out_degrees));
  }
  void recursive(Tensor2DTyped<Scalar> const& src, Sign sign) {
    std::fill(this->swaps.get(), this->swaps.get() + src.height, 0);
    fht2idt_recursive(src, sign, this->swaps.get(), this->swaps_buffer.get(),
                      this->line_buffer.get(), this->out_degrees.get(),
                      this->t_B_to_check, this->t_T_to_check,
                      this->t_processed);
  }

  void non_recursive(Tensor2DTyped<Scalar> const& src, Sign sign) {
    std::fill(this->swaps.get(), this->swaps.get() + src.height, 0);
    fht2idt_non_recursive(src, sign, this->swaps.get(),
                          this->swaps_buffer.get(), this->line_buffer.get(),
                          this->out_degrees.get(), this->t_B_to_check,
                          this->t_T_to_check, this->t_processed);
  }
};

template <typename Scalar>
using fht2idt = idt<Scalar>;

}  // namespace adrt
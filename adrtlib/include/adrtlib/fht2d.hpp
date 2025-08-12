#include <cmath>   // round
#include <memory>  // std::unique_ptr

#include "common_algorithms.hpp"
#include "non_recursive.hpp"

namespace adrt {

struct Slice {
  uint_fast32_t begin;
  uint_fast32_t end;
  Slice(uint_fast32_t begin, uint_fast32_t end) : begin{begin}, end{end} {
    A_NEVER(begin >= end);
  }
  Slice top(uint_fast32_t mid) const {
    A_NEVER(this->begin + mid >= this->end);
    return Slice{this->begin, this->begin + mid};
  }
  Slice bottom(uint_fast32_t mid) const {
    A_NEVER(this->begin + mid >= this->end);
    return Slice{this->begin + mid, this->end};
  }
  uint_fast32_t height() const { return this->end - this->begin; }
};

template <typename Scalar>
static inline void fht2ds_core(Tensor2DTyped<Scalar> const &dst,
                               Tensor2DTyped<Scalar> const &src, int const h,
                               Sign sign, Slice const &slice_T,
                               Slice const &slice_B) {
  A_NEVER(h < 2);
  int const width = src.width;
  double const h_double = static_cast<double>(h);
  double const r0 =
      (static_cast<double>(slice_T.height()) - 1.0) / (h_double - 1.0);
  double const r1 =
      (static_cast<double>(slice_B.height()) - 1.0) / (h_double - 1.0);

  for (int t = 0; t != h; ++t) {
    double const t0 = round05(t * r0);
    double const t1 = round05(t * r1);
    int const shift = apply_sign(sign, t - t1, width);
    add_with_2nd_shifted(A_LINE(dst, slice_T.begin + t),
                         A_LINE(src, slice_T.begin + t0),
                         A_LINE(src, slice_B.begin + t1), width, shift);
  }
}

template <typename Scalar, typename MidCallback>
void fht2ds_recursive_(Tensor2DTyped<Scalar> const &dst,
                       Tensor2DTyped<Scalar> const &src, Slice const &slice,
                       Sign sign, int level, MidCallback mid_callback) {
  auto const height = slice.height();
  A_NEVER(height < 1);
  if A_UNLIKELY (height <= 1) {
    return;
  }
  auto const h_T = mid_callback(height);
  Slice const slice_T{slice.top(h_T)};
  Slice const slice_B{slice.bottom(h_T)};

  fht2ds_recursive_(dst, src, slice_T, sign, level + 1, mid_callback);
  fht2ds_recursive_(dst, src, slice_B, sign, level + 1, mid_callback);
  if ((level & 1) == 0) {
    fht2ds_core(dst, src, height, sign, slice_T, slice_B);
  } else {
    fht2ds_core(src, dst, height, sign, slice_T, slice_B);
  }
}

template <typename Scalar, typename MidCallback>
void fht2d_recursive(Tensor2DTyped<Scalar> const &dst,
                     Tensor2DTyped<Scalar> const &src,
                     Tensor2DTyped<Scalar> const &buffer, Sign sign,
                     MidCallback mid_callback) {
  copy_tensor(buffer, src, sizeof(Scalar));
  copy_tensor(dst, src, sizeof(Scalar));

  fht2ds_recursive_(dst, buffer,
                    Slice{0, static_cast<uint_fast32_t>(src.height)}, sign, 0,
                    mid_callback);
}

template <typename Scalar, typename MidCallback>
static inline void fht2d_non_recursive(Tensor2DTyped<Scalar> const &dst,
                                       Tensor2DTyped<Scalar> const &src,
                                       Tensor2DTyped<Scalar> const &buffer,
                                       Sign sign, MidCallback mid_callback) {
  auto const height = src.height;
  if A_UNLIKELY (height < 1) {
    return;
  }

  copy_tensor(buffer, src, sizeof(Scalar));
  copy_tensor(dst, src, sizeof(Scalar));

  non_recursive(
      height,
      [&](ADRTTask const &task, int level) {
        A_NEVER(task.size < 2);
        Slice const slice_T{static_cast<uint_fast32_t>(task.start),
                            static_cast<uint_fast32_t>(task.mid)};
        Slice const slice_B{static_cast<uint_fast32_t>(task.mid),
                            static_cast<uint_fast32_t>(task.stop)};

        uint_fast32_t const height = static_cast<uint_fast32_t>(task.size);
        if ((level & 1) == 0) {
          fht2ds_core<Scalar>(dst, buffer, height, sign, slice_T, slice_B);
        } else {
          fht2ds_core<Scalar>(buffer, dst, height, sign, slice_T, slice_B);
        }
      },
      mid_callback);
}

template <typename Scalar>
class d {
  Tensor2DTyped<Scalar> buffer;
  std::unique_ptr<uint8_t[]> buffer_data;

 public:
  d(Tensor2DTyped<Scalar> &&buffer, std::unique_ptr<uint8_t[]> &&buffer_data)
      : buffer(std::move(buffer)), buffer_data{std::move(buffer_data)} {}

  static d<Scalar> create(Tensor2DTyped<Scalar> const &prototype) {
    std::unique_ptr<uint8_t[]> buffer_data{
        new uint8_t[prototype.height * prototype.width * sizeof(Scalar)]};
    Tensor2DTyped<Scalar> buffer = prototype;
    buffer.data = buffer_data.get();
    return d{std::move(buffer), std::move(buffer_data)};
  }

  void ds_recursive(Tensor2DTyped<Scalar> const &dst,
                    Tensor2DTyped<Scalar> const &src, Sign sign) const {
    fht2d_recursive(dst, src, this->buffer, sign,
                    [](auto val) { return val / 2; });
  }

  void dt_recursive(Tensor2DTyped<Scalar> const &dst,
                    Tensor2DTyped<Scalar> const &src, Sign sign) const {
    fht2d_recursive(dst, src, this->buffer, sign, [](auto val) {
      return static_cast<int>(div_by_pow2(static_cast<uint32_t>(val)));
    });
  }

  void ds_non_recursive(Tensor2DTyped<Scalar> const &dst,
                        Tensor2DTyped<Scalar> const &src, Sign sign) const {
    fht2d_non_recursive(dst, src, this->buffer, sign,
                        [](auto val) { return val / 2; });
  }

  void dt_non_recursive(Tensor2DTyped<Scalar> const &dst,
                        Tensor2DTyped<Scalar> const &src, Sign sign) const {
    fht2d_non_recursive(dst, src, this->buffer, sign, [](auto val) {
      return static_cast<int>(div_by_pow2(static_cast<uint32_t>(val)));
    });
  }
};

template <typename Scalar>
using fht2d = d<Scalar>;
}  // namespace adrt

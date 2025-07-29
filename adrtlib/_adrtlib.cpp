#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <adrtlib/adrtlib.hpp>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

using Image2D = nb::ndarray<nb::numpy, nb::ndim<2>, nb::device::cpu>;

static adrt::Sign int_to_sign(int sign) {
  switch (sign) {
    case 1:
      return adrt::Sign::Positive;
    case -1:
      return adrt::Sign::Negative;
    default:
      throw nb::value_error("sign must be 1 of -1");
  }
}

enum class Recursive { Yes, No };

template <typename Scalar>
static auto py_fht2ids_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                             Recursive recursive, std::unique_ptr<int[]> &swaps,
                             std::unique_ptr<int[]> &swaps_buffer) {
  std::unique_ptr<Scalar[]> line_buffer{new Scalar[tensor.width]};
  (recursive == Recursive::Yes
       ? adrt::fht2ids_recursive<Scalar>
       : adrt::fht2ids_non_recursive<
             Scalar>)(adrt::Tensor2DTyped<Scalar>{tensor},
                      static_cast<adrt::Sign>(sign), swaps.get(),
                      swaps_buffer.get(), line_buffer.get());
}

auto py_fht2ids(Image2D &image, adrt::Sign sign, Recursive recursive) {
  auto const height = image.shape(0);
  auto const width = image.shape(1);
  auto const &dtype = image.dtype();
  auto *data = image.data();
  std::unique_ptr<float[]> line_buffer{new float[width]};
  std::unique_ptr<int[]> swaps{new int[height]{}};
  std::unique_ptr<int[]> swaps_buffer{new int[height]};

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * sizeof(float)),
      .data = reinterpret_cast<uint8_t *>(data)};

  if (dtype == nb::dtype<float>()) {
    py_fht2ids_visit<float>(tensor, sign, recursive, swaps, swaps_buffer);
  } else if (dtype == nb::dtype<double>()) {
    py_fht2ids_visit<double>(tensor, sign, recursive, swaps, swaps_buffer);
  } else if (dtype == nb::dtype<int32_t>()) {
    py_fht2ids_visit<int32_t>(tensor, sign, recursive, swaps, swaps_buffer);
  } else if (dtype == nb::dtype<uint32_t>()) {
    py_fht2ids_visit<uint32_t>(tensor, sign, recursive, swaps, swaps_buffer);
  } else if (dtype == nb::dtype<int64_t>()) {
    py_fht2ids_visit<int64_t>(tensor, sign, recursive, swaps, swaps_buffer);
  } else if (dtype == nb::dtype<uint64_t>()) {
    py_fht2ids_visit<uint64_t>(tensor, sign, recursive, swaps, swaps_buffer);
  } else {
    throw nb::type_error("unimplemented type");
  }

  nb::capsule swaps_owner(swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ swaps.release(),
      /* shape = */ {height},
      /* owner = */ swaps_owner);
}

template <typename Scalar>
static auto py_fht2idt_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                             Recursive recursive, std::unique_ptr<int[]> &swaps,
                             std::unique_ptr<int[]> &swaps_buffer,
                             std::vector<int> &t_B_to_check,
                             std::vector<int> &t_T_to_check,
                             std::vector<bool> &t_processed,
                             std::unique_ptr<adrt::OutDegree[]> &out_degrees) {
  std::unique_ptr<Scalar[]> line_buffer{new Scalar[tensor.width]};
  (recursive == Recursive::Yes
       ? adrt::fht2idt_recursive<Scalar>
       : adrt::fht2idt_non_recursive<
             Scalar>)(adrt::Tensor2DTyped<Scalar>{tensor}, sign, swaps.get(),
                      swaps_buffer.get(), line_buffer.get(), out_degrees.get(),
                      t_B_to_check, t_T_to_check, t_processed);
}

auto py_fht2idt(Image2D &image, adrt::Sign sign, Recursive recursive) {
  size_t const height = image.shape(0);
  size_t const width = image.shape(1);
  auto const dtype = image.dtype();
  auto const itemsize = image.itemsize();
  auto *data = image.data();
  std::unique_ptr<int[]> swaps{new int[height]{}};
  std::unique_ptr<int[]> swaps_buffer{new int[height]};

  std::vector<int> t_B_to_check;
  std::vector<int> t_T_to_check;
  std::vector<bool> t_processed;
  std::unique_ptr<adrt::OutDegree[]> out_degrees(new adrt::OutDegree[height]);

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * itemsize),
      .data = reinterpret_cast<uint8_t *>(data)};

  if (dtype == nb::dtype<float>()) {
    py_fht2idt_visit<float>(tensor, sign, recursive, swaps, swaps_buffer,
                            t_B_to_check, t_T_to_check, t_processed,
                            out_degrees);
  } else if (dtype == nb::dtype<double>()) {
    py_fht2idt_visit<double>(tensor, sign, recursive, swaps, swaps_buffer,
                             t_B_to_check, t_T_to_check, t_processed,
                             out_degrees);
  } else if (dtype == nb::dtype<int32_t>()) {
    py_fht2idt_visit<int32_t>(tensor, sign, recursive, swaps, swaps_buffer,
                              t_B_to_check, t_T_to_check, t_processed,
                              out_degrees);
  } else if (dtype == nb::dtype<uint32_t>()) {
    py_fht2idt_visit<uint32_t>(tensor, sign, recursive, swaps, swaps_buffer,
                               t_B_to_check, t_T_to_check, t_processed,
                               out_degrees);
  } else if (dtype == nb::dtype<int64_t>()) {
    py_fht2idt_visit<int64_t>(tensor, sign, recursive, swaps, swaps_buffer,
                              t_B_to_check, t_T_to_check, t_processed,
                              out_degrees);
  } else if (dtype == nb::dtype<uint64_t>()) {
    py_fht2idt_visit<uint64_t>(tensor, sign, recursive, swaps, swaps_buffer,
                               t_B_to_check, t_T_to_check, t_processed,
                               out_degrees);
  } else {
    throw nb::type_error("unimplemented type");
  }

  nb::capsule swaps_owner(swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ swaps.release(),
      /* shape = */ {height},
      /* owner = */ swaps_owner);
}

NB_MODULE(_adrtlib, m) {
  m.def(
      "fht2ids_recursive",
      [](Image2D &image, int sign) {
        return py_fht2ids(image, int_to_sign(sign), Recursive::Yes);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2ids_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2ids(image, int_to_sign(sign), Recursive::No);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2idt_recursive",
      [](Image2D &image, int sign) {
        return py_fht2idt(image, int_to_sign(sign), Recursive::Yes);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2idt_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2idt(image, int_to_sign(sign), Recursive::No);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "round05",
      [](double value) {
        if (value < 0.0) {
          throw nb::value_error("negative values not supported");
        }
        return adrt::round05(value);
      },
      nb::arg("value"));
  nb::enum_<adrt::Sign>(m, "Sign", nb::is_arithmetic())
      .value("Positive", adrt::Sign::Positive)
      .value("Negative", adrt::Sign::Negative)
      .export_values();
};

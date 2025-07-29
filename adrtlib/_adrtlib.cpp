#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <adrtlib/adrtlib.hpp>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

using Image2D = nb::ndarray<nb::numpy, float, nb::ndim<2>, nb::device::cpu>;

template <typename Function>
auto py_fht2ids(Image2D &image, int sign, Function apply) {
  if (!(sign == 1 || sign == -1)) {
    throw nb::value_error("sign must be 1 of -1");
  }
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

  apply(&tensor, static_cast<adrt::Sign>(sign), swaps.get(), swaps_buffer.get(),
        line_buffer.get());

  nb::capsule swaps_owner(swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ swaps.release(),
      /* shape = */ {height},
      /* owner = */ swaps_owner);
}

template <typename Function>
auto py_fht2idt(Image2D &image, int sign, Function apply) {
  if (!(sign == 1 || sign == -1)) {
    throw nb::value_error("sign must be 1 of -1");
  }
  auto const height = image.shape(0);
  auto const width = image.shape(1);
  auto const &dtype = image.dtype();
  auto *data = image.data();
  std::unique_ptr<float[]> line_buffer{new float[width]};
  std::unique_ptr<int[]> swaps{new int[height]{}};
  std::unique_ptr<int[]> swaps_buffer{new int[height]};

  std::vector<int> t_B_to_check;
  std::vector<int> t_T_to_check;
  std::vector<bool> t_processed;
  std::unique_ptr<adrt::OutDegree[]> out_degrees(new adrt::OutDegree[height]);

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * sizeof(float)),
      .data = reinterpret_cast<uint8_t *>(data)};

  apply(&tensor, static_cast<adrt::Sign>(sign), swaps.get(), swaps_buffer.get(),
        line_buffer.get(), out_degrees.get(), t_B_to_check, t_T_to_check,
        t_processed);

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
        return py_fht2ids(image, sign, adrt::fht2ids_recursive);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2ids_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2ids(image, sign, adrt::fht2ids_non_recursive);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2idt_recursive",
      [](Image2D &image, int sign) {
        return py_fht2idt(image, sign, adrt::fht2idt_recursive);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2idt_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2idt(image, sign, adrt::fht2idt_non_recursive);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "round05",
      [](double value) {
        if (value < 0.0) {
          throw nb::value_error("negative values not supported");
        }
        if (value > 16777215.0) {
          throw nb::value_error(
              "values larger that 16777215 are not supported");
        }
        return adrt::round05(value);
      },
      nb::arg("value"));
};

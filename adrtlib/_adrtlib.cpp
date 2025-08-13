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
enum class Algorithm { DS, DT };

template <typename Scalar>
static auto py_fht2ids_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                             Recursive recursive) {
  auto ids_core = adrt::ids<Scalar>::create(tensor.as<Scalar>());
  if (recursive == Recursive::Yes) {
    ids_core.recursive(tensor.as<Scalar>(), sign);
  } else {
    ids_core.non_recursive(tensor.as<Scalar>(), sign);
  }
  nb::capsule swaps_owner(ids_core.swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ ids_core.swaps.release(),
      /* shape = */ {static_cast<size_t>(tensor.height)},
      /* owner = */ swaps_owner);
}

auto py_fht2ids(Image2D &image, adrt::Sign sign, Recursive recursive) {
  auto const height = image.shape(0);
  auto const width = image.shape(1);
  auto const &dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * itemsize),
      .data = reinterpret_cast<uint8_t *>(image.data())};

  if (dtype == nb::dtype<float>()) {
    return py_fht2ids_visit<float>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<double>()) {
    return py_fht2ids_visit<double>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_fht2ids_visit<int32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_fht2ids_visit<uint32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_fht2ids_visit<int64_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_fht2ids_visit<uint64_t>(tensor, sign, recursive);
  } else {
    throw nb::type_error("unimplemented type");
  }
}

template <typename Scalar>
static auto py_fht2idt_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                             Recursive recursive) {
  auto idt_core = adrt::idt<Scalar>::create(tensor.as<Scalar>());
  if (recursive == Recursive::Yes) {
    idt_core.recursive(tensor.as<Scalar>(), sign);
  } else {
    idt_core.non_recursive(tensor.as<Scalar>(), sign);
  }
  nb::capsule swaps_owner(idt_core.swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ idt_core.swaps.release(),
      /* shape = */ {static_cast<size_t>(tensor.height)},
      /* owner = */ swaps_owner);
}

auto py_fht2idt(Image2D &image, adrt::Sign sign, Recursive recursive) {
  size_t const height = image.shape(0);
  size_t const width = image.shape(1);
  auto const dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * itemsize),
      .data = reinterpret_cast<uint8_t *>(image.data())};

  if (dtype == nb::dtype<float>()) {
    return py_fht2idt_visit<float>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<double>()) {
    return py_fht2idt_visit<double>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_fht2idt_visit<int32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_fht2idt_visit<uint32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_fht2idt_visit<int64_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_fht2idt_visit<uint64_t>(tensor, sign, recursive);
  } else {
    throw nb::type_error("unimplemented type");
  }
}

template <typename Scalar>
static auto py_fht2d_visit(adrt::Tensor2D const &src, adrt::Sign sign,
                           Recursive recursive, Algorithm algorithm) {
  size_t const height = static_cast<size_t>(src.height);
  size_t const width = static_cast<size_t>(src.width);
  Scalar *data = new Scalar[height * width];

  // Delete 'data' when the 'owner' capsule expires
  nb::capsule owner(data, [](void *p) noexcept { delete[] (Scalar *)p; });
  adrt::Tensor2D dst = src;
  dst.data = reinterpret_cast<uint8_t *>(data);

  auto idt_core = adrt::d<Scalar>::create(src.as<Scalar>());
  if (algorithm == Algorithm::DS) {
    if (recursive == Recursive::Yes) {
      idt_core.ds_recursive(dst.as<Scalar>(), src.as<Scalar>(), sign);
    } else {
      idt_core.ds_non_recursive(dst.as<Scalar>(), src.as<Scalar>(), sign);
    }
  } else {
    if (recursive == Recursive::Yes) {
      idt_core.dt_recursive(dst.as<Scalar>(), src.as<Scalar>(), sign);
    } else {
      idt_core.dt_non_recursive(dst.as<Scalar>(), src.as<Scalar>(), sign);
    }
  }
  return nb::cast(nb::ndarray<nb::numpy, Scalar, nb::ndim<2>>(
      /* data = */ data,
      /* shape = */ {height, width},
      /* owner = */ owner));
}

auto py_fht2d(Image2D &image, adrt::Sign sign, Recursive recursive,
              Algorithm algorithm) {
  size_t const height = image.shape(0);
  size_t const width = image.shape(1);
  auto const dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor = {
      .height = static_cast<int>(height),
      .width = static_cast<int>(width),
      .stride = static_cast<ssize_t>(width * itemsize),
      .data = reinterpret_cast<uint8_t *>(image.data())};
  if (dtype == nb::dtype<float>()) {
    return py_fht2d_visit<float>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<double>()) {
    return py_fht2d_visit<double>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_fht2d_visit<int32_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_fht2d_visit<uint32_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_fht2d_visit<int64_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_fht2d_visit<uint64_t>(tensor, sign, recursive, algorithm);
  } else {
    throw nb::type_error("unimplemented type");
  }
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
      "fht2ds_recursive",
      [](Image2D &image, int sign) {
        return py_fht2d(image, int_to_sign(sign), Recursive::Yes,
                        Algorithm::DS);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2ds_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2d(image, int_to_sign(sign), Recursive::No, Algorithm::DS);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2dt_recursive",
      [](Image2D &image, int sign) {
        return py_fht2d(image, int_to_sign(sign), Recursive::Yes,
                        Algorithm::DT);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "fht2dt_non_recursive",
      [](Image2D &image, int sign) {
        return py_fht2d(image, int_to_sign(sign), Recursive::No, Algorithm::DT);
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

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
static auto py_ids_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                         Recursive recursive) {
  std::unique_ptr<int[]> swaps;
  if (recursive == Recursive::Yes) {
    auto ids_recursive =
        adrt::ids_recursive<Scalar>::create(tensor.as<Scalar>());
    ids_recursive(tensor.as<Scalar>(), sign);
    swaps = std::move(ids_recursive.swaps);
  } else {
    auto ids_non_recursive =
        adrt::ids_non_recursive<Scalar>::create(tensor.as<Scalar>());
    ids_non_recursive(tensor.as<Scalar>(), sign);
    swaps = std::move(ids_non_recursive.swaps);
  }
  nb::capsule swaps_owner(swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ swaps.release(),
      /* shape = */ {static_cast<size_t>(tensor.height)},
      /* owner = */ swaps_owner);
}

auto py_ids(Image2D &image, adrt::Sign sign, Recursive recursive) {
  auto const height = image.shape(0);
  auto const width = image.shape(1);
  auto const &dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor{
      /*height = */ static_cast<int>(height),
      /*width = */ static_cast<int>(width),
      /*stride = */ static_cast<adrt::Tensor2D::stride_t>(width * itemsize),
      /*data = */ reinterpret_cast<uint8_t *>(image.data())};

  if (dtype == nb::dtype<float>()) {
    return py_ids_visit<float>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<double>()) {
    return py_ids_visit<double>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_ids_visit<int32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_ids_visit<uint32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_ids_visit<int64_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_ids_visit<uint64_t>(tensor, sign, recursive);
  } else {
    throw nb::type_error("unimplemented type");
  }
}

template <typename Scalar>
static auto py_idt_visit(adrt::Tensor2D const &tensor, adrt::Sign sign,
                         Recursive recursive) {
  std::unique_ptr<int[]> swaps;
  if (recursive == Recursive::Yes) {
    auto idt_recursive =
        adrt::idt_recursive<Scalar>::create(tensor.as<Scalar>());
    idt_recursive(tensor.as<Scalar>(), sign);
    swaps = std::move(idt_recursive.swaps);
  } else {
    auto idt_non_recursive =
        adrt::idt_non_recursive<Scalar>::create(tensor.as<Scalar>());
    idt_non_recursive(tensor.as<Scalar>(), sign);
    swaps = std::move(idt_non_recursive.swaps);
  }
  nb::capsule swaps_owner(swaps.get(),
                          [](void *p) noexcept { delete[] (int *)p; });

  return nb::ndarray<nb::numpy, int, nb::ndim<1>, nb::device::cpu>(
      /* data = */ swaps.release(),
      /* shape = */ {static_cast<size_t>(tensor.height)},
      /* owner = */ swaps_owner);
}

auto py_idt(Image2D &image, adrt::Sign sign, Recursive recursive) {
  size_t const height = image.shape(0);
  size_t const width = image.shape(1);
  auto const dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor{
      /* height = */ static_cast<int>(height),
      /* width = */ static_cast<int>(width),
      /* stride = */ static_cast<adrt::Tensor2D::stride_t>(width * itemsize),
      /* data = */ reinterpret_cast<uint8_t *>(image.data())};

  if (dtype == nb::dtype<float>()) {
    return py_idt_visit<float>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<double>()) {
    return py_idt_visit<double>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_idt_visit<int32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_idt_visit<uint32_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_idt_visit<int64_t>(tensor, sign, recursive);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_idt_visit<uint64_t>(tensor, sign, recursive);
  } else {
    throw nb::type_error("unimplemented type");
  }
}

template <typename Scalar>
static auto py_d_visit(adrt::Tensor2D const &src, adrt::Sign sign,
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

auto py_d(Image2D &image, adrt::Sign sign, Recursive recursive,
          Algorithm algorithm) {
  size_t const height = image.shape(0);
  size_t const width = image.shape(1);
  auto const dtype = image.dtype();
  auto const itemsize = image.itemsize();

  adrt::Tensor2D const tensor{
      /* height = */ static_cast<int>(height),
      /* width = */ static_cast<int>(width),
      /* stride = */ static_cast<adrt::Tensor2D::stride_t>(width * itemsize),
      /* data = */ reinterpret_cast<uint8_t *>(image.data())};
  if (dtype == nb::dtype<float>()) {
    return py_d_visit<float>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<double>()) {
    return py_d_visit<double>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<int32_t>()) {
    return py_d_visit<int32_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<uint32_t>()) {
    return py_d_visit<uint32_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<int64_t>()) {
    return py_d_visit<int64_t>(tensor, sign, recursive, algorithm);
  } else if (dtype == nb::dtype<uint64_t>()) {
    return py_d_visit<uint64_t>(tensor, sign, recursive, algorithm);
  } else {
    throw nb::type_error("unimplemented type");
  }
}

NB_MODULE(_adrtlib, m) {
  m.def(
      "ids_recursive",
      [](Image2D &image, int sign) {
        return py_ids(image, int_to_sign(sign), Recursive::Yes);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "ids_non_recursive",
      [](Image2D &image, int sign) {
        return py_ids(image, int_to_sign(sign), Recursive::No);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "idt_recursive",
      [](Image2D &image, int sign) {
        return py_idt(image, int_to_sign(sign), Recursive::Yes);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "idt_non_recursive",
      [](Image2D &image, int sign) {
        return py_idt(image, int_to_sign(sign), Recursive::No);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "ds_recursive",
      [](Image2D &image, int sign) {
        return py_d(image, int_to_sign(sign), Recursive::Yes, Algorithm::DS);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "ds_non_recursive",
      [](Image2D &image, int sign) {
        return py_d(image, int_to_sign(sign), Recursive::No, Algorithm::DS);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "dt_recursive",
      [](Image2D &image, int sign) {
        return py_d(image, int_to_sign(sign), Recursive::Yes, Algorithm::DT);
      },
      nb::arg("image"), nb::arg("sign") = 1);
  m.def(
      "dt_non_recursive",
      [](Image2D &image, int sign) {
        return py_d(image, int_to_sign(sign), Recursive::No, Algorithm::DT);
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

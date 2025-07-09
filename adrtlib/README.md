* 📁 `benchmark` - benchmark for c++ code (and maybe python code)
* 📁 `include/adrtlib` - c++ headers for header only "adrtlib"
* 📁 `ref` - python reference adrt functions
* 📁 `test` - test for c++ code
* 🗎 `_adrtlib.cpp` - python bindings using [`nanobind`](https://github.com/wjakob/nanobind)
* 🗎 `__main__.py` - print include path for CMake
* 🗎 `CMakeLists.txt` - for building python bindings, tests and  benchmark:
    - `cmake -S . -B build -G "Ninja Multi-Config"`
    - `cmake --build build --config Release`
* 🗎 `.clang-format` formatting for all c++ files in this directory

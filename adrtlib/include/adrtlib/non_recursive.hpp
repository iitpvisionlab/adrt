#pragma once
#include <utility>  // std::move

#include "common.hpp"

namespace adrt {
struct ADRTTask {
  int start;
  int stop;
  int size;  // stop - start
  int mid;
  bool left_visited;
  ADRTTask(int size, int start, int stop, int mid)
      : start{start}, stop{stop}, size{size}, mid{mid}, left_visited{false} {
    A_NEVER(stop <= start);
    // A_NEVER(size < 2);
  }
  ADRTTask() {};
  int size_left() { return this->mid - this->start; }
  int size_right() { return this->stop - this->mid; }
};

struct ADRTTaskStack {
  ADRTTask stack[32];
  int size{};
  template <typename MidCallback>
  ADRTTaskStack(int size, MidCallback mid_callback) {
    this->append(ADRTTask(size, 0, size, mid_callback));
  }
  void append(ADRTTask&& task) { this->stack[this->size++] = std::move(task); }
  // true - then no element left
  bool pop() { return --this->size == 0; }
  ADRTTask* top() {
    A_NEVER(this->size < 1 ||
            this->size >=
                static_cast<int>(sizeof(this->stack) / sizeof(this->stack[0])));
    return &this->stack[this->size - 1];
  }
};

template <typename ApplyCallback, typename MidCallback>
[[gnu::always_inline]] inline void non_recursive(int size, ApplyCallback apply,
                                                 MidCallback mid_callback) {
  ADRTTaskStack tasks_stack(size, mid_callback(size));

  ADRTTask* task = tasks_stack.top();
  for (;;) {
    while (!task->left_visited) {
      if (task->size > 2) {
        int const size_left = task->size_left();
        int const mid_left = mid_callback(size_left);
        tasks_stack.append(ADRTTask(size_left, task->start, task->mid,
                                    task->start + mid_left));
        task = tasks_stack.top();
      } else {
        break;
      }
    }
    if (task->size > 1) {
      if constexpr (std::is_invocable_v<ApplyCallback, const ADRTTask&, int>) {
        // apply with level
        apply(static_cast<ADRTTask const&>(*task), tasks_stack.size - 1);
      } else {
        // apply without level
        apply(static_cast<ADRTTask const&>(*task));
      }
    }
    if (tasks_stack.pop()) {
      break;
    }
    task = tasks_stack.top();
    if (!task->left_visited) {
      task->left_visited = true;
      int const size_right = task->size_right();
      if (size_right > 1) {
        int const mid_right = mid_callback(size_right);
        tasks_stack.append(
            ADRTTask(size_right, task->mid, task->stop, task->mid + mid_right));
        task = tasks_stack.top();
      }
    }
  }
}

}  // namespace adrt
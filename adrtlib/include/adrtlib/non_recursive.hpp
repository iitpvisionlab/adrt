#pragma once
#include <utility>  // std::move

#include "common.hpp"

namespace adrt {
struct ADRTTask {
  int start;
  int stop;
  int size;
  bool left_visited;
  int mid;
  template <typename MidCallback>
  ADRTTask(int start, int stop, MidCallback mid_callback)
      : start{start},
        stop{stop},
        left_visited{false},
        size{stop - start},
        mid{start + mid_callback(size)} {
    A_NEVER(stop <= start);
  }
  ADRTTask() {};
  template <typename MidCallback>
  ADRTTask left(MidCallback mid_callback) const {
    return ADRTTask(this->start, this->mid, mid_callback);
  }

  template <typename MidCallback>
  ADRTTask right(MidCallback mid_callback) const {
    return ADRTTask(this->mid, this->stop, mid_callback);
  }
};

struct ADRTTaskStack {
  ADRTTask stack[32];
  int size{};
  template <typename MidCallback>
  ADRTTaskStack(int size, MidCallback mid_callback) {
    this->append(ADRTTask(0, size, mid_callback));
  }
  void append(ADRTTask&& task) { this->stack[this->size++] = std::move(task); }
  // true - then no element left
  bool pop() { return --this->size == 0; }
  ADRTTask* top() {
    A_NEVER(this->size < 1 ||
            this->size >= sizeof(this->stack) / sizeof(this->stack)[0]);
    return &this->stack[this->size - 1];
  }
};

template <typename ApplyCallback, typename MidCallback>
constexpr void non_recursive(int size, ApplyCallback apply,
                             MidCallback mid_callback) {
  ADRTTaskStack tasks_stack(size, mid_callback);

  ADRTTask* task = tasks_stack.top();
  for (;;) {
    while (!task->left_visited) {
      if (task->size > 2) {
        tasks_stack.append(task->left(mid_callback));
        task = tasks_stack.top();
      } else {
        task->left_visited = true;
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
      if (task->size > 2) {
        tasks_stack.append(task->right(mid_callback));
        task = tasks_stack.top();
      }
    }
  }
}

}  // namespace adrt
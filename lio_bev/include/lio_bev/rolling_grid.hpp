#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace lio_bev {

// World-aligned rolling grid backed by a circular array. Moving the center by
// one cell preserves overlap and clears only the newly exposed border.
template <typename Cell>
class RollingGrid {
public:
  RollingGrid(double resolution, int rows, int cols)
    : resolution_(resolution), rows_(rows), cols_(cols),
      cells_(checkedSize(rows, cols))
  {
    if (!std::isfinite(resolution_) || resolution_ <= 0.0) {
      throw std::invalid_argument("rolling grid resolution must be positive");
    }
  }

  bool recenter(double world_x, double world_y)
  {
    if (!validWorld(world_x) || !validWorld(world_y)) {
      return false;
    }
    const std::int64_t new_origin_x = worldCell(world_x) - cols_ / 2;
    const std::int64_t new_origin_y = worldCell(world_y) - rows_ / 2;
    if (!initialized_) {
      origin_x_ = new_origin_x;
      origin_y_ = new_origin_y;
      initialized_ = true;
      reset();
      return true;
    }
    const std::int64_t dx = new_origin_x - origin_x_;
    const std::int64_t dy = new_origin_y - origin_y_;
    if (dx == 0 && dy == 0) {
      return false;
    }
    if (std::llabs(dx) >= cols_ || std::llabs(dy) >= rows_) {
      origin_x_ = new_origin_x;
      origin_y_ = new_origin_y;
      row0_ = 0;
      col0_ = 0;
      reset();
      return true;
    }
    shiftColumns(static_cast<int>(dx));
    shiftRows(static_cast<int>(dy));
    origin_x_ = new_origin_x;
    origin_y_ = new_origin_y;
    return true;
  }

  void reset() { std::fill(cells_.begin(), cells_.end(), Cell{}); }
  int rows() const { return rows_; }
  int cols() const { return cols_; }

  Cell* cellByWorld(double x, double y)
  {
    int row = 0;
    int col = 0;
    return worldToLocal(x, y, row, col) ? cell(row, col) : nullptr;
  }

  const Cell* cellByWorld(double x, double y) const
  {
    int row = 0;
    int col = 0;
    return worldToLocal(x, y, row, col) ? cell(row, col) : nullptr;
  }

  Cell* cell(int row, int col)
  {
    return valid(row, col) ? &cells_[physicalIndex(row, col)] : nullptr;
  }

  const Cell* cell(int row, int col) const
  {
    return valid(row, col) ? &cells_[physicalIndex(row, col)] : nullptr;
  }

  bool localToWorldCenter(int row, int col, double& x, double& y) const
  {
    if (!initialized_ || !valid(row, col)) {
      return false;
    }
    const std::int64_t gx = origin_x_ + col;
    const std::int64_t gy = origin_y_ + (rows_ - 1 - row);
    x = (static_cast<double>(gx) + 0.5) * resolution_;
    y = (static_cast<double>(gy) + 0.5) * resolution_;
    return true;
  }

private:
  double resolution_;
  int rows_;
  int cols_;
  std::vector<Cell> cells_;
  std::int64_t origin_x_ = 0;
  std::int64_t origin_y_ = 0;
  int row0_ = 0;
  int col0_ = 0;
  bool initialized_ = false;

  static std::size_t checkedSize(int rows, int cols)
  {
    if (rows <= 0 || cols <= 0) {
      throw std::invalid_argument("rolling grid dimensions must be positive");
    }
    return static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols);
  }

  bool validWorld(double value) const
  {
    // Keep conversion and differences away from signed integer overflow.
    const double cell = value / resolution_;
    return std::isfinite(cell) &&
      std::abs(cell) < static_cast<double>(std::numeric_limits<std::int64_t>::max() / 4);
  }

  std::int64_t worldCell(double value) const
  {
    return static_cast<std::int64_t>(std::floor(value / resolution_));
  }

  bool valid(int row, int col) const
  {
    return row >= 0 && row < rows_ && col >= 0 && col < cols_;
  }

  bool worldToLocal(double x, double y, int& row, int& col) const
  {
    if (!initialized_ || !validWorld(x) || !validWorld(y)) {
      return false;
    }
    const std::int64_t local_col = worldCell(x) - origin_x_;
    const std::int64_t local_from_south = worldCell(y) - origin_y_;
    if (local_col < 0 || local_col >= cols_ ||
        local_from_south < 0 || local_from_south >= rows_) {
      return false;
    }
    col = static_cast<int>(local_col);
    row = rows_ - 1 - static_cast<int>(local_from_south);
    return true;
  }

  std::size_t physicalIndex(int row, int col) const
  {
    const int physical_row = (row0_ + row) % rows_;
    const int physical_col = (col0_ + col) % cols_;
    return static_cast<std::size_t>(physical_row) * cols_ + physical_col;
  }

  void clearColumn(int col)
  {
    for (int row = 0; row < rows_; ++row) {
      *cell(row, col) = Cell{};
    }
  }

  void clearRow(int row)
  {
    for (int col = 0; col < cols_; ++col) {
      *cell(row, col) = Cell{};
    }
  }

  void shiftColumns(int dx)
  {
    if (dx > 0) {
      col0_ = (col0_ + dx) % cols_;
      for (int col = cols_ - dx; col < cols_; ++col) clearColumn(col);
    } else if (dx < 0) {
      const int amount = -dx;
      col0_ = (col0_ - amount) % cols_;
      if (col0_ < 0) col0_ += cols_;
      for (int col = 0; col < amount; ++col) clearColumn(col);
    }
  }

  void shiftRows(int dy)
  {
    // Logical row zero is north, so increasing world Y shifts the array in
    // the opposite logical-row direction.
    if (dy > 0) {
      row0_ = (row0_ - dy) % rows_;
      if (row0_ < 0) row0_ += rows_;
      for (int row = 0; row < dy; ++row) clearRow(row);
    } else if (dy < 0) {
      const int amount = -dy;
      row0_ = (row0_ + amount) % rows_;
      for (int row = rows_ - amount; row < rows_; ++row) clearRow(row);
    }
  }
};

}  // namespace lio_bev

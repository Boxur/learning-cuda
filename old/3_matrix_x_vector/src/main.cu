#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void multiply_(const double *matrix, const double *vector,
                          double *result, std::size_t width,
                          std::size_t height) {
  int v_index = blockIdx.x * blockDim.x + threadIdx.x;

  if (v_index < width) {
    double sum = 0;
    for (int col = 0; col < width; ++col) {
      sum += vector[col] * matrix[v_index * width + col];
    }
    result[v_index] = sum;
  }
}

class Vector {
  friend class Matrix;

private:
  std::vector<double> values_;

public:
  Vector(std::size_t size) { values_.resize(size); }
  Vector(std::vector<double> values) : values_(values) {}

  double &operator[](std::size_t index) { return values_[index]; }

  std::size_t GetSize() const { return values_.size(); }

  void print() {
    std::for_each(values_.begin(), values_.end(),
                  [](auto &x) { std::cout << x << ", "; });
    std::cout << "\n";
  }
};

class Matrix {
private:
  std::vector<double> values_;
  std::size_t width_;
  std::size_t height_;

public:
  Matrix(std::size_t width, std::size_t height)
      : width_(width), height_(height) {
    values_.resize(width * height);
  }

  double &at(std::size_t row, std::size_t column) {
    assert(row < height_ && column < width_);
    return values_[row * width_ + column];
  }

  Vector operator*(const Vector &vec) {
    assert(vec.values_.size() == width_);
    double *matrix, *vector, *result;
    cudaMalloc(&matrix, width_ * height_ * sizeof(double));
    cudaMalloc(&vector, width_ * sizeof(double));
    cudaMalloc(&result, height_ * sizeof(double));

    cudaMemcpy(matrix, values_.data(), width_ * height_ * sizeof(double),
               cudaMemcpyHostToDevice);
    cudaMemcpy(vector, vec.values_.data(), width_ * sizeof(double),
               cudaMemcpyHostToDevice);
    std::size_t blockSize = 256;
    std::size_t gridSize = (int)std::ceil((float)height_ / blockSize);
    multiply_<<<gridSize, blockSize>>>(matrix, vector, result, width_, height_);

    Vector ret(height_);

    cudaMemcpy(ret.values_.data(), result, height_ * sizeof(double),
               cudaMemcpyDeviceToHost);
    return ret;
  }

  void print() {
    for (int i = 0; i < height_; ++i) {
      for (int j = 0; j < width_; ++j)
        std::cout << this->at(i, j) << ", ";
      std::cout << "\n";
    }
  }
};

int main() {
  Vector v({1, 2, 3, 4, 5});
  Matrix m(5, 3);
  for (int i = 0; i < 3; ++i)
    for (int j = 0; j < 5; ++j)
      m.at(i, j) = i * 5 + j;
  v.print();
  std::cout << "\n";
  m.print();
  std::cout << "\n";
  auto r = m * v;
  r.print();
}

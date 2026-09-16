#include <cassert>
#include <iostream>
#include <vector>

static __global__ void Add_(const double *first, const double *second,
                            double *result, std::size_t width,
                            std::size_t height) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;

  if (i < width && j < height) {
    int index = j * width + i;
    result[index] = first[index] + second[index];
  }
}

template <std::size_t WIDTH, std::size_t HEIGHT> class Matrix {
private:
  std::vector<double> values;

public:
  Matrix() { values.resize(WIDTH * HEIGHT); }

  double &at(std::size_t row, std::size_t column) {
    assert(column < WIDTH && row < HEIGHT);
    return values[row * WIDTH + column];
  }

  Matrix<WIDTH, HEIGHT> operator+(Matrix<WIDTH, HEIGHT> other) {
    double *first, *second, *result;
    int size = WIDTH * HEIGHT * sizeof(double);
    cudaMalloc(&first, size);
    cudaMalloc(&second, size);
    cudaMalloc(&result, size);

    cudaMemcpy(first, values.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(second, other.values.data(), size, cudaMemcpyHostToDevice);

    dim3 blockDim(2048, 2048);
    dim3 gridDim((int)std::ceil((float)WIDTH / blockDim.x),
                 (int)std::ceil((float)HEIGHT / blockDim.y));

    Add_<<<gridDim, blockDim>>>(first, second, result, WIDTH, HEIGHT);

    cudaDeviceSynchronize();
    Matrix<WIDTH, HEIGHT> ret;
    cudaMemcpy(ret.values.data(), result, size, cudaMemcpyDeviceToHost);

    cudaFree(first);
    cudaFree(second);
    cudaFree(result);

    return ret;
  }

  void print() {
    for (int i = 0; i < HEIGHT; ++i) {
      for (int j = 0; j < WIDTH; ++j)
        std::cout << this->at(i, j) << ", ";
      std::cout << "\n";
    }
  }

private:
};

int main() {
  constexpr size_t size = 15000;
  Matrix<size, size> matrix1, matrix2;
  for (int i = 0; i < size; ++i)
    for (int j = 0; j < size; ++j) {
      matrix1.at(i, j) = i * size + j;
      matrix2.at(i, j) = i * size + j + 10;
    }
  auto result = matrix1 + matrix2;
  // result.print();
}

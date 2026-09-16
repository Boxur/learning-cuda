#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>
__global__ void vector_add(const float *A, const float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int n = 10;
  std::vector<float> A(n), B(n), C(n);

  for (int i = 0; i < n; i++) {
    A[i] = i * 0.5;
    B[i] = i + n;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  cudaMemcpy(d_a, A.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B.data(), n * sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = 256;
  int gridsize = (int)std::ceil((float)n / blocksize);
  vector_add<<<gridsize, blocksize>>>(d_a, d_b, d_c, n);

  cudaMemcpy(C.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  std::for_each(C.begin(), C.end(), [](auto x) { std::cout << x << " "; });
  std::cout << "\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

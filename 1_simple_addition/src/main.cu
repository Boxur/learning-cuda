#include <cmath>
#include <iostream>
__global__ void vector_add(const float *A, const float *B, float *C, int n) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  int n = 10;
  float A[n], B[n], C[n];

  for (int i = 0; i < n; i++) {
    A[i] = i * 0.5;
    B[i] = i + n;
    C[i] = 0;
  }

  float *d_a, *d_b, *d_c;
  cudaMalloc(&d_a, n * sizeof(float));
  cudaMalloc(&d_b, n * sizeof(float));
  cudaMalloc(&d_c, n * sizeof(float));

  cudaMemcpy(d_a, A, n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, n * sizeof(float), cudaMemcpyHostToDevice);

  int blocksize = 256;
  int gridsize = (int)std::ceil((float)n / blocksize);
  vector_add<<<gridsize, blocksize>>>(d_a, d_b, d_c, n);

  cudaMemcpy(C, d_c, n * sizeof(float), cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; i++)
    std::cout << C[i] << " ";
  std::cout << "\n";

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
}

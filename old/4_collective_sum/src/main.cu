#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

__global__ void collectiveSumKernel(const int *input, int *output, int size) {
  extern __shared__ int sharedMemory[];
  int thread_index = threadIdx.x;
  int index = blockIdx.x * blockDim.x * 2 + thread_index;
  if (index < size) {
    sharedMemory[index] = input[index] + input[index + blockDim.x];
  }
  __syncthreads();
  for (int s = 1; s < blockDim.x; s *= 2) {
    int temp = 0;
    if (index < size && index >= s)
      temp = sharedMemory[index - s];
    __syncthreads();
    if (index < size)
      sharedMemory[index] += temp;
    __syncthreads();
  }
  if (index < size)
    output[index] = sharedMemory[index];
}

int main() {
  std::vector<int> input{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
  std::vector<int> output(8);
  int *d_in, *d_out;
  cudaMalloc(&d_in, input.size() * sizeof(int));
  cudaMalloc(&d_out, output.size() * sizeof(int));

  cudaMemcpy(d_in, input.data(), input.size() * sizeof(int),
             cudaMemcpyHostToDevice);
  int blockSize = 8;
  int gridSize = (int)std::ceil((float)input.size() / blockSize);
  collectiveSumKernel<<<gridSize, blockSize, input.size() * sizeof(int)>>>(
      d_in, d_out, input.size());
  cudaMemcpy(output.data(), d_out, output.size() * sizeof(int),
             cudaMemcpyDeviceToHost);
  std::for_each(output.begin(), output.end(),
                [](auto x) { std::cout << x << ", "; });
  std::cout << "\n";

  cudaFree(d_in);
  cudaFree(d_out);
}

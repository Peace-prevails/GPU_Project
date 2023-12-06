#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel function for exclusive scan
void reduce_cpu(float *h_data, float *h_result, int size);
void reduce_single_gpu(float *h_data, float *h_result, int size);
void reduce_dual_gpu(float *h_data, float *h_result, int size);

void reduce_cpu(float *h_data, float *h_result, int size)
{
  float sum = 0.0f;
  for (int i = 0; i < size; i++)
  {
    sum += h_data[i];
  }
  *h_result = sum;
}

__global__ void reduce_kernel(float *d_out, float *d_in, int size)
{
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  // Load shared mem from global mem
  sdata[tid] = (i < size) ? d_in[i] : 0;
  __syncthreads();

  // Perform reduction in shared mem
  for (int s = blockDim.x / 2; s > 0; s >>= 1)
  {
    if (tid < s)
    {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Write result for this block to global mem
  if (tid == 0)
    d_out[blockIdx.x] = sdata[0];
}

void reduce_single_gpu(float *h_data, float *h_result, int size)
{
  float *d_in, *d_out;
  cudaMalloc(&d_in, size * sizeof(float));
  cudaMalloc(&d_out, size * sizeof(float));

  cudaMemcpy(d_in, h_data, size * sizeof(float), cudaMemcpyHostToDevice);

  int blockSize = 1024; // Adjust as needed
  int gridSize = (size + blockSize - 1) / blockSize;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_out, d_in, size);

  // Copy partial results back and finish reduction on CPU
  float *partial_sums = (float *)malloc(gridSize * sizeof(float));

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Single GPU execution time: %f ms\n", milliseconds);
  cudaMemcpy(partial_sums, d_out, gridSize * sizeof(float), cudaMemcpyDeviceToHost);

  float final_sum = 0;
  for (int i = 0; i < gridSize; i++)
  {
    final_sum += partial_sums[i];
  }
  *h_result = final_sum;

  free(partial_sums);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  cudaFree(d_in);
  cudaFree(d_out);
}
// __global__ void exclusive_scan_kernel(float *d_out, float *d_in, int size) {
//     extern __shared__ float temp[];
//     int thid = threadIdx.x;
//     int offset = 1;

//     temp[2 * thid] = (thid > 0) ? d_in[thid - 1] : 0;
//     temp[2 * thid + 1] = d_in[thid];

//     for (int d = size >> 1; d > 0; d >>= 1) {
//         __syncthreads();
//         if (thid < d) {
//             int ai = offset * (2 * thid + 1) - 1;
//             int bi = offset * (2 * thid + 2) - 1;
//             temp[bi] += temp[ai];
//         }
//         offset *= 2;
//     }

//     if (thid == 0) temp[size - 1] = 0;

//     for (int d = 1; d < size; d *= 2) {
//         offset >>= 1;
//         __syncthreads();
//         if (thid < d) {
//             int ai = offset * (2 * thid + 1) - 1;
//             int bi = offset * (2 * thid + 2) - 1;
//             float t = temp[ai];
//             temp[ai] = temp[bi];
//             temp[bi] += t;
//         }
//     }
//     __syncthreads();

//     d_out[thid] = temp[2 * thid];
//     d_out[thid + 1] = temp[2 * thid + 1];
// }

// void run_scan_single_gpu(float *h_data, float *h_result, int size) {
//     size_t bytes = size * sizeof(float);
//     float *d_in, *d_out;
//     cudaMalloc(&d_in, bytes);
//     cudaMalloc(&d_out, bytes);

//     cudaMemcpy(d_in, h_data, bytes, cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);
//     exclusive_scan_kernel<<<1, size / 2, size * sizeof(float)>>>(d_out, d_in, size);
//     cudaEventRecord(stop);

//     cudaMemcpy(h_result, d_out, bytes, cudaMemcpyDeviceToHost);

//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Single GPU time: %f ms\n", milliseconds);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     cudaFree(d_in);
//     cudaFree(d_out);
// }

// void run_scan_multi_gpu(float *h_data, float *h_result, int size) {
//     int halfSize = size / 2;
//     size_t halfBytes = halfSize * sizeof(float);

//     float *d_data0, *d_out0, *d_data1, *d_out1;

//     cudaSetDevice(0);
//     cudaMalloc(&d_data0, halfBytes);
//     cudaMalloc(&d_out0, halfBytes);

//     cudaSetDevice(1);
//     cudaMalloc(&d_data1, halfBytes);
//     cudaMalloc(&d_out1, halfBytes);

//     cudaMemcpy(d_data0, h_data, halfBytes, cudaMemcpyHostToDevice);
//     cudaMemcpy(d_data1, &h_data[halfSize], halfBytes, cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     cudaEventRecord(start);

//     cudaSetDevice(0);
//     exclusive_scan_kernel<<<1, halfSize / 2, halfSize * sizeof(float)>>>(d_out0, d_data0, halfSize);

//     cudaSetDevice(1);
//     exclusive_scan_kernel<<<1, halfSize / 2, halfSize * sizeof(float)>>>(d_out1, d_data1, halfSize);

//     cudaSetDevice(0);
//     cudaDeviceSynchronize();

//     cudaSetDevice(1);
//     cudaDeviceSynchronize();

//     cudaEventRecord(stop);

//     cudaMemcpy(&h_result[0], d_out0, halfBytes, cudaMemcpyDeviceToHost);
//     cudaMemcpy(&h_result[halfSize], d_out1, halfBytes, cudaMemcpyDeviceToHost);

//     float offset = h_result[halfSize - 1];
//     for (int i = halfSize; i < size; i++) {
//         h_result[i] += offset;
//     }

//     cudaEventSynchronize(stop);
//     float milliseconds = 0;
//     cudaEventElapsedTime(&milliseconds, start, stop);
//     printf("Multi GPU time: %f ms\n", milliseconds);

//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);
//     cudaFree(d_data0);
//     cudaFree(d_out0);
//     cudaFree(d_data1);
//     cudaFree(d_out1);
// }
void cudaCheckErrors(const char *msg)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess)
  {
    fprintf(stderr, "Error: %s: %s\n", msg, cudaGetErrorString(err));
    exit(1);
  }
}
void reduce_dual_gpu(float *h_data, float *h_result, int size)
{
  int halfSize = (size + 1) / 2; // Adjust for odd sizes
  float *d_data0, *d_out0, *d_data1, *d_out1;

  // Allocate memory and initialize data on both GPUs
  cudaSetDevice(0);
  cudaMalloc(&d_data0, halfSize * sizeof(float));
  cudaMalloc(&d_out0, halfSize * sizeof(float));
  cudaMemcpy(d_data0, h_data, halfSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMalloc and cudaMemcpy on GPU 0 failed");

  cudaSetDevice(1);
  cudaMalloc(&d_data1, halfSize * sizeof(float));
  cudaMalloc(&d_out1, halfSize * sizeof(float));
  cudaMemcpy(d_data1, &h_data[halfSize], halfSize * sizeof(float), cudaMemcpyHostToDevice);
  cudaCheckErrors("cudaMalloc and cudaMemcpy on GPU 1 failed");

  // Launch reduction kernel on both GPUs
  int blockSize = 1024; // Adjust as needed
  int gridSize = (halfSize + blockSize - 1) / blockSize;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);

  cudaSetDevice(0);
  reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_out0, d_data0, halfSize);
  cudaCheckErrors("Kernel launch on GPU 0 failed");
  cudaDeviceSynchronize();

  cudaSetDevice(1);
  reduce_kernel<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_out1, d_data1, halfSize);
  cudaCheckErrors("Kernel launch on GPU 1 failed");
  cudaDeviceSynchronize();

  // Copy partial results back to CPU from both GPUs
  float *partial_sums0 = (float *)malloc(gridSize * sizeof(float));
  float *partial_sums1 = (float *)malloc(gridSize * sizeof(float));

  cudaSetDevice(0);
  cudaMemcpy(partial_sums0, d_out0, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy from GPU 0 failed");

  cudaSetDevice(1);
  cudaMemcpy(partial_sums1, d_out1, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
  cudaCheckErrors("cudaMemcpy from GPU 1 failed");

  // Final reduction on CPU
  float milliseconds = 0;
  float final_sum = 0;
  for (int i = 0; i < gridSize; i++)
  {
    final_sum += partial_sums0[i] + partial_sums1[i];
  }
  *h_result = final_sum;

  cudaEventRecord(stop);
  cudaDeviceSynchronize(); // Ensure all operations are completed
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Multi GPU execution time: %f ms\n", milliseconds);

  // Free resources
  free(partial_sums0);
  free(partial_sums1);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_data0);
  cudaFree(d_out0);
  cudaFree(d_data1);
  cudaFree(d_out1);
}

int main(int argc, char *argv[])
{
  // int size = 1024; // Must be a power of 2
  // int size = atoi(argv[1]);
  // printf("Size: %d\n", size);
  if (argc != 3)
  {
    fprintf(stderr, "Usage: %s <number of elements> <device type>\n", argv[0]);
    return 1;
  }

  int size = atoi(argv[1]);
  int device_type = atoi(argv[2]);

  // Allocate memory for host data
  float *h_data = (float *)malloc(size * sizeof(float));
  float h_result;

  // Initialize data
  for (int i = 0; i < size; i++)
  {
    h_data[i] = 1.0f; // Or any other initialization as required
  }

  // Run the reduction based on the device type
  clock_t start, end;
  float milliseconds = 0;
  switch (device_type)
  {
  case 0: // CPU version
    start = clock();
    reduce_cpu(h_data, &h_result, size);
    end = clock();
    milliseconds = ((float)(end - start)) / CLOCKS_PER_SEC * 1000;
    break;

  case 1: // Single GPU version
    printf("Running on a single GPU\n");
    reduce_single_gpu(h_data, &h_result, size);
    break;

  case 2: // Dual GPU version
    printf("Running on two GPUs\n");
    reduce_dual_gpu(h_data, &h_result, size);
    break;
  default:
    fprintf(stderr, "Invalid device type\n");
    free(h_data);
    return 1;
  }
  printf("Execution Time: %f ms\n", milliseconds);

  free(h_data);
  return 0;
}

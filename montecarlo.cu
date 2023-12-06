#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>

__global__ void monteCarloKernel(int *block_sums, long int num_iterations, unsigned long seed) {
    extern __shared__ int shared_data[];

    int tid = threadIdx.x;
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_count = 0;
    curandState state;
    curand_init(seed, index, 0, &state);

    for (int i = index; i < num_iterations; i += stride) {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }

    // Load local count to shared memory
    shared_data[tid] = local_count;
    __syncthreads();

    // Perform reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    // Write the result for this block to global memory
    if (tid == 0) {
        block_sums[blockIdx.x] = shared_data[0];
    }
}

double monteCarloSingleGPU(long int n) {
    int* block_sums;
    int* d_block_sums;
    int blocks = 1024;
    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * sizeof(int); // Shared memory size per block


    // Allocate host and device memory
    block_sums = (int *) malloc(blocks * sizeof(int));
    cudaMalloc(&d_block_sums, blocks * sizeof(int));
    // Launch kernel
    monteCarloKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_block_sums, n, time(NULL));
    // Copy result back to host
    cudaMemcpy(block_sums, d_block_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    int total_count = 0;
    for (int i = 0; i < blocks; ++i) {
        total_count += block_sums[i];
    }
    // Free device memory   
    cudaFree(d_block_sums);
    free(block_sums);

    // Calculate Pi
    return 4.0 * total_count / n;
}
double monteCarloMulGPU(long int n) {
    int* block_sums, *block_sums2;
    int* d_block_sums, *d_block_sums2;
    int blocks = 1024;
    int threadsPerBlock = 256;
    int sharedMemSize = threadsPerBlock * sizeof(int); // Shared memory size per block

    // Allocate host and device memory
    block_sums = (int *) malloc(blocks * sizeof(int));
    block_sums2 = (int *) malloc(blocks * sizeof(int));
    cudaSetDevice(0);
    cudaMalloc(&d_block_sums, blocks * sizeof(int));
    cudaSetDevice(1);
    cudaMalloc(&d_block_sums2, blocks * sizeof(int));
    // Launch kernel 
    cudaSetDevice(0);
    monteCarloKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_block_sums, n / 2, time(NULL));
    cudaSetDevice(1);
    monteCarloKernel<<<blocks, threadsPerBlock, sharedMemSize>>>(d_block_sums2, n / 2, time(NULL));
    // Copy result back to host
    cudaSetDevice(0);
    cudaMemcpy(block_sums, d_block_sums, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(block_sums2, d_block_sums2, blocks * sizeof(int), cudaMemcpyDeviceToHost);
    
    int total_count = 0;
    for (int i = 0; i < blocks; ++i) {
        total_count += block_sums[i] + block_sums2[i];
    }
    // Free device memory   
    cudaFree(d_block_sums);
    cudaFree(d_block_sums2);
    free(block_sums);
    free(block_sums2);

    // Calculate Pi
    return 4.0 * total_count / n;
}
// Function to estimate Pi using Monte Carlo simulation
double monteCarloSeq(long int n) {
    long int points_inside_circle = 0;
    double x, y;

    for (long int i = 0; i < n; i++) {
        // Generate random point (x, y)
        x = (double)rand() / RAND_MAX;
        y = (double)rand() / RAND_MAX;

        // Check if the point is inside the quarter circle
        if (x * x + y * y <= 1.0) {
            points_inside_circle++;
        }
    }

    // Return the estimation of Pi
    return 4.0 * points_inside_circle / n;
}

int main(int argc, char *argv[]) {
  long int n = 0, type_of_device = 0;

  // to measure the time
  double time_taken = 0;
  clock_t start, end;

  
  if(argc != 3)
  {
    fprintf(stderr, "usage: montecarlo n who\n");
    fprintf(stderr, "n = number of iterations\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: Single GPU version, 2: Multi-GPU version\n");
    exit(1);
  }

  n = atoi(argv[1]);
  type_of_device = atoi(argv[2]);

  double pi;

  switch(type_of_device) {
    case 0: 
        printf("Sequential version:\n");

        start = clock();
        pi = monteCarloSeq(n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("CPU time = %lf secs\n", time_taken); 
        printf("Estimated pi = %lf\n", pi);
        break;
	case 1: 
        printf("Single GPU version:\n");

        start = clock();
        pi = monteCarloSingleGPU(n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("Single GPU time = %lf secs\n", time_taken); 
        printf("Estimated pi = %lf\n", pi);
        break;
	case 2: 
        printf("Multi GPU version:\n");

        start = clock();
        pi = monteCarloMulGPU(n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("Multi GPU time = %lf secs\n", time_taken); 
        printf("Estimated pi = %lf\n", pi);
        break;
	default: 
        printf("Invalid device type\n");
		exit(1);
  }
}


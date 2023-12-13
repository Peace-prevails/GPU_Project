#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <curand_kernel.h>
#include <algorithm>


__device__ void gpuMerge(int* source, int* dest, int start, int mid, int end) {
    int i = start;
    int j = mid;
    for (int k = start; k < end; k++) {
        if (i < mid && (j >= end || source[i] < source[j])) {
            dest[k] = source[i];
            i++;
        } else {
            dest[k] = source[j];
            j++;
        }
    }
}

__global__ void mergeSortKernel(int* source, int* dest, int width, int len) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    int stride = blockDim.x * gridDim.x * width;

    for (int start = index * width; start < len; start += stride) {
      int mid = min(start + (width >> 1), len);
      int end = min(start + width, len);
      gpuMerge(source, dest, start, mid, end);
    }

}

void mergeSortSingleGPU(int* list, int len) {
    int* d_list, *d_temp;
    int blocks = 1024;
    int threadsPerBlock = 256;

    double time_malloc = 0, time_memcpy = 0, time_launch = 0, time_run = 0;
    clock_t start, end;

    // Allocate and copy device memory
    start = clock();
    cudaMalloc(&d_list, len * sizeof(int));
    cudaMalloc(&d_temp, len * sizeof(int));
    end = clock();
    time_malloc += ((double)(end-start)) / CLOCKS_PER_SEC;

    start = clock();
    cudaMemcpy(d_list, list, len * sizeof(int), cudaMemcpyHostToDevice);
    end = clock();
    time_memcpy += ((double)(end-start)) / CLOCKS_PER_SEC;
    // Launch kernel

    for (int width = 2; width < (len << 1); width <<= 1) {
      start = clock();
      mergeSortKernel<<<blocks, threadsPerBlock>>>(d_list, d_temp, width, len);
      end = clock();
      time_launch += ((double)(end-start)) / CLOCKS_PER_SEC;

      cudaDeviceSynchronize();
      end = clock();
      time_run += ((double)(end-start)) / CLOCKS_PER_SEC;

      int* temp = d_list;
      d_list = d_temp;
      d_temp = temp;
    }
    // Copy result back to host
    start = clock();
    cudaMemcpy(list, d_list, len * sizeof(int), cudaMemcpyDeviceToHost);
    end = clock();
    time_memcpy += ((double)(end-start)) / CLOCKS_PER_SEC;

    cudaFree(d_list);
    cudaFree(d_temp);
    printf("cudaMalloc time = %lf secs\n", time_malloc); 
    printf("cudaMemcpy time = %lf secs\n", time_memcpy); 
    printf("kernel launch time = %lf secs\n", time_launch); 
    printf("kernel run time = %lf secs\n", time_run); 
}

void mergeSortMulGPU(int* list, int len) {
    int *d_list, *d_temp, *d_list1, *d_temp1, *d_list2, *d_temp2;
    int blocks = 1024;
    int threadsPerBlock = 256;

    // Allocate and copy device memory
    
    //cudaMemcpy(d_list, list, len * sizeof(int), cudaMemcpyHostToDevice);
    
    cudaSetDevice(0);
    cudaMalloc(&d_list, len * sizeof(int));
    cudaMalloc(&d_temp, len * sizeof(int));


    cudaMalloc(&d_list1, len/2 * sizeof(int));
    cudaMalloc(&d_temp1, len/2 * sizeof(int));
    cudaMemcpy(d_list1, list, len / 2 * sizeof(int), cudaMemcpyHostToDevice);

    cudaSetDevice(1);
    cudaMalloc(&d_list2, len/2 * sizeof(int));
    cudaMalloc(&d_temp2, len/2 * sizeof(int));
    cudaMemcpy(d_list2, &list[len/2], len / 2 * sizeof(int), cudaMemcpyHostToDevice);

    // Launch kernel 
    for (int width = 2; width < (len << 1); width <<= 1) {
      cudaSetDevice(0);
      mergeSortKernel<<<blocks, threadsPerBlock>>>(d_list1, d_temp1, width, len/2);
      cudaSetDevice(1);
      mergeSortKernel<<<blocks, threadsPerBlock>>>(d_list2, d_temp2, width, len/2);

      cudaSetDevice(0);
      cudaDeviceSynchronize();
      cudaSetDevice(1);
      cudaDeviceSynchronize();

      int* temp = d_list1;
      d_list1 = d_temp1;
      d_temp1 = temp;

      temp = d_list2;
      d_list2 = d_temp2;
      d_temp2 = temp;
    }
    
    
    // Copy result back to host
    cudaSetDevice(0);
    cudaMemcpy(list, d_list1, len/2 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(&list[len/2], d_list2, len/2 * sizeof(int), cudaMemcpyDeviceToHost);

    //final merge
    cudaSetDevice(0);
    cudaMemcpy(d_list, list, len * sizeof(int), cudaMemcpyHostToDevice);
    mergeSortKernel<<<blocks, threadsPerBlock>>>(d_list, d_temp, len, len);
    cudaMemcpy(list, d_temp, len * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_list);
    cudaFree(d_temp);
    cudaFree(d_list1);
    cudaFree(d_temp1);
    cudaFree(d_list2);
    cudaFree(d_temp2);
}


int main(int argc, char *argv[]) {
  long int n = 0, type_of_device = 0;

  // to measure the time
  double time_taken = 0;
  clock_t start, end;

  if(argc != 3)
  {
    fprintf(stderr, "usage: mergesort n who\n");
    fprintf(stderr, "n = size of int array to be sorted\n");
    fprintf(stderr, "who = 0: sequential code on CPU, 1: Single GPU version, 2: Multi-GPU version\n");
    exit(1);
  }
  n = atoi(argv[1]);
  type_of_device = atoi(argv[2]);

  // generate random list
  int* list;
  list = (int *) malloc(n * sizeof(int));
  
  srand(time(NULL));
  for (int i = 0; i < n; i++) {
        list[i] = rand(); // Generates a random number
  }

  // printf("Original array:\n");
  // for (int i = 0; i < n; ++i) {
  //   printf("%d\n", list[i]);
  // }

  switch(type_of_device) {
    case 0: 
        printf("Sequential version:\n");

        start = clock();
        std::sort(list, list + n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("CPU time = %lf secs\n", time_taken); 
        break;
	case 1: 
        printf("Single GPU version:\n");

        start = clock();
        mergeSortSingleGPU(list, n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("Single GPU time = %lf secs\n", time_taken); 
        break;
	case 2: 
        printf("Multi GPU version:\n");

        start = clock();
        mergeSortMulGPU(list, n);
        end = clock();  // end of measuring
        time_taken = ((double)(end-start)) / CLOCKS_PER_SEC;
        printf("Multi GPU time = %lf secs\n", time_taken); 
        break;
	default: 
        printf("Invalid device type\n");
		exit(1);
  }

  // printf("Sorted array:\n");
  // for (int i = 0; i < n; ++i) {
  //   printf("%d\n", list[i]);
  // }
  free(list);
}
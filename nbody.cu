#include <cuda.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h> 
#include <cuda_runtime.h>

#define G 6.67430e-11 // Gravitational constant

struct Body {
    float3 position;
    float3 velocity;
    float mass;
};
__global__ void nBodyKernel(Body *bodies, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float3 force = make_float3(0.0f, 0.0f, 0.0f);
        for (int j = 0; j < n; j++) {
            if (idx != j) {
                float3 r = make_float3(bodies[j].position.x - bodies[idx].position.x,
                                       bodies[j].position.y - bodies[idx].position.y,
                                       bodies[j].position.z - bodies[idx].position.z);
                float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + 1e-10f;
                float dist = sqrtf(distSqr);
                float F = G * bodies[idx].mass * bodies[j].mass / distSqr;
                force.x += r.x / dist * F;
                force.y += r.y / dist * F;
                force.z += r.z / dist * F;
            }
        }
        bodies[idx].velocity.x += force.x / bodies[idx].mass;
        bodies[idx].velocity.y += force.y / bodies[idx].mass;
        bodies[idx].velocity.z += force.z / bodies[idx].mass;
    }
}
void initBodies(Body *bodies, int n) {
    for (int i = 0; i < n; i++) {
        bodies[i].position = make_float3(rand() / (float)RAND_MAX, rand() / (float)RAND_MAX, rand() / (float)RAND_MAX);
        bodies[i].velocity = make_float3(0, 0, 0);
        bodies[i].mass = rand() / (float)RAND_MAX * 10.0f;
    }
}

void runNBodySimulationSingleGPU(Body *h_bodies, int n) {
    Body *d_bodies;
    size_t bytes = n * sizeof(Body);
    cudaMalloc(&d_bodies, bytes);

    cudaMemcpy(d_bodies, h_bodies, bytes, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize = (n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    nBodyKernel<<<gridSize, blockSize>>>(d_bodies, n);
    cudaError_t cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }

      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Cuda synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Single GPU execution time: %f ms\n", milliseconds);

    cudaMemcpy(h_bodies, d_bodies, bytes, cudaMemcpyDeviceToHost);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_bodies);
}

void runNBodySimulationMultiGPU(Body *h_bodies, int n) {
    int half_n = n / 2;
    // size_t bytes = n * sizeof(Body);
    size_t half_bytes = half_n * sizeof(Body);

    Body *d_bodies0, *d_bodies1;

    // Initialize GPU 0
    cudaSetDevice(0);
    cudaMalloc(&d_bodies0, half_bytes);
    cudaMemcpy(d_bodies0, h_bodies, half_bytes, cudaMemcpyHostToDevice);

    // Initialize GPU 1
    cudaSetDevice(1);
    cudaMalloc(&d_bodies1, half_bytes);
    cudaMemcpy(d_bodies1, &h_bodies[half_n], half_bytes, cudaMemcpyHostToDevice);

    int blockSize = 32;
    int gridSize0 = (half_n + blockSize - 1) / blockSize;
    int gridSize1 = (n - half_n + blockSize - 1) / blockSize;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaSetDevice(0);
    nBodyKernel<<<gridSize0, blockSize>>>(d_bodies0, half_n);
    cudaError_t cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Kernel1 launch failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }

      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Cuda1 synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }

    cudaSetDevice(1);
    nBodyKernel<<<gridSize1, blockSize>>>(d_bodies1, half_n);
    cudaStatus = cudaGetLastError();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Kernel2 launch failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }

      cudaStatus = cudaDeviceSynchronize();
      if (cudaStatus != cudaSuccess) {
          fprintf(stderr, "Cuda2 synchronize failed: %s\n", cudaGetErrorString(cudaStatus));
          return;
      }

    cudaSetDevice(0);
    cudaDeviceSynchronize();

    cudaSetDevice(1);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);

    cudaMemcpy(h_bodies, d_bodies0, half_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_bodies[half_n], d_bodies1, half_bytes, cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Multi GPU execution time: %f ms\n", milliseconds);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_bodies0);
    cudaFree(d_bodies1);
}


int main(int argc, char * argv[])
{
  unsigned int n; /* Dimention of NxN matrix */
  int type_of_device = 0; // CPU or GPU
  // int iterations = 0;
  // int i;

  n =  atoi(argv[1]);
  type_of_device = atoi(argv[2]);
  Body *h_bodies = (Body *)malloc(n * sizeof(Body));

  initBodies(h_bodies, n);
  
  switch(type_of_device)
  {
	case 0: printf("Single GPU version:\n");
      runNBodySimulationSingleGPU(h_bodies, n);
      break;


		
	case 1: printf("Multi GPU version:\n");
      runNBodySimulationMultiGPU(h_bodies, n);
			break;
			
	default: printf("Invalid device type\n");
			 exit(1);
  }
  free(h_bodies);
  return 0;
}
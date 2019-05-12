/**********************************************************************/
// An unoptimized Naive N-Body solver for Gravity Simulations         //
// G is assumed to be 1.0                                             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin                                             //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include "string.h"
#include <chrono>
#include <math.h>
#include <stdio.h>

#define BLOCKSIZE 1024

void checkCUDAError(const char *msg);

__global__ void forceKernel(double *xPos, double *yPos, double *zPos,
                            double *mass, double *xFor, double *yFor,
                            double *zFor, size_t N) {
  size_t mIdx = blockIdx.x * blockDim.x + threadIdx.x;

  __shared__ double x[BLOCKSIZE];
  __shared__ double y[BLOCKSIZE];
  __shared__ double z[BLOCKSIZE];
  __shared__ double m[BLOCKSIZE];

  double x_force = 0;
  double y_force = 0;
  double z_force = 0;

  double x_mIdx = xPos[mIdx];
  double y_mIdx = yPos[mIdx];
  double z_mIdx = zPos[mIdx];
  double m_mIdx = mass[mIdx];

  for (size_t b = 0; b < N; b += BLOCKSIZE) {
    x[threadIdx.x] = xPos[b + threadIdx.x];
    y[threadIdx.x] = yPos[b + threadIdx.x];
    z[threadIdx.x] = zPos[b + threadIdx.x];
    m[threadIdx.x] = mass[b + threadIdx.x];

    __syncthreads();

#pragma unroll 16
    for (size_t i = 0; i < BLOCKSIZE; ++i) {
      double xDist = x_mIdx - x[i];
      double yDist = y_mIdx - y[i];
      double zDist = z_mIdx - z[i];

      double r = rsqrt(xDist * xDist + yDist * yDist + zDist * zDist + 1e-16);
      double tmp = m_mIdx * m[i] * (r * r * r);

      x_force += xDist * tmp;
      y_force += yDist * tmp;
      z_force += zDist * tmp;
    }

    __syncthreads();
  }

  xFor[mIdx] = x_force;
  yFor[mIdx] = y_force;
  zFor[mIdx] = z_force;
}

int main(int argc, char *argv[]) {
  size_t N0 = 80;
  size_t N = N0 * N0 * N0;

  // Initializing N-Body Problem

  double *xPos = (double *)calloc(N, sizeof(double));
  double *yPos = (double *)calloc(N, sizeof(double));
  double *zPos = (double *)calloc(N, sizeof(double));
  double *xFor = (double *)calloc(N, sizeof(double));
  double *yFor = (double *)calloc(N, sizeof(double));
  double *zFor = (double *)calloc(N, sizeof(double));
  double *mass = (double *)calloc(N, sizeof(double));

  size_t current = 0;
  for (size_t i = 0; i < N0; i++)
    for (size_t j = 0; j < N0; j++)
      for (size_t k = 0; k < N0; k++) {
        xPos[current] = i;
        yPos[current] = j;
        zPos[current] = k;
        mass[current] = 1.0;
        xFor[current] = 0.0;
        yFor[current] = 0.0;
        zFor[current] = 0.0;
        current++;
      }

  // Allocating and initializing GPU memory

  double *dxPos;
  cudaMalloc((void **)&dxPos, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dyPos;
  cudaMalloc((void **)&dyPos, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dzPos;
  cudaMalloc((void **)&dzPos, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dxFor;
  cudaMalloc((void **)&dxFor, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dyFor;
  cudaMalloc((void **)&dyFor, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dzFor;
  cudaMalloc((void **)&dzFor, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");
  double *dmass;
  cudaMalloc((void **)&dmass, sizeof(double) * N);
  checkCUDAError("Unable to allocate storage on the device");

  cudaMemcpy(dxPos, xPos, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dyPos, yPos, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dzPos, zPos, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dxFor, xFor, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dyFor, yFor, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dzFor, zFor, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");
  cudaMemcpy(dmass, mass, sizeof(double) * N, cudaMemcpyHostToDevice);
  checkCUDAError("Failed Initial Conditions Memcpy");

  // Calculating Kernel Geometry
  size_t threadsPerBlock = BLOCKSIZE;
  size_t blocksPerGrid = ceil(double(((double)N) / ((double)threadsPerBlock)));

  // Running Force-calculation kernel
  auto startTime = std::chrono::system_clock::now();
  forceKernel<<<blocksPerGrid, threadsPerBlock>>>(dxPos, dyPos, dzPos, dmass,
                                                  dxFor, dyFor, dzFor, N);
  checkCUDAError("Failed Force Kernel");
  cudaDeviceSynchronize();
  auto endTime = std::chrono::system_clock::now();

  cudaMemcpy(xFor, dxFor, sizeof(double) * N, cudaMemcpyDeviceToHost);
  checkCUDAError("Failed Final Conditions Memcpy");
  cudaMemcpy(yFor, dyFor, sizeof(double) * N, cudaMemcpyDeviceToHost);
  checkCUDAError("Failed Final Conditions Memcpy");
  cudaMemcpy(zFor, dzFor, sizeof(double) * N, cudaMemcpyDeviceToHost);
  checkCUDAError("Failed Final Conditions Memcpy");

  double netForce = 0.0;
  double absForce = 0.0;
  for (size_t i = 0; i < N; i++)
    netForce += xFor[i] + yFor[i] + zFor[i];
  for (size_t i = 0; i < N; i++)
    absForce += abs(xFor[i] + yFor[i] + zFor[i]);

  printf("     Net Force: %.12f\n", netForce);
  printf("Absolute Force: %.12f\n", absForce);

  if (isfinite(netForce) == false) {
    printf("Verification Failed: Net force is not a finite value!\n");
    exit(-1);
  }
  if (fabs(netForce) > 0.00001) {
    printf("Verification Failed: Force equilibrium not conserved!\n");
    exit(-1);
  }
  if (isfinite(absForce) == false) {
    printf("Verification Failed: Absolute Force is not a finite value!\n");
    exit(-1);
  }

  printf("Time: %.8fs\n",
         std::chrono::duration<double>(endTime - startTime).count());
  return 0;
}

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA Error: %s: %s.\n", msg, cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }
}

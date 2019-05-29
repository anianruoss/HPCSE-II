/**********************************************************************/
// A now optimized Multigrid Solver for the Heat Equation             //
// Course Material for HPCSE-II, Spring 2019, ETH Zurich              //
// Authors: Sergio Martin, Georgios Arampatzis                        //
// License: Use if you like, but give us credit.                      //
/**********************************************************************/

#include "heat2d_mpi.hpp"
#include "string.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>
#include <math.h>
#include <mpi.h>
#include <stdio.h>

pointsInfo __p;

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);

  int myRank, rankCount;
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &rankCount);
  bool isMainRank = myRank == 0;

  double tolerance =
      1e-0;       // L2 Difference Tolerance before reaching convergence.
  size_t N0 = 10; // 2^N0 + 1 elements per side
  size_t N = pow(2, N0) - 1; // interior elements per side

  // Multigrid parameters -- Find the best configuration!
  size_t gridCount = 1;       // Number of Multigrid levels to use
  size_t downRelaxations = 3; // Number of Relaxations before restriction
  size_t upRelaxations = 3;   // Number of Relaxations after prolongation

  int dims[2] = {0, 0};
  MPI_Dims_create(rankCount, 2, dims);

  int periodic[2] = {false, false};
  MPI_Comm gridComm;
  MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periodic, false, &gridComm);

  int neighbors[4];
  MPI_Cart_shift(gridComm, 0, 1, &neighbors[0], &neighbors[1]);
  MPI_Cart_shift(gridComm, 1, 1, &neighbors[2], &neighbors[3]);

  size_t nx = N / dims[0];
  size_t ny = N / dims[1];

  // since grid is not divisible by two
  if (neighbors[1] == MPI_PROC_NULL) {
    nx = N - (dims[0] - 1) * (N / dims[0]);
  }
  if (neighbors[3] == MPI_PROC_NULL) {
    ny = N - (dims[1] - 1) * (N / dims[1]);
  }

  size_t fx = nx + 2;
  size_t fy = ny + 2;

  gridLevel *g = allocateRankGrid(N0, fx, fy, gridCount);

  int requestCountInitialization = 0;
  MPI_Request requestsInitialization[4 * (rankCount - 1)];

  if (isMainRank) {
    gridLevel *initialConditions = generateInitialConditions(N0, gridCount);
    int rank, xStart, xEnd, xLen, yStart, yEnd, yLen;
    int coords[2];

    for (int xCoord = 0; xCoord < dims[0]; ++xCoord) {
      for (int yCoord = 0; yCoord < dims[1]; ++yCoord) {
        coords[0] = xCoord;
        coords[1] = yCoord;
        MPI_Cart_rank(gridComm, coords, &rank);

        xStart = xCoord * nx;
        xEnd = coords[0] == dims[0] - 1 ? N + 2 : (xCoord + 1) * nx + 2;
        yStart = yCoord * ny;
        yEnd = coords[1] == dims[1] - 1 ? N + 2 : (yCoord + 1) * ny + 2;

        xLen = xEnd - xStart;
        yLen = yEnd - yStart;

        if (rank == myRank) {
          // copy initial data if on same rank
          for (int i = 0; i < xLen; ++i) {
            for (int j = 0; j < yLen; ++j) {
              g[0].U[i * yLen + j] =
                  initialConditions[0].U[(xStart + i) * g[0].N + yStart + j];
              g[0].f[i * yLen + j] =
                  initialConditions[0].f[(xStart + i) * g[0].N + yStart + j];
            }
          }

        } else {
          // send initial data to other ranks
          double *tmpU = (double *)calloc(xLen * yLen, sizeof(double));
          double *tmpF = (double *)calloc(xLen * yLen, sizeof(double));

          for (int i = 0; i < xLen; ++i) {
            for (int j = 0; j < yLen; ++j) {
              tmpU[i * yLen + j] =
                  initialConditions[0].U[(xStart + i) * g[0].N + yStart + j];
              tmpF[i * yLen + j] =
                  initialConditions[0].f[(xStart + i) * g[0].N + yStart + j];
            }
          }

          MPI_Isend(tmpU, xLen * yLen, MPI_DOUBLE, rank, 0, gridComm,
                    &requestsInitialization[requestCountInitialization++]);
          MPI_Isend(tmpF, xLen * yLen, MPI_DOUBLE, rank, 1, gridComm,
                    &requestsInitialization[requestCountInitialization++]);
        }
      }
    }

    freeGrids(initialConditions, gridCount);

  } else {
    MPI_Irecv(g[0].U, fx * fy, MPI_DOUBLE, 0, 0, gridComm,
              &requestsInitialization[requestCountInitialization++]);
    MPI_Irecv(g[0].f, fx * fy, MPI_DOUBLE, 0, 1, gridComm,
              &requestsInitialization[requestCountInitialization++]);
  }

  MPI_Waitall(requestCountInitialization, requestsInitialization,
              MPI_STATUS_IGNORE);

  MPI_Datatype boundaryX, boundaryY;
  MPI_Type_vector(1, fy, 0, MPI_DOUBLE, &boundaryX);
  MPI_Type_vector(fx, 1, fy, MPI_DOUBLE, &boundaryY);
  MPI_Type_commit(&boundaryX);
  MPI_Type_commit(&boundaryY);

  MPI_Request requests[8];

  auto startTime = std::chrono::system_clock::now();
  while (g[0].L2NormDiff > tolerance) // Multigrid solver start
  {
    // Relaxing the finest grid first
    for (size_t r = 0; r < downRelaxations; r++) {
      applyJacobi(g, 0, fx, fy);

      int request_count = 0;

      MPI_Irecv(&g[0].U[0], 1, boundaryX, neighbors[0], 1, gridComm,
                &requests[request_count++]);
      MPI_Irecv(&g[0].U[(fx - 1) * fy], 1, boundaryX, neighbors[1], 1, gridComm,
                &requests[request_count++]);
      MPI_Irecv(&g[0].U[0], 1, boundaryY, neighbors[2], 1, gridComm,
                &requests[request_count++]);
      MPI_Irecv(&g[0].U[fy - 1], 1, boundaryY, neighbors[3], 1, gridComm,
                &requests[request_count++]);

      MPI_Isend(&g[0].U[fy], 1, boundaryX, neighbors[0], 1, gridComm,
                &requests[request_count++]);
      MPI_Isend(&g[0].U[(fx - 2) * fy], 1, boundaryX, neighbors[1], 1, gridComm,
                &requests[request_count++]);
      MPI_Isend(&g[0].U[1], 1, boundaryY, neighbors[2], 1, gridComm,
                &requests[request_count++]);
      MPI_Isend(&g[0].U[fy - 2], 1, boundaryY, neighbors[3], 1, gridComm,
                &requests[request_count++]);

      MPI_Waitall(request_count, requests, MPI_STATUS_IGNORE);
    }

    // Calculating Initial Residual
    calculateResidual(g, 0, fx, fy);

    // not needed for gridCount == 1
    /*
    for (size_t grid = 1; grid < gridCount; grid++) // Going down the V-Cycle
    {
      applyRestriction(g, grid); // Restricting the residual to the coarser
                                 // grid's solution vector (f)
      applyJacobi(g, grid, downRelaxations); // Smoothing coarser level
      calculateResidual(g, grid);            // Calculating Coarse Grid Residual
    }

    for (size_t grid = gridCount - 1; grid > 0; grid--) // Going up the V-Cycle
    {
      applyProlongation(
          g, grid); // Prolonging solution for coarser level up to finer level
      applyJacobi(g, grid, upRelaxations); // Smoothing finer level
    }
    */

    calculateL2Norm(g, 0, fx, fy); // Calculating Residual L2 Norm
  }                                // Multigrid solver end

  auto endTime = std::chrono::system_clock::now();
  totalTime = std::chrono::duration<double>(endTime - startTime).count();

  for (size_t grid = 0; grid < gridCount; ++grid) {
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &smoothingTime[grid],
               &smoothingTime[grid], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &residualTime[grid],
               &residualTime[grid], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &restrictionTime[grid],
               &restrictionTime[grid], 1, MPI_DOUBLE, MPI_MAX, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &prolongTime[grid],
               &prolongTime[grid], 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &L2NormTime[grid], &L2NormTime[grid],
               1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(isMainRank ? MPI_IN_PLACE : &totalTime, &totalTime, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  }

  if (isMainRank) {
    printTimings(gridCount);
    printf("L2Norm: %.4f\n", g[0].L2Norm);
  }

  freeGrids(g, gridCount);

  MPI_Type_free(&boundaryX);
  MPI_Type_free(&boundaryY);
  MPI_Comm_free(&gridComm);

  MPI_Finalize();

  return 0;
}

void applyJacobi(gridLevel *g, size_t l, size_t fx, size_t fy) {
  auto t0 = std::chrono::system_clock::now();

  double h1 = 0.25;
  double h2 = g[l].h * g[l].h;

  double *tmp = g[l].Un;
  g[l].Un = g[l].U;
  g[l].U = tmp;

  // Perform a Jacobi Iteration
  for (size_t i = 1; i < fx - 1; i++) {
    for (size_t j = 1; j < fy - 1; j++) {
      g[l].U[i * fy + j] =
          (g[l].Un[(i - 1) * fy + j] + g[l].Un[(i + 1) * fy + j] +
           g[l].Un[i * fy + j - 1] + g[l].Un[i * fy + j + 1] +
           g[l].f[i * fy + j] * h2) *
          h1;
    }
  }

  auto t1 = std::chrono::system_clock::now();
  smoothingTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateResidual(gridLevel *g, size_t l, size_t fx, size_t fy) {
  auto t0 = std::chrono::system_clock::now();

  double h2 = 1.0 / pow(g[l].h, 2);

  for (size_t i = 1; i < fx - 1; i++)
    for (size_t j = 1; j < fy - 1; j++)
      g[l].Res[i * fy + j] =
          g[l].f[i * fy + j] +
          (g[l].U[(i - 1) * fy + j] + g[l].U[(i + 1) * fy + j] -
           4 * g[l].U[i * fy + j] + g[l].U[i * fy + j - 1] +
           g[l].U[i * fy + j + 1]) *
              h2;

  auto t1 = std::chrono::system_clock::now();
  residualTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

void calculateL2Norm(gridLevel *g, size_t l, size_t fx, size_t fy) {
  auto t0 = std::chrono::system_clock::now();

  double tmp = 0.0;

  for (size_t i = 1; i < fx - 1; i++)
    for (size_t j = 1; j < fy - 1; j++)
      g[l].Res[i * fy + j] = g[l].Res[i * fy + j] * g[l].Res[i * fy + j];

  for (size_t i = 1; i < fx - 1; i++) {
    for (size_t j = 1; j < fy - 1; j++) {
      tmp += g[l].Res[i * fy + j];
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &tmp, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

  g[l].L2Norm = sqrt(tmp);
  g[l].L2NormDiff = fabs(g[l].L2NormPrev - g[l].L2Norm);
  g[l].L2NormPrev = g[l].L2Norm;

  auto t1 = std::chrono::system_clock::now();
  L2NormTime[l] += std::chrono::duration<double>(t1 - t0).count();
}

// not needed for gridCount == 1
/*
void applyRestriction(gridLevel *g, size_t l) {
  auto t0 = std::chrono::system_clock::now();

  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l].f[i][j] = (1.0 * (g[l - 1].Res[2 * i - 1][2 * j - 1] +
                             g[l - 1].Res[2 * i - 1][2 * j + 1] +
                             g[l - 1].Res[2 * i + 1][2 * j - 1] +
                             g[l - 1].Res[2 * i + 1][2 * j + 1]) +
                      2.0 * (g[l - 1].Res[2 * i - 1][2 * j] +
                             g[l - 1].Res[2 * i][2 * j - 1] +
                             g[l - 1].Res[2 * i + 1][2 * j] +
                             g[l - 1].Res[2 * i][2 * j + 1]) +
                      4.0 * (g[l - 1].Res[2 * i][2 * j])) *
                     0.0625;

  for (size_t i = 0; i < g[l].N; i++)
    for (size_t j = 0; j < g[l].N;
         j++) // Resetting U vector for the coarser level before smoothing --
              // Find out if this is really necessary.
      g[l].U[i][j] = 0;

  auto t1 = std::chrono::system_clock::now();
  restrictionTime[l] += std::chrono::duration<double>(t1 - t0).count();
}
*/

// not needed for gridCount == 1
/*
void applyProlongation(gridLevel *g, size_t l) {
  auto t0 = std::chrono::system_clock::now();

  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l - 1].U[2 * i][2 * j] += g[l].U[i][j];

  for (size_t i = 1; i < g[l].N; i++)
    for (size_t j = 1; j < g[l].N - 1; j++)
      g[l - 1].U[2 * i - 1][2 * j] += (g[l].U[i - 1][j] + g[l].U[i][j]) * 0.5;

  for (size_t i = 1; i < g[l].N - 1; i++)
    for (size_t j = 1; j < g[l].N; j++)
      g[l - 1].U[2 * i][2 * j - 1] += (g[l].U[i][j - 1] + g[l].U[i][j]) * 0.5;

  for (size_t i = 1; i < g[l].N; i++)
    for (size_t j = 1; j < g[l].N; j++)
      g[l - 1].U[2 * i - 1][2 * j - 1] +=
          (g[l].U[i - 1][j - 1] + g[l].U[i - 1][j] + g[l].U[i][j - 1] +
           g[l].U[i][j]) *
          0.25;

  auto t1 = std::chrono::system_clock::now();
  prolongTime[l] += std::chrono::duration<double>(t1 - t0).count();
}
*/

gridLevel *allocateRankGrid(size_t N0, size_t fx, size_t fy, size_t gridCount) {
  // only works with gridCount == 1
  assert(gridCount == 1);

  // Allocating Timers
  smoothingTime = (double *)calloc(gridCount, sizeof(double));
  residualTime = (double *)calloc(gridCount, sizeof(double));
  restrictionTime = (double *)calloc(gridCount, sizeof(double));
  prolongTime = (double *)calloc(gridCount, sizeof(double));
  L2NormTime = (double *)calloc(gridCount, sizeof(double));

  // Allocating Grids
  gridLevel *g = (gridLevel *)malloc(sizeof(gridLevel) * gridCount);
  for (size_t i = 0; i < gridCount; i++) {
    g[i].N = pow(2, N0 - i) + 1;
    g[i].h = 1.0 / (g[i].N - 1);

    g[i].U = (double *)calloc(fx * fy, sizeof(double));
    g[i].Un = (double *)calloc(fx * fy, sizeof(double));
    g[i].Res = (double *)calloc(fx * fy, sizeof(double));
    g[i].f = (double *)calloc(fx * fy, sizeof(double));

    g[i].L2Norm = 0.0;
    g[i].L2NormPrev = std::numeric_limits<double>::max();
    g[i].L2NormDiff = std::numeric_limits<double>::max();
  }

  return g;
}

gridLevel *generateInitialConditions(size_t N0, size_t gridCount) {
  // Default values:
  __p.nCandles = 4;
  std::vector<double> pars;
  pars.push_back(0.228162);
  pars.push_back(0.226769);
  pars.push_back(0.437278);
  pars.push_back(0.0492324);
  pars.push_back(0.65915);
  pars.push_back(0.499616);
  pars.push_back(0.59006);
  pars.push_back(0.0566329);
  pars.push_back(0.0186672);
  pars.push_back(0.894063);
  pars.push_back(0.424229);
  pars.push_back(0.047725);
  pars.push_back(0.256743);
  pars.push_back(0.754483);
  pars.push_back(0.490461);
  pars.push_back(0.0485152);

  // Allocating Grids
  gridLevel *g = (gridLevel *)malloc(sizeof(gridLevel) * gridCount);
  for (size_t i = 0; i < gridCount; i++) {
    g[i].N = pow(2, N0 - i) + 1;
    g[i].h = 1.0 / (g[i].N - 1);

    g[i].U = (double *)calloc(g[i].N * g[i].N, sizeof(double));
    g[i].Un = (double *)calloc(g[i].N * g[i].N, sizeof(double));
    g[i].Res = (double *)calloc(g[i].N * g[i].N, sizeof(double));
    g[i].f = (double *)calloc(g[i].N * g[i].N, sizeof(double));

    g[i].L2Norm = 0.0;
    g[i].L2NormPrev = std::numeric_limits<double>::max();
    g[i].L2NormDiff = std::numeric_limits<double>::max();
  }

  // Initial Guess
  for (size_t i = 0; i < g[0].N; i++)
    for (size_t j = 0; j < g[0].N; j++)
      g[0].U[i * g[0].N + j] = 1.0;

  // Boundary Conditions
  for (size_t i = 0; i < g[0].N; i++)
    g[0].U[i] = 0.0;
  for (size_t i = 0; i < g[0].N; i++)
    g[0].U[(g[0].N - 1) * g[0].N + i] = 0.0;
  for (size_t i = 0; i < g[0].N; i++)
    g[0].U[i * g[0].N] = 0.0;
  for (size_t i = 0; i < g[0].N; i++)
    g[0].U[i * g[0].N + g[0].N - 1] = 0.0;

  // F
  for (size_t i = 0; i < g[0].N; i++)
    for (size_t j = 0; j < g[0].N; j++) {
      double h = 1.0 / (g[0].N - 1);
      double x = i * h;
      double y = j * h;

      g[0].f[i * g[0].N + j] = 0.0;

      for (size_t c = 0; c < __p.nCandles; c++) {
        double c3 = pars[c * 4 + 0]; // x0
        double c4 = pars[c * 4 + 1]; // y0
        double c1 = pars[c * 4 + 2];
        c1 *= 100000; // intensity
        double c2 = pars[c * 4 + 3];
        c2 *= 0.01; // Width
        g[0].f[i * g[0].N + j] +=
            c1 * exp(-(pow(c4 - y, 2) + pow(c3 - x, 2)) / c2);
      }
    }

  return g;
}

void freeGrids(gridLevel *g, size_t gridCount) {
  for (size_t i = 0; i < gridCount; i++) {
    free(g[i].U);
    free(g[i].Un);
    free(g[i].f);
    free(g[i].Res);
  }
  free(g);
}

void printTimings(size_t gridCount) {
  double *timePerGrid = (double *)calloc(sizeof(double), gridCount);
  double totalSmoothingTime = 0.0;
  double totalResidualTime = 0.0;
  double totalRestrictionTime = 0.0;
  double totalProlongTime = 0.0;
  double totalL2NormTime = 0.0;

  for (size_t i = 0; i < gridCount; i++)
    timePerGrid[i] = smoothingTime[i] + residualTime[i] + restrictionTime[i] +
                     prolongTime[i] + L2NormTime[i];
  for (size_t i = 0; i < gridCount; i++)
    totalSmoothingTime += smoothingTime[i];
  for (size_t i = 0; i < gridCount; i++)
    totalResidualTime += residualTime[i];
  for (size_t i = 0; i < gridCount; i++)
    totalRestrictionTime += restrictionTime[i];
  for (size_t i = 0; i < gridCount; i++)
    totalProlongTime += prolongTime[i];
  for (size_t i = 0; i < gridCount; i++)
    totalL2NormTime += L2NormTime[i];

  double totalMeasured = totalSmoothingTime + totalResidualTime +
                         totalRestrictionTime + totalProlongTime +
                         totalL2NormTime;

  printf("   Time (s)    ");
  for (size_t i = 0; i < gridCount; i++)
    printf("Grid%lu   ", i);
  printf("   Total  \n");
  printf("-------------|-");
  for (size_t i = 0; i < gridCount; i++)
    printf("--------");
  printf("|---------\n");
  printf("Smoothing    | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", smoothingTime[i]);
  printf("|  %2.3f  \n", totalSmoothingTime);
  printf("Residual     | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", residualTime[i]);
  printf("|  %2.3f  \n", totalResidualTime);
  printf("Restriction  | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", restrictionTime[i]);
  printf("|  %2.3f  \n", totalRestrictionTime);
  printf("Prolongation | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", prolongTime[i]);
  printf("|  %2.3f  \n", totalProlongTime);
  printf("L2Norm       | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", L2NormTime[i]);
  printf("|  %2.3f  \n", totalL2NormTime);
  printf("-------------|-");
  for (size_t i = 0; i < gridCount; i++)
    printf("--------");
  printf("|---------\n");
  printf("Total        | ");
  for (size_t i = 0; i < gridCount; i++)
    printf("%2.3f   ", timePerGrid[i]);
  printf("|  %2.3f  \n", totalMeasured);
  printf("-------------|-");
  for (size_t i = 0; i < gridCount; i++)
    printf("--------");
  printf("|---------\n");
  printf("\n");
  printf("Running Time      : %.3fs\n", totalTime);
}

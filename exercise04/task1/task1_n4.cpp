#include "model/heat2d.hpp"
#include "korali.h"

int main(int argc, char *argv[]) {

  // Loading temperature measurement data
  FILE *dataFile = fopen("data_n4.in", "r");
  fscanf(dataFile, "%lu", &p.nPoints);

  p.xPos = (double *)calloc(sizeof(double), p.nPoints);
  p.yPos = (double *)calloc(sizeof(double), p.nPoints);
  p.refTemp = (double *)calloc(sizeof(double), p.nPoints);

  for (int i = 0; i < p.nPoints; i++) {
    fscanf(dataFile, "%le ", &p.xPos[i]);
    fscanf(dataFile, "%le ", &p.yPos[i]);
    fscanf(dataFile, "%le ", &p.refTemp[i]);
  }

  auto problem = Korali::Problem::Posterior(heat2DSolver);

  // 4-Candle Model x4 parameters/candle
  p.nCandles = 4;

  Korali::Parameter::Uniform torch_1_x("torch_1_x", 0.0, 0.5);
  Korali::Parameter::Uniform torch_2_x("torch_2_x", 0.0, 0.5);
  Korali::Parameter::Uniform torch_3_x("torch_3_x", 0.5, 1.0);
  Korali::Parameter::Uniform torch_4_x("torch_4_x", 0.5, 1.0);

  Korali::Parameter::Uniform torch_1_y("torch_1_y", 0.0, 1.0);
  Korali::Parameter::Uniform torch_2_y("torch_2_y", 0.0, 1.0);
  Korali::Parameter::Uniform torch_3_y("torch_3_y", 0.0, 1.0);
  Korali::Parameter::Uniform torch_4_y("torch_4_y", 0.0, 1.0);

  Korali::Parameter::Uniform torch_1_intensity("torch_1_intensity", 0.4, 0.6);
  Korali::Parameter::Uniform torch_2_intensity("torch_2_intensity", 0.4, 0.6);
  Korali::Parameter::Uniform torch_3_intensity("torch_3_intensity", 0.4, 0.6);
  Korali::Parameter::Uniform torch_4_intensity("torch_4_intensity", 0.4, 0.6);

  Korali::Parameter::Uniform torch_1_width("torch_1_width", 0.04, 0.06);
  Korali::Parameter::Uniform torch_2_width("torch_2_width", 0.04, 0.06);
  Korali::Parameter::Uniform torch_3_width("torch_3_width", 0.04, 0.06);
  Korali::Parameter::Uniform torch_4_width("torch_4_width", 0.04, 0.06);

  problem.addParameter(&torch_1_x);
  problem.addParameter(&torch_1_y);
  problem.addParameter(&torch_1_intensity);
  problem.addParameter(&torch_1_width);

  problem.addParameter(&torch_2_x);
  problem.addParameter(&torch_2_y);
  problem.addParameter(&torch_2_intensity);
  problem.addParameter(&torch_2_width);

  problem.addParameter(&torch_3_x);
  problem.addParameter(&torch_3_y);
  problem.addParameter(&torch_3_intensity);
  problem.addParameter(&torch_3_width);

  problem.addParameter(&torch_4_x);
  problem.addParameter(&torch_4_y);
  problem.addParameter(&torch_4_intensity);
  problem.addParameter(&torch_4_width);

  problem.setReferenceData(p.nPoints, p.refTemp);

  auto solver = Korali::Solver::CMAES(&problem);

  // int Ng = 2000; // max generations for CMAES
  int Ng = 1e7; // max generations for CMAES

  solver.setStopMinDeltaX(1e-6);
  // solver.setPopulationSize(8); // ~4+3*log(N)
  solver.setPopulationSize(24); // ~4+3*log(N)
  solver.setMu(4);
  solver.setMaxGenerations(Ng);
  solver.run();

  return 0;
}

#include "model/heat2d.hpp"
#include "korali.h"


int main(int argc, char* argv[]) {
	// Loading temperature measurement data

	FILE* dataFile = fopen("data.in", "r");
	fscanf(dataFile, "%lu", &p.nPoints);

	p.xPos    = (double*) calloc (sizeof(double), p.nPoints);
	p.yPos    = (double*) calloc (sizeof(double), p.nPoints);
	p.refTemp = (double*) calloc (sizeof(double), p.nPoints);

	for (int i = 0; i < p.nPoints; i++)
	{
		fscanf(dataFile, "%le ", &p.xPos[i]);
		fscanf(dataFile, "%le ", &p.yPos[i]);
		fscanf(dataFile, "%le ", &p.refTemp[i]);
	}

	// Start configuring the Problem and the Korali Engine
	Korali::Problem::Posterior problem(heat2DSolver);

	// 3-Candle Model
	p.nCandles = 3;

	Korali::Parameter::Gaussian pos_x_1("pos_x_1", 0.25, 0.05);
	Korali::Parameter::Gaussian pos_x_2("pos_x_2", 0.75, 0.05);
	Korali::Parameter::Gaussian pos_x_3("pos_x_3", 0.75, 0.05);

	pos_x_1.setBounds(0.0, 0.5);
	pos_x_2.setBounds(0.5, 1.0);
	pos_x_3.setBounds(0.5, 1.0);

	Korali::Parameter::Uniform pos_y_1("pos_y_1", 0.0, 1.0);
	Korali::Parameter::Uniform pos_y_2("pos_y_2", 0.0, 1.0);
	Korali::Parameter::Uniform pos_y_3("pos_y_3", 0.0, 1.0);

	problem.addParameter(&pos_x_1);
	problem.addParameter(&pos_y_1);
	problem.addParameter(&pos_x_2);
	problem.addParameter(&pos_y_2);
	problem.addParameter(&pos_x_3);
	problem.addParameter(&pos_y_3);

	problem.setReferenceData(p.nPoints, p.refTemp);

	// Sample distribution
	Korali::Solver::TMCMC solver(&problem);

	solver.setPopulationSize(10000);
	solver.setCovarianceScaling(0.2);
	
	// Find maximum parameters
	/*
	Korali::Solver::CMAES solver(&problem);

	solver.setStopMinDeltaX(1e-11);
	solver.setPopulationSize(64);
	solver.setMaxGenerations(1000);
	*/

	solver.run();

	return 0;
}


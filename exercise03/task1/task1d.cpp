#include "model/grass.hpp"
#include "korali.h"


// Grass Height at different spots, as measured by Herr Kueheli.
size_t  nSpots;
double* xPos;
double* yPos;
double* heights;


void LikelihoodGrassHeight(double* x, double* fx) {
	double ph = x[0];
	double mm = x[1];

	for (int i = 0; i < nSpots; ++i) {
		fx[i] = getGrassHeight(xPos[i], yPos[i], ph, mm);
	}
}



int main(int argc, char* argv[]) {

	// Loading grass height data

	FILE* dataFile = fopen("grass.in", "r");

	fscanf(dataFile, "%lu", &nSpots);
	xPos     = (double*) calloc (sizeof(double), nSpots);
	yPos     = (double*) calloc (sizeof(double), nSpots);
	heights  = (double*) calloc (sizeof(double), nSpots);

	for (int i = 0; i < nSpots; i++)
	{
		fscanf(dataFile, "%le ", &xPos[i]);
		fscanf(dataFile, "%le ", &yPos[i]);
		fscanf(dataFile, "%le ", &heights[i]);
	}


	Korali::Problem::Posterior problem(LikelihoodGrassHeight);

	Korali::Parameter::Uniform ph("pH", 4.0, 9.0);
	Korali::Parameter::Gaussian mm("mm", 90.0, 20.0);
	mm.setBounds(0.0, 180.0);

	problem.addParameter(&ph);
	problem.addParameter(&mm);

	problem.setReferenceData(nSpots, heights);

	// used this to compute the MAP for pH and mm
	/*
	Korali::Solver::CMAES solver(&problem);

	solver.setStopMinDeltaX(1e-11);
	solver.setPopulationSize(256);
	solver.setMaxGenerations(1000);

	solver.run();
	*/

	// used this to sample the posterior
	Korali::Solver::TMCMC solver(&problem);
	
	solver.setPopulationSize(10000);
	solver.setCovarianceScaling(0.2);

	solver.run();

	return 0;
}


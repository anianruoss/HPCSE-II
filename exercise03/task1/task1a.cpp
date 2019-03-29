#include "model/grass.hpp"
#include "korali.h"

#define MM 80.0
#define PH 6.0


double MaximizeGrassHeight(double* x) {
	return getGrassHeight(x[0], x[1], PH, MM);
}


int main(int argc, char* argv[]) {

	Korali::Problem::Direct problem(MaximizeGrassHeight);
	
	Korali::Parameter::Uniform x("x", 0., 5.);
	Korali::Parameter::Uniform y("y", 0., 5.);
	
	problem.addParameter(&x);
	problem.addParameter(&y);

	Korali::Solver::CMAES solver(&problem);

	solver.setStopMinDeltaX(1e-11);
	solver.setPopulationSize(256);

	solver.run();

	return 0;
}


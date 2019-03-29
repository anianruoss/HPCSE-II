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

	Korali::Solver::TMCMC solver(&problem);

	solver.setPopulationSize(10000);
	solver.setCovarianceScaling(0.2);

	solver.run();

	return 0;
}


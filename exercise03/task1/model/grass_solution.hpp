#ifndef GRASS_SOLUTION_HPP
#define GRASS_SOLUTION_HPP

template<typename T> T sqr(T const& v) {return v*v;}

namespace reverse_engineered {
	/**
	 * Model of grass growth
	 * @param x X-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
	 * @param y Y-axis position of the point to be evaluated. Range: [0.0-5.0] meters.
	 * @param ph pH of the soil
	 * @param mm mm of rain of the previous month.
	 * @return height of grass at coordinate
	 */
	double getGrassHeight(
			double x, 
			double y, 
			double ph, 
			double mm)
	{
		return -(mm+20.0)*sqr(0.8*(y-2.4976)-sqr((x-2.8352)*0.8))-(ph/6.0)*sqr((x-2.8352)*0.8 - 1.0);
	}
}

#endif /* GRASS_SOLUTION_HPP */


#include <iostream>
//#include <Python.h>
#include <cmath>

using namespace std;

//int sample(self,nsteps) {
int sample(int nsteps) {
	/* Perform nsteps of posterior sampling. */

	for( int step = 0; step < nsteps; step = step + 1) {
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
		if (x < 0.16) {
			std::cout << x << " < 0.16" << std::endl;
		}
		else if (x < 0.32) {
			std::cout << x << " < 0.32" << std::endl;
		}
		else if (x < 0.48) {
			std::cout << x << " < 0.48" << std::endl;
		}
		else if (x < 0.60) {
			std::cout << x << " < 0.60" << std::endl;
		}
		else if (x < 0.78) {
			std::cout << x << " < 0.78" << std::endl;
		}
		else if (x < 0.99) {
			std::cout << x << " < 0.99" << std::endl;
		}
		else {
			std::cout << x << "switch to random pair" << std::endl;
		}
		//
	}
}


int main() {
	std::cout << sample(100);
	return 0;
}














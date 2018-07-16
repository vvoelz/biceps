#include <iostream>
//#include <Python.h>
#include <cmath>

using namespace std;

/* The parameters that need to be placed as elements inside the array
 * sigma_noe,sigma_J,sigma_cs_H,sigma_cs_Ha,sigma_cs_Ca,sigma_cs_N,sigma_pf
 * need indices for all above
 * gamma,gmma_index
 * sse
 *
 */


/* Need to create a pointer for int step
 * Create a pointer for the array length
 *
 *
 */

//float *Para;
//Para = &(Number of parameters - ?length of self.restraints?)

///* Need to know how many parameters to sample over */
//float array[&Para];  // array to hold parameters
//
// nsteps  = &steps  // set step to point at the input argument for nsteps


// array[step-1]


//int sample(self,nsteps) {
int sample(int nsteps) {
	/* Perform nsteps of posterior sampling. */

	for( int step = 0; step < nsteps; step++) {
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














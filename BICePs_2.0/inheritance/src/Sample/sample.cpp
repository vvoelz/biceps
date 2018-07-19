#include <iostream>
//#include <Python.h>
#include <cmath>
#include "sample.h"    // header file for cpp code
#include <fstream>    // Prints to a file
#include <ctime>      // contains the time for the random number generation
#include <cstdlib>  // srand() - random number generation

using namespace std;

FILE *t;    // Setting a pointer to the new array file

void c_sample(int nsteps, std::vector<double> array)
{
	/* Perform nsteps of posterior sampling. */

    int len_array = array.size(); // length of the input array

    // Print out the length of the input array
    std::cout << "vector length " << len_array << std::endl;

    // Open the file for the new array
    t = fopen("new_array.txt","w");

    // Set the random number generator
    srand(time(NULL));

	for( int step = 0; step < nsteps; step++) {
        // Generate random values
		float x = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

		if (x < 0.16) {
			std::cout << x << " < 0.16" << std::endl;
            // add 1 to first element in matrix
            array[0] = array[0] + 1.0;
		}
		else if (x < 0.32) {
			std::cout << x << " < 0.32" << std::endl;
            // add 1 to second element in matrix
            array[1] = array[1] + 1.0;
		}
		else if (x < 0.48) {
			std::cout << x << " < 0.48" << std::endl;
            // add 1 to third element in matrix
            array[2] = array[2] + 1.0;
		}
		else if (x < 0.60) {
			std::cout << x << " < 0.60" << std::endl;
            // add 1 to first element in matrix
            array[0] = array[0] + 1.0;
		}
		else if (x < 0.78) {
			std::cout << x << " < 0.78" << std::endl;
            // add 1 to second element in matrix
            array[1] = array[1] + 1.0;
		}
		else if (x < 0.99) {
			std::cout << x << " < 0.99" << std::endl;
            // add 1 to third element in matrix
            array[2] = array[2] + 1.0;
		}
		else {
			std::cout << x << "switch to random pair" << std::endl;
            array[2] = array[2] + 1.0;
		}
	}
    // Output the final values of the array
    std::cout << "array[0] =" << array[0] << std::endl;
    std::cout << "array[1] =" << array[1] << std::endl;
    std::cout << "array[2] =" << array[2] << std::endl;
    // Print the file values of the new array to a file
    fprintf(t,"%f\n %f\n %f\n",array[0],array[1],array[2]);
    fclose(t);
}







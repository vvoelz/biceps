
//##### INCLUDE #####:{{{
#include <iostream> // basic input/output stream
#include <cmath>
#include <vector>
#include <fstream>  // writing to file
#include <sstream>
#include <stdexcept>
#include <cstdarg>
#include <cstring>
#include <ctime>
#include <cstdlib>  // srand() - random number generation
#include <omp.h>
#include <stdio.h>
#include <numeric>


using namespace std;
using std::vector;

//}}}

// ##### Templates #####: {{{
template<class T=float, class U=float>
vector<T> rangeE(T start, T stop, U step=1.0){

   if (step == float(0) || int(0)) {
      throw invalid_argument("step for range must be non-zero");
    }

    vector<T> v;
    while(1) {
        if (start >= stop-step/2.0) {
            break;
        }
        v.push_back(start);
        start += step;
    }
    return v;
}
template<class T=float, class U=float>
vector<T> rangeI(T start, T stop, U step=1.0){

   if (step == float(0) || int(0)) {
      throw invalid_argument("step for range must be non-zero");
    }

    vector<T> v;
    while(1) {
        if (start >= stop+step) {
            break;
        }
        v.push_back(start);
        start += step;
    }
    return v;
}

//TODO: Need to make this function general, where the user doesn't have to
// specify the type





void print_4D_vec(vector< vector< vector< vector<float> > > > v) {
    for (int i=0; i<v.size(); ++i) {
        for (int j=0; j<v[i].size(); ++j) {
            for (int k=0; k<v[i].size(); ++k) {
                for (int l=0; l<v[i].size(); ++l) {
                    cout << v[i][j][k][l] << endl;
                }
            }
        }
    }
}

void print_3D_vec(vector< vector< vector<float> > > v) {
    for (int i=0; i<v.size(); ++i) {
        for (int j=0; j<v[i].size(); ++j) {
            for (int k=0; k<v[i].size(); ++k) {
                cout << v[i][j][k] << endl;
            }
        }
    }
}

void print_2D_vec(vector< vector<float> > v) {
    int x = 0;
    for (int i=0; i<v.size(); ++i) {
        for (int j=0; j<v[i].size(); ++j) {
            if (x != i) {
                x += 1;
            }
            cout << v[i][j] << endl;
        }
    }
}

void print_vec(vector<float> v) {
    int x = 0;
    for (int i=0; i<v.size(); ++i) {
        if (x != i) {
            x += 1;
        }
        cout << v[i] << endl;
    }
}



// Function to transpose a 2-D vector
vector< vector<float> > transpose(vector< vector<float> > v) {
    vector<vector<float> > trans_v(v[0].size(), vector<float>(v.size()));
    for (int i=0; i<v.size(); i++) {
       for (int j=0; j<v[i].size(); j++) {
           trans_v[j][i] = v[i][j];
       }
    }
    return trans_v;
}



vector<float> getFileContent(string fileName) {
    /* Obtain the file contents of all lines and output
     * the data as a vector.
     */
	// Open the File
	ifstream in(fileName.c_str());

	// Check if object is valid
	if(!in) {
		cerr << "Cannot open the File : "<<fileName<<endl;
		exit(1);
	}
	string num;
    vector<float> vecOfInts;
	// Read the next line from File untill it reaches the end.
	while (getline(in, num)) {
        if (num.size() > 0) {
            //float line = stoi(num);
            istringstream iss(num);
            float line;
            iss >> line;
            vecOfInts.push_back(line);
        }
	}
	//Close The File
	in.close();
	return vecOfInts;
}


vector<float> get_file_content(string fileName) {
    /* Obtain the file contents of all lines and output
     * the data as a vector.
     */
	// Open the File
    FILE* in = fopen(fileName.c_str(), "r");

	// Check if object is valid
	if(!in) {
		cerr << "Cannot open the File : "<<fileName<<endl;
		exit(1);
	}
    char text[80];
	//string num;
    vector<float> vecOfInts;
	// Read the next line from File untill it reaches the end.
	while (fgets(text, sizeof(text), in)) {
        if (strlen(text) > 0) {
            istringstream iss(text);
            float line;
            iss >> line;
            vecOfInts.push_back(line);
        }
	}
	//Close The File
	fclose(in);
	return vecOfInts;
}



string stringFormat(const string& format, ...) {
    va_list args;
    va_start(args, format);
    size_t len = vsnprintf(NULL, 0, format.c_str(), args);
    va_end(args);
    vector<char> vec(len+1);
    va_start(args, format);
    vsnprintf(&vec[0], len+1, format.c_str(), args);
    va_end(args);
    return &vec[0];
}

template<class T=float, class U=int>
vector<T> slice_vector(vector<T> vec, U start, U stop, U step=1){
    /* This function will partition a vector. This is the equivalent to
     * the python example:
     * >> vec[start:stop+1]  # note that the +1 is added because python doesn't
     * # include the last index.
     */

   if (step == float(0) || int(0)) {
      throw invalid_argument("step for range must be non-zero");
    }

    int size = vec.size();
    vector<T> v;
    // You can't slice nothing...
    if (start == stop) {
        throw invalid_argument("Slice Error: must have a valid range");
    }
    while(1) {
        //if (start >= stop+step) {
        if (start >= stop) {
            break;
        }
        v.push_back(vec[start]);
        start += step;
    }
    return v;
}


/*}}}*/

// Autocorrelation Method:{{{
//vector<float> traj,
vector< vector<float>> c_autocorrelation(vector< vector<float>> sampled_parameters,
        int maxtau=10000, bool normalize=true) {

    /*Calculates the autocorrelation
     *
     *
     * :para vector<float> :
     * :para vector<float> :
     * :para vector<float> sampled_parameters:
     * :para vector<float> traj: trajectory traces
     */

    printf("Calculating autocorrelation...\n");
    //int maxtau = 10000;
    vector< vector<float>> result(sampled_parameters.size());

    for (int k=0; k<sampled_parameters.size(); k++) {
        vector<float> f = sampled_parameters[k];
        //vector<float> f = sampled_parameters

        float f_mean = accumulate( f.begin(), f.end(), 0.0)/f.size();
        //printf("f_mean = %f\n",f_mean);
        vector<float> f_zeroed;
        for (int i=0; i<f.size(); i++) {
            //cout << f[i]-f_mean << endl;
            f_zeroed.push_back(f[i]-f_mean);
        }

        int T = f_zeroed.size();
        //result = np.zeros(maxtau+1)
        //for tau in range(maxtau+1):
        //cout << "slice_vector = " << endl;
        //vector<float> X = {3.5, 0.2, 0.2, 1.4, 4.2, 1.3};
        //print_vec(slice_vector(X, 0, int(X.size()-2)));
        //exit(1);

        vector<float> _result;
        float norm;
        for (int tau=0; tau<(maxtau+1); tau++) {
            vector<float> series1 = slice_vector(f_zeroed, 0, int(f_zeroed.size()-tau));
            vector<float> series2 = slice_vector(f_zeroed, tau, int(f_zeroed.size()));
            //printf("\n\n series1 = \n");
            //print_vec(series1);
            //printf("\n\n series2 = \n");
            //print_vec(series2);
            //exit(1);
            if ((f_zeroed.size()-tau) != 0) {
                _result.push_back(inner_product(
                            series1.begin(),series1.end(),
                            series2.begin(),0.0)/(f_zeroed.size()-tau));
            series1.clear();
            series2.clear();
            }

            if (normalize) {
                if (tau==0) {
                    norm = _result[0];
                }
                _result[tau] = _result[tau]/norm;
            }

            //printf("result = \n");
            //print_vec(result);
            //printf("\n\n series1 = \n");
            //print_vec(series1);
            //printf("\n\n series2 = \n");
            //print_vec(series2);
            //exit(1);
        }
        result[k].swap(_result);

    }
    return result;
    //printf("size of result vector = %i\n",result.size());
    //printf("result = \n");
    //print_vec(result);
    //exit(1);

    // destructor: sampled_parameters.clear(); or better yet ~sampled_parameters;
}
//}}}


//int main() {
//
//
//    // Generate random number
//	double rnd = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
//    //std::cout << "Random Number, rnd = "<< rnd << std::endl;
//
//
//    vector< vector<float>> sampled_parameters = {{1.0, 2.0, 3.0},{1.1, 2.2, 3.3},{1.4, 4.2, 1.3}};
//
//    cout << "sampled_parameters = " << endl;
//    //print_2D_vec(sampled_parameters);
//
//    vector<vector<float>> result = c_autocorrelation(sampled_parameters,
//            sampled_parameters[0].size());
//    printf("Done!\nResult = \n");
//    print_2D_vec(result);
//
//
//    return 0;
//}










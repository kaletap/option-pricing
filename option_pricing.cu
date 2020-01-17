#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <curand.h>
#include <iostream>

#define S_0 98
#define TIME 1
#define SIGMA 0.2
#define R 0.05
#define N_MAX 50000
#define K 100
//#define double float
#define DEFAULT_N 50000

using namespace std;

struct OptionPrice {
    double r;
    double v;
    double t;
    double s;
    double k;
    OptionPrice(
        double _underlying_price, 
        double _interest_rate, 
        double _sigma, 
        double _time_to_expiry,
        double _strike_price) {
            s = _underlying_price;
            r = _interest_rate;
            v = _sigma;
            t = _time_to_expiry;
            k = _strike_price;
    }
    __device__ double operator()(const double &std_normal_variable) const {
        double asset_price 
            = s * exp((r - 0.5 * v*v) * t + v*sqrt(t)*std_normal_variable);
        return exp(-r*t) * max(0.0, asset_price - k);
    }
};

struct SquaredError {
    double mean;
    SquaredError(const double _mean) {
        mean = _mean;
    }
    __host__ __device__ double operator()(const double x) const {
        return (x - mean)*(x - mean);
    }
};

int main(int argc, char *argv[]) {
    size_t n;
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    else {
        n = DEFAULT_N;
    }
	// Allocate n doubles on host
    thrust::device_vector<double> d_data(n, 0);

    // Creating CURAND generator
    curandGenerator_t gen;
    curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT); 
    curandSetPseudoRandomGeneratorSeed(gen, 1234ll);

    // Generate points from random distribution
    double *d_data_ptr = thrust::raw_pointer_cast(&d_data[0]);
    curandGenerateNormalDouble(gen, d_data_ptr, n, 0.0, 1.0);

    // Calculate 
    OptionPrice option_price(S_0, R, SIGMA, TIME, K);
    thrust::transform(d_data.begin(), d_data.end(), d_data.begin(), option_price);
    double sum =
        thrust::reduce(d_data.begin(), d_data.end(), 0.0, thrust::plus<double>());
    double mean = sum / n;
    double squared_error =  
        thrust::transform_reduce(d_data.begin(), d_data.end(),
            SquaredError(mean), 0.0, thrust::plus<double>());
    double standard_deviation = sqrt(squared_error / n - 1);

    cout << "First 10 profits:" << endl;
    for (int i = 0; i < 10; ++i) {
        cout << d_data[i] << " ";
    }
    cout << endl;

    cout << "Proft mean of " << n << " observations is " 
        << mean << " with standard deviation of " << standard_deviation << endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
using namespace std;

class MLE {
public:
    MLE(const vector<double>& data) : data(data) {}
    void compute() {
        // Compute mean (μ)
        double sum = 0.0;
        for (double x : data) {
            sum += x;
        }
        mu = sum / data.size();

        // Compute variance (σ²)
        double squared_sum = 0.0;
        for (double x : data) {
            squared_sum += (x - mu) * (x - mu);
        }
        sigma2 = squared_sum / data.size();
    }
    void print_data(){
        cout << "Estimated mean (mu): "<<mu<<endl;
        cout << "Estimated variance (sigma^2): "<<sigma2<<endl;
    }

private:
    vector<double> data;
    double mu = 0.0;
    double sigma2 = 0.0;
};

int main() {
    vector<double> data1 = {0.42,-0.2,1.3,0.39,-1.6,-0.029,-0.23,0.27,-1.9,0.87};
    vector<double> data2 = {-0.087,-3.3,-0.32,0.71,-5.3,0.89,1.9,-0.3,0.76,-1.0};
    vector<double> data3 = {0.58,-3.4,1.7,0.23,-0.15,-4.7,2.2,-0.87,-2.1,-2.6};

    MLE mle1(data1),mle2(data2),mle3(data3);

    mle1.compute();
    mle1.print_data();
    mle2.compute();
    mle2.print_data();
    mle3.compute();
    mle3.print_data();
    return 0;
}

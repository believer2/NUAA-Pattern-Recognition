#include <iostream>
#include <vector>
#include <array>
#include <cmath>

using namespace std;

class SeparableGaussianMLE {
public:
    SeparableGaussianMLE(const vector<array<double, 3>>& data) : data(data) {}

    void compute() {
        // Compute mean vector (μ)
        double sum_x = 0.0, sum_y = 0.0, sum_z = 0.0;
        for (const auto& point : data) {
            sum_x += point[0];
            sum_y += point[1];
            sum_z += point[2];
        }
        mu[0] = sum_x / data.size();
        mu[1] = sum_y / data.size();
        mu[2] = sum_z / data.size();

        // Compute variance parameters (σ^2)
        double var_x = 0.0, var_y = 0.0, var_z = 0.0;
        for (const auto& point : data) {
            var_x += (point[0] - mu[0]) * (point[0] - mu[0]);
            var_y += (point[1] - mu[1]) * (point[1] - mu[1]);
            var_z += (point[2] - mu[2]) * (point[2] - mu[2]);
        }
        sigma2[0] = var_x / data.size();
        sigma2[1] = var_y / data.size();
        sigma2[2] = var_z / data.size();
    }

    array<double, 3> getMu() const {
        return mu;
    }

    array<double, 3> getSigma2() const {
        return sigma2;
    }

private:
    vector<array<double, 3>> data;
    array<double, 3> mu = {0.0, 0.0, 0.0};
    array<double, 3> sigma2 = {0.0, 0.0, 0.0};
};

int main() {
    vector<array<double, 3>> data = {
        {-0.4,0.58,0.089},{-0.31,0.27,-0.04},{0.38,0.055,-0.035},{-0.15,0.53,0.011},{-0.35,0.47,0.034},
        {0.17,0.69,0.1},{-0.011,0.55,-0.18},{-0.27,0.61,0.12},{-0.065,0.49,0.0012},{-0.12,0.054,-0.063}
    }; 

    SeparableGaussianMLE mle(data);
    mle.compute();

    auto mu = mle.getMu();
    auto sigma2 = mle.getSigma2();

    cout << "Estimated mean vector (mu): [" << mu[0] << ", " << mu[1] << ", " << mu[2] << "]" << endl;
    cout << "Estimated variance parameters (sigma^2): [" << sigma2[0] << ", " << sigma2[1] << ", " << sigma2[2] << "]" << endl;

    return 0;
}

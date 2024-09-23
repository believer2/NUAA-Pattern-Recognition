#include <iostream>
#include <vector>
#include <array>

using namespace std;

class MLE {
public:
    MLE(const vector<array<double, 3>>& data) : data(data) {}

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

        // Compute covariance matrix (Σ)
        double var_xx = 0.0, var_yy = 0.0, var_zz = 0.0;
        double var_xy = 0.0, var_xz = 0.0, var_yz = 0.0;
        for (const auto& point : data) {
            double diff_x = point[0] - mu[0];
            double diff_y = point[1] - mu[1];
            double diff_z = point[2] - mu[2];
            var_xx += diff_x * diff_x;
            var_yy += diff_y * diff_y;
            var_zz += diff_z * diff_z;
            var_xy += diff_x * diff_y;
            var_xz += diff_x * diff_z;
            var_yz += diff_y * diff_z;
        }
        sigma[0][0] = var_xx / data.size();
        sigma[1][1] = var_yy / data.size();
        sigma[2][2] = var_zz / data.size();
        sigma[0][1] = sigma[1][0] = var_xy / data.size();
        sigma[0][2] = sigma[2][0] = var_xz / data.size();
        sigma[1][2] = sigma[2][1] = var_yz / data.size();
    }

    array<double, 3> getMu() const {
        return mu;
    }

    array<array<double, 3>, 3> getSigma() const {
        return sigma;
    }

private:
    vector<array<double, 3>> data;
    array<double, 3> mu = {0.0, 0.0, 0.0};
    array<array<double, 3>, 3> sigma = {{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}}};
};

int main() {
    vector<array<double, 3>> data = {
         {0.42,-0.087,0.58},{-0.2,-3.3,-3.4},{1.3,-0.32,1.7},{0.39,0.71,0.23},{-1.6,-5.3,-0.15},
         {-0.029,0.89,-4.7},{-0.23,1.9,2.2},{0.27,-0.3,-0.87},{-1.9,0.76,-2.1},{0.87,-1.0,-2.6}
    }; 

    MLE mle(data);
    mle.compute();

    auto mu = mle.getMu();
    auto sigma = mle.getSigma();

    cout << "Estimated mean vector (mu): [" << mu[0] << ", " << mu[1] << ", " << mu[2] << "]" << endl;
    cout << "Estimated covariance matrix (sigma):\n";
    cout << "[" << sigma[0][0] << ", " << sigma[0][1] << ", " << sigma[0][2] << "]\n";
    cout << "[" << sigma[1][0] << ", " << sigma[1][1] << ", " << sigma[1][2] << "]\n";
    cout << "[" << sigma[2][0] << ", " << sigma[2][1] << ", " << sigma[2][2] << "]" << endl;

    return 0;
}

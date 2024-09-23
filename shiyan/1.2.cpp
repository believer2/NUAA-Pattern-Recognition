#include <iostream>
#include <vector>
#include <array>

using namespace std;

class MLE {
public:
    MLE(const vector<array<double, 2>>& data) : data(data) {}

    void compute() {
        // Compute mean vector (μ)
        double sum_x = 0.0, sum_y = 0.0;
        for (const auto& point : data) {
            sum_x += point[0];
            sum_y += point[1];
        }
        mu[0] = sum_x / data.size();
        mu[1] = sum_y / data.size();

        // Compute covariance matrix (Σ)
        double var_xx = 0.0, var_yy = 0.0, var_xy = 0.0;
        for (const auto& point : data) {
            double diff_x = point[0] - mu[0];
            double diff_y = point[1] - mu[1];
            var_xx += diff_x * diff_x;
            var_yy += diff_y * diff_y;
            var_xy += diff_x * diff_y;
        }
        sigma[0][0] = var_xx / data.size();
        sigma[1][1] = var_yy / data.size();
        sigma[0][1] = sigma[1][0] = var_xy / data.size();
    }

    array<double, 2> getMu() const {
        return mu;
    }

    array<array<double, 2>, 2> getSigma() const {
        return sigma;
    }

private:
    vector<array<double, 2>> data;
    array<double, 2> mu = {0.0, 0.0};
    array<array<double, 2>, 2> sigma = {0.0, 0.0, 0.0, 0.0};
};

int main() {
    vector<array<double, 2>> data = {
        {0.42,-0.087},{-0.2,-3.3},{1.3,-0.32},{0.39,0.71},{-1.6,-5.3},{-0.029,0.89},{-0.23,1.9},{0.27,-0.3},{-1.9,0.76},{0.87,-1.0}
    };

    MLE mle(data);
    mle.compute();

    auto mu = mle.getMu();
    auto sigma = mle.getSigma();

    cout << "Estimated mean vector (mu): [" << mu[0] << ", " << mu[1] << "]" << endl;
    cout << "Estimated covariance matrix (sigma):\n";
    cout << "[" << sigma[0][0] << ", " << sigma[0][1] << "]\n";
    cout << "[" << sigma[1][0] << ", " << sigma[1][1] << "]" << endl;

    return 0;
}

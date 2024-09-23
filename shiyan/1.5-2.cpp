#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace cv;
using namespace std;

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

Matrix invertMatrix(const Matrix& matrix) {
    int n = matrix.size();
    Matrix inverse(n, Vector(n, 0.0));
    Mat cvMatrix(n, n, CV_64F);
    Mat cvInverse(n, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            cvMatrix.at<double>(i, j) = matrix[i][j];
        }
    }
    cv::invert(cvMatrix, cvInverse);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            inverse[i][j] = cvInverse.at<double>(i, j);
        }
    }
    return inverse;
}

Vector fisherLinearDiscriminant(const Matrix& class1Data, const Matrix& class2Data) {
    Vector mean1(class1Data[0].size(), 0.0);
    Vector mean2(class2Data[0].size(), 0.0);
    int numSamples1 = class1Data.size();
    int numSamples2 = class2Data.size();

    for (const auto& sample : class1Data) {
        for (size_t i = 0; i < mean1.size(); ++i) {
            mean1[i] += sample[i] / numSamples1;
        }
    }

    for (const auto& sample : class2Data) {
        for (size_t i = 0; i < mean2.size(); ++i) {
            mean2[i] += sample[i] / numSamples2;
        }
    }

    Matrix Sw(mean1.size(), Vector(mean1.size(), 0.0));
    for (const auto& sample : class1Data) {
        for (size_t i = 0; i < mean1.size(); ++i) {
            for (size_t j = 0; j < mean1.size(); ++j) {
                Sw[i][j] += (sample[i] - mean1[i]) * (sample[j] - mean1[j]) / numSamples1;
            }
        }
    }

    for (const auto& sample : class2Data) {
        for (size_t i = 0; i < mean2.size(); ++i) {
            for (size_t j = 0; j < mean2.size(); ++j) {
                Sw[i][j] += (sample[i] - mean2[i]) * (sample[j] - mean2[j]) / numSamples2;
            }
        }
    }

    Matrix SwInv = invertMatrix(Sw);

    Vector w(mean1.size(), 0.0);
    for (size_t i = 0; i < mean1.size(); ++i) {
        for (size_t j = 0; j < mean1.size(); ++j) {
            w[i] += SwInv[i][j] * (mean2[j] - mean1[j]);
        }
    }

    double norm = 0.0;
    for (double component : w) {
        norm += component * component;
    }
    norm = sqrt(norm);
    for (double& component : w) {
        component /= norm;
    }

    return w;
}

double projectPoint(const Vector& point, const Vector& direction) {
    double projectionLength = 0.0;
    for (size_t i = 0; i < point.size(); ++i) {
        projectionLength += point[i] * direction[i];
    }
    return projectionLength;
}

double gaussianPDF(double x, double mean, double variance) {
    double exponent = -0.5 * pow((x - mean) / sqrt(variance), 2);
    return exp(exponent) / sqrt(2 * M_PI * variance);
}

pair<double, double> meanAndVariance(const vector<double>& data) {
    double mean = 0.0;
    for (double value : data) {
        mean += value;
    }
    mean /= data.size();

    double variance = 0.0;
    for (double value : data) {
        variance += pow(value - mean, 2);
    }
    variance /= data.size();

    return make_pair(mean, variance);
}

int main() {
    Matrix class1Data = {
        {-0.4, 0.58, 0.089}, {-0.31, 0.27, -0.04}, {0.38, 0.055, -0.035}, 
        {-0.15, 0.53, 0.011}, {-0.35, 0.47, 0.034}, {0.17, 0.69, 0.1}, 
        {-0.011, 0.55, -0.18}, {-0.27, 0.61, 0.12}, {-0.065, 0.49, 0.0012}, 
        {-0.12, 0.054, -0.063}
    };
    Matrix class2Data = {
        {0.83, 1.6, -0.014}, {1.1, 1.6, 0.48}, {-0.44, -0.41, 0.32}, 
        {0.047, -0.45, 1.4}, {0.28, 0.35, 3.1}, {-0.39, -0.48, 0.11}, 
        {0.34, -0.079, 0.14}, {-0.3, -0.22, 2.2}, {1.1, 1.2, -0.46}, 
        {0.18, -0.11, -0.49}
    };

    Vector fisherW = fisherLinearDiscriminant(class1Data, class2Data);
    cout << "由 Fisher 线性判别计算出来的最优方向为:" << endl;
    for (double val : fisherW) {
        cout << val << " ";
    }
    cout << endl;

    vector<double> class1Projections, class2Projections;
    for (const auto& point : class1Data) {
        class1Projections.push_back(projectPoint(point, fisherW));
    }
    for (const auto& point : class2Data) {
        class2Projections.push_back(projectPoint(point, fisherW));
    }

    auto [mean1, var1] = meanAndVariance(class1Projections);
    auto [mean2, var2] = meanAndVariance(class2Projections);

    double decisionBoundary = (mean1 * var2 - mean2 * var1) / (var2 - var1);
    cout << "Fisher线性判别的决策边界为: " << decisionBoundary << endl;

    int errorCount = 0;
    for (double value : class1Projections) {
        if (value > decisionBoundary) {
            ++errorCount;
        }
    }
    for (double value : class2Projections) {
        if (value < decisionBoundary) {
            ++errorCount;
        }
    }
    double errorRate = static_cast<double>(errorCount) / (class1Projections.size() + class2Projections.size());
    cout << "Fisher线性判别的误差率为: " << errorRate << endl;

    Vector nonOptimalW = {1.4, 2.7, -1.2};
    vector<double> nonOptimalProjections;
    for (const auto& point : class1Data) {
        nonOptimalProjections.push_back(projectPoint(point, nonOptimalW));
    }
    for (const auto& point : class2Data) {
        nonOptimalProjections.push_back(projectPoint(point, nonOptimalW));
    }

    auto [nonOptimalMean1, nonOptimalVar1] = meanAndVariance(nonOptimalProjections);

    double nonOptimalDecisionBoundary = (mean1 * nonOptimalVar1 - nonOptimalMean1 * var1) / (nonOptimalVar1 - var1);
    cout << "非最优方向的决策边界为: " << nonOptimalDecisionBoundary << endl;

    errorCount = 0;
    for (double value : nonOptimalProjections) {
        if (value > nonOptimalDecisionBoundary && value < decisionBoundary) {
            ++errorCount;
        }
    }
    errorRate = static_cast<double>(errorCount) / (class1Projections.size() + class2Projections.size());
    cout << "非最优方向的误差率为: " << errorRate << endl;

    return 0;
}

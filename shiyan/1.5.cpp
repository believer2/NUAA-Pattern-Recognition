#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef vector<double> Vector;
typedef vector<Vector> Matrix;

Vector calculateMean(const Matrix& data) {
    Vector mean(data[0].size(), 0.0);
    int numSamples = data.size();
    for (const auto& sample : data) {
        for (size_t i = 0; i < mean.size(); ++i) {
            mean[i] += sample[i] / numSamples;
        }
    }
    return mean;
}

Matrix calculateCovarianceMatrix(const Matrix& data, const Vector& mean) {
    Matrix covarianceMatrix(mean.size(), Vector(mean.size(), 0.0));
    int numSamples = data.size();
    for (const auto& sample : data) {
        for (size_t i = 0; i < mean.size(); ++i) {
            for (size_t j = 0; j < mean.size(); ++j) {
                covarianceMatrix[i][j] += (sample[i] - mean[i]) * (sample[j] - mean[j]) / numSamples;
            }
        }
    }
    return covarianceMatrix;
}

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
    Vector mean1 = calculateMean(class1Data);
    Vector mean2 = calculateMean(class2Data);
    Matrix Sw(mean1.size(), Vector(mean1.size(), 0.0));
    Matrix class1Covariance = calculateCovarianceMatrix(class1Data, mean1);
    Matrix class2Covariance = calculateCovarianceMatrix(class2Data, mean2);
    for (size_t i = 0; i < mean1.size(); ++i) {
        for (size_t j = 0; j < mean1.size(); ++j) {
            Sw[i][j] = class1Covariance[i][j] + class2Covariance[i][j];
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

double projectPointToLine(const Vector& point, const Vector& direction) {
    double projectionLength = 0.0;
    for (size_t i = 0; i < point.size(); ++i) {
        projectionLength += point[i] * direction[i];
    }
    return projectionLength;
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

    Vector w = fisherLinearDiscriminant(class1Data, class2Data);
    cout << "由 Fisher 线性判别计算出来的最优方向为:" << endl;
    for (double val : w) {
        cout << val << " ";
    }
    cout << endl;

    Vector class1Projections, class2Projections;
    for (const auto& point : class1Data) {
        class1Projections.push_back(projectPointToLine(point, w));
    }
    for (const auto& point : class2Data) {
        class2Projections.push_back(projectPointToLine(point, w));
    }

    int width = 800, height = 600;
    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));

    for (const auto& point : class1Data) {
        circle(image, Point(100 + static_cast<int>(point[0] * 100), 500 - static_cast<int>(point[1] * 100)), 5, Scalar(0, 0, 255), -1);
    }
    for (const auto& point : class2Data) {
        circle(image, Point(100 + static_cast<int>(point[0] * 100), 500 - static_cast<int>(point[1] * 100)), 5, Scalar(255, 0, 0), -1);
    }

    Point origin(100, 500);
    double scale = 200.0; 
    Point endPoint(
        static_cast<int>(origin.x + w[0] * scale),
        static_cast<int>(origin.y - w[1] * scale)
    );
    Point extendedStart(
        static_cast<int>(origin.x - w[0] * scale),
        static_cast<int>(origin.y + w[1] * scale)
    );

    line(image, extendedStart, endPoint, Scalar(0, 0, 0), 2);

    for (double projection : class1Projections) {
        Point projPoint(
            static_cast<int>(origin.x + projection * w[0] * scale),
            static_cast<int>(origin.y - projection * w[1] * scale)
        );
        circle(image, projPoint, 5, Scalar(0, 0, 255), -1);
    }
    for (double projection : class2Projections) {
        Point projPoint(
            static_cast<int>(origin.x + projection * w[0] * scale),
            static_cast<int>(origin.y - projection * w[1] * scale)
        );
        circle(image, projPoint, 5, Scalar(255, 0, 0), -1);
    }

    imshow("Fisher Linear Discriminant", image);
    waitKey(0);

    return 0;
}

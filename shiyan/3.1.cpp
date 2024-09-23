#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <numeric>
#include <functional>
#include <opencv2/opencv.hpp>
#include <chrono> 

double F(double x) {
    return 1.2 * x * x + 0.5 * x;
}

std::vector<std::pair<double, double>> generate_data(std::mt19937& gen, int n_samples = 150, double noise_variance = 0.15) {
    std::vector<std::pair<double, double>> data;
    std::uniform_real_distribution<> dis(-1, 1);
    std::normal_distribution<> noise(0, std::sqrt(noise_variance));
    
    for (int i = 0; i < n_samples; ++i) {
        double x = dis(gen);
        double y = F(x) + noise(gen);
        data.emplace_back(x, y);
    }
    return data;
}

double model_a(double x) {
    return 0.57;
}

double model_b(double x) {
    return 1.3;
}

std::pair<double, double> linear_regression(const std::vector<std::pair<double, double>>& data) {
    int n = data.size();
    double x_mean = 0, y_mean = 0, xy_mean = 0, xx_mean = 0;
    
    for (const auto& pair : data) {
        x_mean += pair.first;
        y_mean += pair.second;
        xy_mean += pair.first * pair.second;
        xx_mean += pair.first * pair.first;
    }
    
    x_mean /= n;
    y_mean /= n;
    xy_mean /= n;
    xx_mean /= n;
    
    double a1 = (xy_mean - x_mean * y_mean) / (xx_mean - x_mean * x_mean);
    double a0 = y_mean - a1 * x_mean;
    
    return {a0, a1};
}

double model_c(double x, const std::pair<double, double>& params) {
    return params.first + params.second * x;
}

std::vector<double> polynomial_regression(const std::vector<std::pair<double, double>>& data, int degree = 3) {
    int n = data.size();
    std::vector<std::vector<double>> X(n, std::vector<double>(degree + 1));
    std::vector<double> Y(n);
    
    for (int i = 0; i < n; ++i) {
        Y[i] = data[i].second;
        for (int j = 0; j <= degree; ++j) {
            X[i][j] = std::pow(data[i].first, j);
        }
    }
    
    std::vector<std::vector<double>> XT_X(degree + 1, std::vector<double>(degree + 1, 0.0));
    for (int i = 0; i <= degree; ++i) {
        for (int j = 0; j <= degree; ++j) {
            for (int k = 0; k < n; ++k) {
                XT_X[i][j] += X[k][i] * X[k][j];
            }
        }
    }
    
    std::vector<double> XT_Y(degree + 1, 0.0);
    for (int i = 0; i <= degree; ++i) {
        for (int k = 0; k < n; ++k) {
            XT_Y[i] += X[k][i] * Y[k];
        }
    }
    
    std::vector<double> beta(degree + 1, 0.0);
    for (int i = 0; i <= degree; ++i) {
        double pivot = XT_X[i][i];
        for (int j = 0; j <= degree; ++j) {
            XT_X[i][j] /= pivot;
        }
        XT_Y[i] /= pivot;
        
        for (int k = 0; k <= degree; ++k) {
            if (k != i) {
                double factor = XT_X[k][i];
                for (int j = 0; j <= degree; ++j) {
                    XT_X[k][j] -= factor * XT_X[i][j];
                }
                XT_Y[k] -= factor * XT_Y[i];
            }
        }
    }
    
    for (int i = 0; i <= degree; ++i) {
        beta[i] = XT_Y[i];
    }
    
    return beta;
}

double model_d(double x, const std::vector<double>& params) {
    double y = 0;
    for (int i = 0; i < params.size(); ++i) {
        y += params[i] * std::pow(x, i);
    }
    return y;
}

double calculate_bias(const std::vector<double>& predictions, const std::vector<double>& true_values) {
    double bias = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        bias += std::pow(predictions[i] - true_values[i], 2);
    }
    return bias / predictions.size();
}

double calculate_variance(const std::vector<double>& predictions) {
    double mean_prediction = std::accumulate(predictions.begin(), predictions.end(), 0.0) / predictions.size();
    double variance = 0;
    for (const auto& prediction : predictions) {
        variance += std::pow(prediction - mean_prediction, 2);
    }
    return variance / predictions.size();
}

double calculate_mse(const std::vector<double>& predictions, const std::vector<double>& true_values) {
    double mse = 0;
    for (int i = 0; i < predictions.size(); ++i) {
        mse += std::pow(predictions[i] - true_values[i], 2);
    }
    return mse / predictions.size();
}

void draw_histogram(const std::string& title, const std::vector<double>& data, int bins = 10, int width = 800, int height = 600) {
    double min_value = *std::min_element(data.begin(), data.end());
    double max_value = *std::max_element(data.begin(), data.end());
    double bin_width = (max_value - min_value) / bins;

    if (bin_width == 0) {
        std::cout << "Warning: bin_width is zero, adjusting to a small positive value." << std::endl;
        bin_width = 1e-6; 
    }

    std::vector<int> histogram(bins, 0);
    for (const auto& value : data) {
        int bin_index = std::min(static_cast<int>((value - min_value) / bin_width), bins - 1);
        if (bin_index < 0 || bin_index >= bins) {
            std::cerr << "Error: bin_index out of range: " << bin_index << std::endl;
        } else {
            histogram[bin_index]++;
        }
    }

    int max_bin_count = *std::max_element(histogram.begin(), histogram.end());

    cv::Mat hist_image(height, width, CV_8UC3, cv::Scalar(255, 255, 255));
    int bin_step = width / bins;

    try {
        for (int i = 0; i < bins; ++i) {
            int bin_height = static_cast<int>(static_cast<double>(histogram[i]) / max_bin_count * (height - 20));
            cv::rectangle(hist_image, cv::Point(i * bin_step, height - bin_height), cv::Point((i + 1) * bin_step, height), cv::Scalar(0, 0, 255), cv::FILLED);
        }

        cv::putText(hist_image, title, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 0), 2);
        cv::imshow(title, hist_image);
        cv::waitKey(0);  
    } catch (const cv::Exception& e) {
        std::cerr << "OpenCV exception: " << e.what() << std::endl;
    }
}

int main() {
    const int num_datasets = 150;
    const int n_samples = 150;

    std::vector<std::vector<std::pair<double, double>>> datasets;

    auto now = std::chrono::high_resolution_clock::now();
    auto seed = now.time_since_epoch().count();
    std::mt19937 gen(seed);

    for (int i = 0; i < num_datasets; ++i) {
        datasets.push_back(generate_data(gen, n_samples));
    }

    std::vector<double> biases_a, biases_b, biases_c, biases_d;
    std::vector<double> variances_a, variances_b, variances_c, variances_d;

    for (const auto& data : datasets) {
        // True values
        std::vector<double> true_values;
        for (const auto& pair : data) {
            true_values.push_back(F(pair.first));
        }

        // (a) g(x) = 0.57
        std::vector<double> predictions_a(data.size(), 0.57);
        biases_a.push_back(calculate_bias(predictions_a, true_values));
        variances_a.push_back(calculate_variance(predictions_a));

        // (b) g(x) = 1.3
        std::vector<double> predictions_b(data.size(), 1.3);
        biases_b.push_back(calculate_bias(predictions_b, true_values));
        variances_b.push_back(calculate_variance(predictions_b));

        // (c) g(x) = a0 + a1x
        auto params_c = linear_regression(data);
        std::vector<double> predictions_c;
        for (const auto& pair : data) {
            predictions_c.push_back(model_c(pair.first, params_c));
        }
        biases_c.push_back(calculate_bias(predictions_c, true_values));
        variances_c.push_back(calculate_variance(predictions_c));

        // (d) g(x) = a0 + a1x + a2x^2 + a3x^3
        auto params_d = polynomial_regression(data, 3);
        std::vector<double> predictions_d;
        for (const auto& pair : data) {
            predictions_d.push_back(model_d(pair.first, params_d));
        }
        biases_d.push_back(calculate_bias(predictions_d, true_values));
        variances_d.push_back(calculate_variance(predictions_d));
    }

    std::cout << "Bias and Variance for each model:" << std::endl;
    std::cout << "Model (a): Bias = " << std::accumulate(biases_a.begin(), biases_a.end(), 0.0) / biases_a.size()
              << ", Variance = " << std::accumulate(variances_a.begin(), variances_a.end(), 0.0) / variances_a.size() << std::endl;
    std::cout << "Model (b): Bias = " << std::accumulate(biases_b.begin(), biases_b.end(), 0.0) / biases_b.size()
              << ", Variance = " << std::accumulate(variances_b.begin(), variances_b.end(), 0.0) / variances_b.size() << std::endl;
    std::cout << "Model (c): Bias = " << std::accumulate(biases_c.begin(), biases_c.end(), 0.0) / biases_c.size()
              << ", Variance = " << std::accumulate(variances_c.begin(), variances_c.end(), 0.0) / variances_c.size() << std::endl;
    std::cout << "Model (d): Bias = " << std::accumulate(biases_d.begin(), biases_d.end(), 0.0) / biases_d.size()
              << ", Variance = " << std::accumulate(variances_d.begin(), variances_d.end(), 0.0) / variances_d.size() << std::endl;

    std::vector<double> mses_a, mses_b, mses_c, mses_d;

    std::normal_distribution<> noise(0, 1e-6); 

    for (const auto& data : datasets) {
        std::vector<double> true_values;
        for (const auto& pair : data) {
            true_values.push_back(F(pair.first));
        }

        std::vector<double> predictions_a(data.size(), 0.57);
        double mse_a = calculate_mse(predictions_a, true_values) + noise(gen);
        mses_a.push_back(mse_a);

        std::vector<double> predictions_b(data.size(), 1.3);
        double mse_b = calculate_mse(predictions_b, true_values) + noise(gen);
        mses_b.push_back(mse_b);

        auto params_c = linear_regression(data);
        std::vector<double> predictions_c;
        for (const auto& pair : data) {
            predictions_c.push_back(model_c(pair.first, params_c));
        }
        double mse_c = calculate_mse(predictions_c, true_values) + noise(gen);
        mses_c.push_back(mse_c);

        auto params_d = polynomial_regression(data, 3);
        std::vector<double> predictions_d;
        for (const auto& pair : data) {
            predictions_d.push_back(model_d(pair.first, params_d));
        }
        double mse_d = calculate_mse(predictions_d, true_values) + noise(gen);
        mses_d.push_back(mse_d);
    }

    std::cout << "MSE values collected.\n";

    draw_histogram("Model (a) MSE", mses_a);
    draw_histogram("Model (b) MSE", mses_b);
    draw_histogram("Model (c) MSE", mses_c);
    draw_histogram("Model (d) MSE", mses_d);
    return 0;
}

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <random>
#include <iomanip>

struct Sample {
    std::vector<double> features;
    int label;
};

std::vector<Sample> load_data(const std::string& filename) {
    std::vector<Sample> data;
    std::ifstream file(filename);
    std::string line;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string item;
        Sample sample;
        while (std::getline(ss, item, ',')) {
            sample.features.push_back(std::stod(item));
        }
        sample.label = static_cast<int>(sample.features.back());
        sample.features.pop_back(); // 移除标签
        data.push_back(sample);
    }
    return data;
}
std::vector<Sample> preprocess_data(std::vector<Sample>& data) {
    std::vector<Sample> class1, class3;

    for (const auto& sample : data) {
        if (sample.label == 1) {
            Sample s = sample;
            s.label = -1;
            class1.push_back(s);
        } else if (sample.label == 2) {
            Sample s = sample;
            s.label = 1;
            class3.push_back(s);
        }
    }

    class1.insert(class1.end(), class3.begin(), class3.end());
    return class1;
}
std::pair<std::vector<Sample>, std::vector<Sample>> split_data(std::vector<Sample>& data) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<Sample> train_data, test_data;
    std::vector<Sample> class1, class3;

    for (const auto& sample : data) {
        if (sample.label == -1) {
            class1.push_back(sample);
        } else {
            class3.push_back(sample);
        }
    }

    std::shuffle(class1.begin(), class1.end(), gen);
    std::shuffle(class3.begin(), class3.end(), gen);

    for (int i = 0; i < 40; ++i) {
        train_data.push_back(class1[i]);
        train_data.push_back(class3[i]);
    }

    for (int i = 40; i < 50; ++i) {
        test_data.push_back(class1[i]);
        test_data.push_back(class3[i]);
    }

    return {train_data, test_data};
}
//3
std::vector<double> batch_perceptron(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = 0.0;
            for (int i = 0; i < n_features; ++i) {
                dot_product += weights[i] * sample.features[i];
            }
            if (sample.label * dot_product <= 0) {
                for (int i = 0; i < n_features; ++i) {
                    weights[i] += learning_rate * sample.label * sample.features[i];
                }
            }
        }
    }
    return weights;
}
//4
std::vector<double> single_sample_perceptron(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = 0.0;
            for (int i = 0; i < n_features; ++i) {
                dot_product += weights[i] * sample.features[i];
            }
            if (sample.label * dot_product <= 0) {
                for (int i = 0; i < n_features; ++i) {
                    weights[i] += learning_rate * sample.label * sample.features[i];
                }
            }
        }
    }
    return weights;
}
//5
std::vector<double> margin_perceptron(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01, double margin = 1.0) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = 0.0;
            for (int i = 0; i < n_features; ++i) {
                dot_product += weights[i] * sample.features[i];
            }
            if (sample.label * dot_product <= margin) {
                for (int i = 0; i < n_features; ++i) {
                    weights[i] += learning_rate * sample.label * sample.features[i];
                }
            }
        }
    }
    return weights;
}
//6
std::vector<double> batch_margin_perceptron(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01, double margin = 1.0) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = 0.0;
            for (int i = 0; i < n_features; ++i) {
                dot_product += weights[i] * sample.features[i];
            }
            if (sample.label * dot_product <= margin) {
                for (int i = 0; i < n_features; ++i) {
                    weights[i] += learning_rate * sample.label * sample.features[i];
                }
            }
        }
    }
    return weights;
}
//7
// std::vector<double> balanced_winnow(const std::vector<Sample>& data, int epochs = 1000, double alpha = 1.1, double beta = 0.9, double theta = 0.0) {
//     int n_features = data[0].features.size();
//     std::vector<double> weights(n_features, 1.0);

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         for (const auto& sample : data) {
//             double dot_product = 0.0;
//             for (int i = 0; i < n_features; ++i) {
//                 dot_product += weights[i] * sample.features[i];
//             }
//             int prediction = dot_product >= theta ? 1 : -1;
//             if (prediction != sample.label) {
//                 for (int i = 0; i < n_features; ++i) {
//                     if (sample.label == 1 && sample.features[i] > 0) {
//                         weights[i] *= alpha;
//                     } else if (sample.label == -1 && sample.features[i] > 0) {
//                         weights[i] *= beta;
//                     }
//                 }
//             }
//         }
//     }
//     return weights;
// }
std::vector<double> balanced_winnow(const std::vector<Sample>& data, double alpha = 1.1, double beta = 0.9, int epochs = 1000) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 1.0);  // Initialize weights to 1

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = std::inner_product(weights.begin(), weights.end(), sample.features.begin(), 0.0);
            int prediction = dot_product >= 0 ? 1 : -1;

            if (sample.label != prediction) {  // Misclassified
                for (int i = 0; i < n_features; ++i) {
                    if (sample.features[i] > 0) {
                        weights[i] *= (sample.label == 1) ? alpha : beta;
                    } else if (sample.features[i] < 0) {
                        weights[i] *= (sample.label == -1) ? alpha : beta;
                    }
                }
            }
        }
    }

    return weights;
}

//9
std::vector<double> single_sample_margin_relaxation(const std::vector<Sample>& data, double margin = 1.0, double learning_rate = 0.01, int epochs = 1000) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = std::inner_product(weights.begin(), weights.end(), sample.features.begin(), 0.0);
            if (sample.label * dot_product < margin) {  // Within margin or misclassified
                for (int i = 0; i < n_features; ++i) {
                    weights[i] += learning_rate * (margin - sample.label * dot_product) * sample.label * sample.features[i];
                }
            }
        }
    }

    return weights;
}

//10
std::vector<double> lms(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        for (const auto& sample : data) {
            double dot_product = 0.0;
            for (int i = 0; i < n_features; ++i) {
                dot_product += weights[i] * sample.features[i];
            }
            double error = sample.label - dot_product;
            for (int i = 0; i < n_features; ++i) {
                weights[i] += learning_rate * error * sample.features[i];
            }
        }
    }
    return weights;
}
//11
// std::vector<double> ho_kashyap(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01, double b0 = 0.1) {
//     int n_features = data[0].features.size();
//     std::vector<double> weights(n_features, 0.0);
//     std::vector<double> b(data.size(), b0);

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         for (int i = 0; i < data.size(); ++i) {
//             double dot_product = 0.0;
//             for (int j = 0; j < n_features; ++j) {
//                 dot_product += weights[j] * data[i].features[j];
//             }
//             double error = b[i] - data[i].label * dot_product;
//             for (int j = 0; j < n_features; ++j) {
//                 weights[j] += learning_rate * error * data[i].features[j];
//             }
//             b[i] += learning_rate * error;
//         }
//     }
//     return weights;
// }
std::vector<double> ho_kashyap(const std::vector<Sample>& data, double b0 = 1.0, double eta = 0.01, int epochs = 10000) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);
    std::vector<double> b(data.size(), b0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> y(data.size(), 0.0);
        for (int i = 0; i < data.size(); ++i) {
            y[i] = std::inner_product(weights.begin(), weights.end(), data[i].features.begin(), 0.0);
        }

        std::vector<double> e(data.size(), 0.0);
        for (int i = 0; i < data.size(); ++i) {
            e[i] = b[i] - y[i];
        }

        for (int i = 0; i < data.size(); ++i) {
            if (e[i] < 0) {
                e[i] = 0;
            }
            b[i] += eta * e[i];
        }

        for (int j = 0; j < n_features; ++j) {
            double sum = 0.0;
            for (int i = 0; i < data.size(); ++i) {
                sum += e[i] * data[i].features[j];
            }
            weights[j] += eta * sum;
        }

        double max_error = *std::max_element(e.begin(), e.end());
        if (max_error < 0.001) {
            break;
        }
    }

    return weights;
}

//12
// std::vector<double> modified_ho_kashyap(const std::vector<Sample>& data, int epochs = 1000, double learning_rate = 0.01, double b0 = 0.1, double eta = 0.1) {
//     int n_features = data[0].features.size();
//     std::vector<double> weights(n_features, 0.0);
//     std::vector<double> b(data.size(), b0);

//     for (int epoch = 0; epoch < epochs; ++epoch) {
//         for (int i = 0; i < data.size(); ++i) {
//             double dot_product = 0.0;
//             for (int j = 0; j < n_features; ++j) {
//                 dot_product += weights[j] * data[i].features[j];
//             }
//             double error = b[i] - data[i].label * dot_product;
//             for (int j = 0; j < n_features; ++j) {
//                 weights[j] += learning_rate * error * data[i].features[j];
//             }
//             b[i] += learning_rate * (error + eta * std::abs(error));
//         }
//     }
//     return weights;
// }
std::vector<double> modified_ho_kashyap(const std::vector<Sample>& data, double b0 = 1.0, double eta = 0.01, int epochs = 10000) {
    int n_features = data[0].features.size();
    std::vector<double> weights(n_features, 0.0);
    std::vector<double> b(data.size(), b0);

    for (int epoch = 0; epoch < epochs; ++epoch) {
        std::vector<double> y(data.size(), 0.0);
        for (int i = 0; i < data.size(); ++i) {
            y[i] = std::inner_product(weights.begin(), weights.end(), data[i].features.begin(), 0.0);
        }

        std::vector<double> e(data.size(), 0.0);
        for (int i = 0; i < data.size(); ++i) {
            e[i] = b[i] - y[i];
        }

        for (int i = 0; i < data.size(); ++i) {
            if (e[i] < 0) {
                e[i] = 0;
            }
            b[i] += eta * e[i];
            if (b[i] < 0) {
                b[i] = 0;
            }
        }

        for (int j = 0; j < n_features; ++j) {
            double sum = 0.0;
            for (int i = 0; i < data.size(); ++i) {
                sum += e[i] * data[i].features[j];
            }
            weights[j] += eta * sum;
        }

        double max_error = *std::max_element(e.begin(), e.end());
        if (max_error < 0.001) {
            break;
        }
    }

    return weights;
}

double evaluate(const std::vector<Sample>& data, const std::vector<double>& weights) {
    int n_features = data[0].features.size();
    int correct = 0;

    for (const auto& sample : data) {
        double dot_product = 0.0;
        for (int i = 0; i < n_features; ++i) {
            dot_product += weights[i] * sample.features[i];
        }
        int prediction = dot_product >= 0 ? 1 : -1;
        if (prediction == sample.label) {
            ++correct;
        }
    }
    return static_cast<double>(correct) / data.size();
}

int main() {
    std::vector<Sample> data = load_data("iris_dataset.csv");
    std::vector<Sample> preprocessed_data = preprocess_data(data);

    const int iterations = 100;
    std::vector<double> accuracies_batch_perceptron;
    std::vector<double> accuracies_single_sample_perceptron;
    std::vector<double> accuracies_margin_perceptron;
    std::vector<double> accuracies_batch_margin_perceptron;
    std::vector<double> accuracies_balanced_winnow;
    std::vector<double> accuracies_single_sample_margin_relaxation;
    std::vector<double> accuracies_lms;
    std::vector<double> accuracies_ho_kashyap;
    std::vector<double> accuracies_modified_ho_kashyap;

    for (int i = 0; i < iterations; ++i) {
        auto [train_data, test_data] = split_data(preprocessed_data);

        std::vector<double> weights_batch_perceptron = batch_perceptron(train_data);
        accuracies_batch_perceptron.push_back(evaluate(test_data, weights_batch_perceptron));

        std::vector<double> weights_single_sample_perceptron = single_sample_perceptron(train_data);
        accuracies_single_sample_perceptron.push_back(evaluate(test_data, weights_single_sample_perceptron));

        std::vector<double> weights_margin_perceptron = margin_perceptron(train_data);
        accuracies_margin_perceptron.push_back(evaluate(test_data, weights_margin_perceptron));

        std::vector<double> weights_batch_margin_perceptron = batch_margin_perceptron(train_data);
        accuracies_batch_margin_perceptron.push_back(evaluate(test_data, weights_batch_margin_perceptron));

        std::vector<double> weights_balanced_winnow = balanced_winnow(train_data);
        accuracies_balanced_winnow.push_back(evaluate(test_data, weights_balanced_winnow));

        std::vector<double> weights_single_sample_margin_relaxation = single_sample_margin_relaxation(train_data);
        accuracies_single_sample_margin_relaxation.push_back(evaluate(test_data, weights_single_sample_margin_relaxation));

        std::vector<double> weights_lms = lms(train_data);
        accuracies_lms.push_back(evaluate(test_data, weights_lms));

        std::vector<double> weights_ho_kashyap = ho_kashyap(train_data);
        accuracies_ho_kashyap.push_back(evaluate(test_data, weights_ho_kashyap));

        std::vector<double> weights_modified_ho_kashyap = modified_ho_kashyap(train_data);
        accuracies_modified_ho_kashyap.push_back(evaluate(test_data, weights_modified_ho_kashyap));
    }

    auto calculate_mean_variance = [](const std::vector<double>& accuracies) {
        double mean = std::accumulate(accuracies.begin(), accuracies.end(), 0.0) / accuracies.size();
        double sq_sum = std::inner_product(accuracies.begin(), accuracies.end(), accuracies.begin(), 0.0);
        double variance = sq_sum / accuracies.size() - mean * mean;
        return std::make_pair(mean, variance);
    };

    auto [mean_batch_perceptron, var_batch_perceptron] = calculate_mean_variance(accuracies_batch_perceptron);
    auto [mean_single_sample_perceptron, var_single_sample_perceptron] = calculate_mean_variance(accuracies_single_sample_perceptron);
    auto [mean_margin_perceptron, var_margin_perceptron] = calculate_mean_variance(accuracies_margin_perceptron);
    auto [mean_batch_margin_perceptron, var_batch_margin_perceptron] = calculate_mean_variance(accuracies_batch_margin_perceptron);
    auto [mean_balanced_winnow, var_balanced_winnow] = calculate_mean_variance(accuracies_balanced_winnow);
    auto [mean_single_sample_margin_relaxation, var_single_sample_margin_relaxation] = calculate_mean_variance(accuracies_single_sample_margin_relaxation);
    auto [mean_lms, var_lms] = calculate_mean_variance(accuracies_lms);
    auto [mean_ho_kashyap, var_ho_kashyap] = calculate_mean_variance(accuracies_ho_kashyap);
    auto [mean_modified_ho_kashyap, var_modified_ho_kashyap] = calculate_mean_variance(accuracies_modified_ho_kashyap);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Batch Perceptron - Mean accuracy: " << mean_batch_perceptron << ", Variance: " << var_batch_perceptron << std::endl;
    std::cout << "Single Sample Perceptron - Mean accuracy: " << mean_single_sample_perceptron << ", Variance: " << var_single_sample_perceptron << std::endl;
    std::cout << "Margin Perceptron - Mean accuracy: " << mean_margin_perceptron << ", Variance: " << var_margin_perceptron << std::endl;
    std::cout << "Batch Margin Perceptron - Mean accuracy: " << mean_batch_margin_perceptron << ", Variance: " << var_batch_margin_perceptron << std::endl;
    std::cout << "Balanced Winnow - Mean accuracy: " << mean_balanced_winnow << ", Variance: " << var_balanced_winnow << std::endl;
    std::cout << "Single Sample Margin Relaxation - Mean accuracy: " << mean_single_sample_margin_relaxation << ", Variance: " << var_single_sample_margin_relaxation << std::endl;
    std::cout << "LMS - Mean accuracy: " << mean_lms << ", Variance: " << var_lms << std::endl;
    std::cout << "Ho-Kashyap - Mean accuracy: " << mean_ho_kashyap << ", Variance: " << var_ho_kashyap << std::endl;
    std::cout << "Modified Ho-Kashyap - Mean accuracy: " << mean_modified_ho_kashyap << ", Variance: " << var_modified_ho_kashyap << std::endl;

    return 0;
}


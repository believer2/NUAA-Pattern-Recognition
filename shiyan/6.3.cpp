#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <iostream>
#include <windows.h>
#include <vector>
#include <random>
#include <chrono>

using namespace cv;
using namespace cv::ml;
using namespace std;

void loadImages(const string& datasetPath, vector<Mat>& images, vector<int>& labels) {
    for (int label = 1; label <= 40; ++label) {
        string personPath = datasetPath + "\\s" + to_string(label);
        WIN32_FIND_DATA findFileData;
        HANDLE hFind = FindFirstFile((personPath + "\\*.pgm").c_str(), &findFileData);

        if (hFind == INVALID_HANDLE_VALUE) {
            cerr << "Error opening directory: " << personPath << endl;
            continue;
        }

        do {
            string filePath = personPath + "\\" + findFileData.cFileName;
            Mat img = imread(filePath, IMREAD_GRAYSCALE);
            if (!img.empty()) {
                if (images.empty()) {
                    images.push_back(img);
                } else {
                    resize(img, img, images[0].size());
                    images.push_back(img);
                }
                labels.push_back(label);
            }
        } while (FindNextFile(hFind, &findFileData) != 0);

        FindClose(hFind);
    }
}

void splitData(const vector<Mat>& images, const vector<int>& labels,
               vector<Mat>& trainImages, vector<int>& trainLabels,
               vector<Mat>& testImages, vector<int>& testLabels,
               int numPersons, int numTrainSamples) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(0, numPersons - 1);

    cout << "Chosen persons: ";
    for (int i = 0; i < numPersons; ++i) {
        int person = dis(gen);
        cout << person + 1 << " "; // 人数从 1 开始，而不是 0
        vector<Mat> personImages;
        vector<int> personLabels;
        for (int j = 0; j < images.size(); ++j) {
            if (labels[j] == person + 1) { // 人数从 1 开始，而不是 0
                personImages.push_back(images[j]);
                personLabels.push_back(labels[j]);
            }
        }
        shuffle(personImages.begin(), personImages.end(), gen);
        if (personImages.size() >= numTrainSamples) {
            trainImages.insert(trainImages.end(), personImages.begin(), personImages.begin() + numTrainSamples);
            trainLabels.insert(trainLabels.end(), personLabels.begin(), personLabels.begin() + numTrainSamples);
            testImages.insert(testImages.end(), personImages.begin() + numTrainSamples, personImages.end());
            testLabels.insert(testLabels.end(), personLabels.begin() + numTrainSamples, personLabels.end());
        } else {
            cerr << "Person " << person + 1 << " has insufficient images for training" << endl;
        }
    }
    cout << endl;

    cout << "Train data size: " << trainImages.size() << ", Test data size: " << testImages.size() << endl;
}



Mat flattenImages(const vector<Mat>& images) {
    int numImages = images.size();
    int imageSize = images[0].total();
    Mat data(numImages, imageSize, CV_64F);

    for (int i = 0; i < numImages; ++i) {
        Mat imgFloat;
        images[i].convertTo(imgFloat, CV_64F);
        Mat imgRow = imgFloat.reshape(1, 1);
        imgRow.copyTo(data.row(i));
    }
    return data;
}

Mat computeDistanceMatrix(const Mat& data) {
    Mat distMat(data.rows, data.rows, CV_64F);
    for (int i = 0; i < data.rows; ++i) {
        for (int j = 0; j < data.rows; ++j) {
            double dist = norm(data.row(i) - data.row(j));
            distMat.at<double>(i, j) = dist;
        }
    }
    return distMat;
}

Mat centerDistanceMatrix(const Mat& distMat) {
    int n = distMat.rows;
    Mat centeredMat(n, n, CV_64F);
    
    // Compute row and column means
    Mat rowMeans = Mat::zeros(n, 1, CV_64F);
    Mat colMeans = Mat::zeros(1, n, CV_64F);
    for (int i = 0; i < n; ++i) {
        rowMeans.at<double>(i, 0) = mean(distMat.row(i))[0];
        colMeans.at<double>(0, i) = mean(distMat.col(i))[0];
    }

    // Compute centered distance matrix
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            centeredMat.at<double>(i, j) = distMat.at<double>(i, j) - rowMeans.at<double>(i, 0) - colMeans.at<double>(0, j) + mean(distMat)[0];
        }
    }

    return centeredMat;
}

Mat dpdr(const Mat& data, int targetDim,string distanceMetric) {
    // Compute distance matrix and center it
    Mat distMat = computeDistanceMatrix(data);
    Mat centeredDistMat = centerDistanceMatrix(distMat);

    cout << "Distance matrix size: " << distMat.rows << " x " << distMat.cols << endl;
    cout << "Centered distance matrix size: " << centeredDistMat.rows << " x " << centeredDistMat.cols << endl;

    // Perform eigenvalue decomposition
    Mat eigenvalues, eigenvectors;
    eigen(centeredDistMat, eigenvalues, eigenvectors);

    // Sort eigenvalues and eigenvectors in descending order
    std::vector<std::pair<double, Mat>> eigenPairs;
    for (int i = 0; i < eigenvalues.rows; ++i) {
        eigenPairs.push_back(std::make_pair(eigenvalues.at<double>(i), eigenvectors.col(i)));
    }
    std::sort(eigenPairs.begin(), eigenPairs.end(), [](const auto& left, const auto& right) {
        return left.first > right.first;
    });

    // Select the top eigenvalues and eigenvectors
    Mat projection = Mat::zeros(centeredDistMat.rows, targetDim, CV_64F);
    for (int i = 0; i < targetDim; ++i) {
        eigenPairs[i].second.copyTo(projection.col(i));
    }

    cout << "Projection matrix size: " << projection.rows << " x " << projection.cols << endl;

    // Project the data onto the reduced subspace
    Mat reducedData = centeredDistMat * projection;

    cout << "Reduced data size: " << reducedData.rows << " x " << reducedData.cols << endl;

    return reducedData;
}


int classifyKNN(const Mat& trainData, const vector<int>& trainLabels, const Mat& testData, const vector<int>& testLabels, int k) {
    int correctCount = 0;
    for (int i = 0; i < testData.rows; ++i) {
        Mat sample = testData.row(i);

        // Find the k nearest neighbors in trainData
        vector<pair<double, int>> neighbors;
        for (int j = 0; j < trainData.rows; ++j) {
            double dist = norm(sample - trainData.row(j));
            neighbors.push_back(make_pair(dist, trainLabels[j]));
        }
        sort(neighbors.begin(), neighbors.end());

        // Count votes
        map<int, int> votes;
        for (int j = 0; j < k; ++j) {
            votes[neighbors[j].second]++;
        }

        // Predict the class with the most votes
        int predictedLabel = -1;
        int maxVotes = 0;
        for (const auto& pair : votes) {
            if (pair.second > maxVotes) {
                predictedLabel = pair.first;
                maxVotes = pair.second;
            }
        }

        // Check if the classification is correct
        if (predictedLabel == testLabels[i]) {
            correctCount++;
        }
    }
    return correctCount;
}

int main() {
    // Load dataset
    string datasetPath = "ORL_Faces";
    vector<Mat> images;
    vector<int> labels;
    loadImages(datasetPath, images, labels);
    cout << "Loaded images" << endl;

    // Split dataset
    vector<Mat> trainImages, testImages;
    vector<int> trainLabels, testLabels;
    int numComponents = 25;
    splitData(images, labels, trainImages, trainLabels, testImages, testLabels, numComponents, 6);
    cout << "Split data" << endl;

    // Flatten images
    Mat trainData = flattenImages(trainImages);
    Mat testData = flattenImages(testImages);
    cout << "Flattened images" << endl;

    vector<string> distanceMetrics = {"euclidean", "cosine"};
    vector<int> targetDims = {5, 10, 15};
    vector<int> kValues = {1, 3, 5, 7, 9};

    int bestK = -1;
    double bestAccuracy = 0.0;
    string bestDistanceMetric;
    int bestTargetDim = -1;

    for (const string& distanceMetric : distanceMetrics) {
        for (int targetDim : targetDims) {
            // Apply DPDR
            Mat reducedTrainData = dpdr(trainData, targetDim, distanceMetric);
            Mat reducedTestData = dpdr(testData, targetDim, distanceMetric);

            for (int k : kValues) {
                // Perform classification using KNN with current K value
                int correctCount = classifyKNN(reducedTrainData, trainLabels, reducedTestData, testLabels, k);
                double accuracy = static_cast<double>(correctCount) / reducedTestData.rows * 100.0;

                cout << "Distance Metric: " << distanceMetric << ", Target Dim: " << targetDim << ", K = " << k << ", Accuracy = " << accuracy << "%" << endl;

                // Update best parameters if necessary
                if (accuracy > bestAccuracy) {
                    bestK = k;
                    bestAccuracy = accuracy;
                    bestDistanceMetric = distanceMetric;
                    bestTargetDim = targetDim;
                }
            }
        }
    }

    cout << "Best Distance Metric: " << bestDistanceMetric << ", Best Target Dim: " << bestTargetDim << ", Best K: " << bestK << ", Best Accuracy: " << bestAccuracy << "%" << endl;

    return 0;
}


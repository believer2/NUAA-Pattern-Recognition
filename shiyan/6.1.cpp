#include <opencv2/opencv.hpp>
#include <iostream>
#include <windows.h>
#include <vector>
#include <random>
#include <chrono>

using namespace cv;
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
    uniform_int_distribution<> dis(0, images.size() - 1);

    vector<int> chosenPersons;
    while (chosenPersons.size() < numPersons) {
        int person = labels[dis(gen)];
        if (find(chosenPersons.begin(), chosenPersons.end(), person) == chosenPersons.end()) {
            chosenPersons.push_back(person);
        }
    }

    for (int person : chosenPersons) {
        vector<Mat> personImages;
        vector<int> personLabels;
        for (int i = 0; i < images.size(); ++i) {
            if (labels[i] == person) {
                personImages.push_back(images[i]);
                personLabels.push_back(labels[i]);
            }
        }
        shuffle(personImages.begin(), personImages.end(), gen);
        for (int i = 0; i < numTrainSamples; ++i) {
            trainImages.push_back(personImages[i]);
            trainLabels.push_back(personLabels[i]);
        }
        for (int i = numTrainSamples; i < personImages.size(); ++i) {
            testImages.push_back(personImages[i]);
            testLabels.push_back(personLabels[i]);
        }
    }
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

void performPCA(const Mat& data, Mat& projectedData, Mat& mean, Mat& eigenVectors, int numComponents) {
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, numComponents);
    mean = pca.mean.clone();
    eigenVectors = pca.eigenvectors.clone();
    pca.project(data, projectedData);
}

void performPCAWithSVD(const Mat& data, Mat& projectedData, Mat& mean, Mat& eigenVectors, int numComponents) {
    Mat centeredData;
    mean = Mat::zeros(1, data.cols, data.type());
    for (int i = 0; i < data.rows; ++i) {
        mean += data.row(i);
    }
    mean /= data.rows;
    centeredData = data - repeat(mean, data.rows, 1);

    Mat W, U, Vt;
    SVD::compute(centeredData, W, U, Vt);
    eigenVectors = Vt.rowRange(0, numComponents);
    projectedData = centeredData * eigenVectors.t();
}

int classify(const Mat& trainData, const vector<int>& trainLabels, const Mat& sample) {
    double minDist = DBL_MAX;
    int bestLabel = -1;

    for (int i = 0; i < trainData.rows; ++i) {
        double dist = norm(trainData.row(i), sample);
        if (dist < minDist) {
            minDist = dist;
            bestLabel = trainLabels[i];
        }
    }
    return bestLabel;
}

double evaluateClassifier(const Mat& trainData, const vector<int>& trainLabels,
                          const Mat& testData, const vector<int>& testLabels) {
    int correctCount = 0;
    for (int i = 0; i < testData.rows; ++i) {
        int predictedLabel = classify(trainData, trainLabels, testData.row(i));
        if (predictedLabel == testLabels[i]) {
            correctCount++;
        }
    }
    return (double)correctCount / testData.rows;
}

int main() {
    string datasetPath = "ORL_Faces";
    vector<Mat> images;
    vector<int> labels;

    loadImages(datasetPath, images, labels);
    cout << "Loaded images" << endl;

    vector<Mat> trainImages, testImages;
    vector<int> trainLabels, testLabels;
    splitData(images, labels, trainImages, trainLabels, testImages, testLabels, 25, 5);
    cout << "Split data" << endl;

    Mat trainData = flattenImages(trainImages);
    Mat testData = flattenImages(testImages);
    cout << "Flattened images" << endl;

    trainData.convertTo(trainData, CV_32F);
    testData.convertTo(testData, CV_32F);

    Scalar mean, stddev;
    meanStdDev(trainData, mean, stddev);
    trainData = (trainData - mean[0]) / stddev[0];
    testData = (testData - mean[0]) / stddev[0];

    int numComponents = 50;
    Mat projectedTrainDataGeneral, meanVectorGeneral, eigenVectorsGeneral;
    auto start = chrono::high_resolution_clock::now();
    performPCA(trainData, projectedTrainDataGeneral, meanVectorGeneral, eigenVectorsGeneral, numComponents);
    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> durationGeneral = end - start;
    cout << "PCA (general method) Time: " << durationGeneral.count() << " seconds" << endl;

    Mat testDataCenteredGeneral = testData - repeat(meanVectorGeneral, testData.rows, 1);
    Mat projectedTestDataGeneral = testDataCenteredGeneral * eigenVectorsGeneral.t();

    double accuracyGeneral = evaluateClassifier(projectedTrainDataGeneral, trainLabels, projectedTestDataGeneral, testLabels);
    cout << "Classification accuracy (general method): " << accuracyGeneral * 100 << "%" << endl;

    Mat projectedTrainDataSVD, meanVectorSVD, eigenVectorsSVD;
    start = chrono::high_resolution_clock::now();
    performPCAWithSVD(trainData, projectedTrainDataSVD, meanVectorSVD, eigenVectorsSVD, numComponents);
    end = chrono::high_resolution_clock::now();
    chrono::duration<double> durationSVD = end - start;
    cout << "PCA (SVD method) Time: " << durationSVD.count() << " seconds" << endl;

    Mat testDataCenteredSVD = testData - repeat(meanVectorSVD, testData.rows, 1);
    Mat projectedTestDataSVD = testDataCenteredSVD * eigenVectorsSVD.t();

    double accuracySVD = evaluateClassifier(projectedTrainDataSVD, trainLabels, projectedTestDataSVD, testLabels);
    cout << "Classification accuracy (SVD method): " << accuracySVD * 100 << "%" << endl;

    return 0;
}

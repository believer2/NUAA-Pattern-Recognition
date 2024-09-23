#include <opencv2/opencv.hpp>
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
// 数据增广
void augmentData(vector<Mat>& images, vector<int>& labels, int numAugmentedImagesPerPerson) {
    vector<Mat> augmentedImages;
    vector<int> augmentedLabels;

    for (size_t i = 0; i < images.size(); ++i) {
        // 原始图像
        augmentedImages.push_back(images[i]);
        augmentedLabels.push_back(labels[i]);

        // 添加数据增广操作，例如翻转图像
        Mat flippedImage;
        flip(images[i], flippedImage, 1); // 水平翻转
        augmentedImages.push_back(flippedImage);
        augmentedLabels.push_back(labels[i]); // 对应的标签也需要增加

        // 如果还有其他增广操作，可以继续添加
    }

    // 将增广后的数据合并到原始数据中
    images.insert(images.end(), augmentedImages.begin(), augmentedImages.end());
    labels.insert(labels.end(), augmentedLabels.begin(), augmentedLabels.end());
}

void splitData(const vector<Mat>& images, const vector<int>& labels,
               vector<Mat>& trainImages, vector<int>& trainLabels,
               vector<Mat>& testImages, vector<int>& testLabels,
               int numPersons, int numTrainSamples) {
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, numPersons); // assuming labels are from 1 to numPersons

    set<int> chosenPersons;
    while (chosenPersons.size() < numPersons) {
        int person = dis(gen);
        chosenPersons.insert(person);
    }

    cout << "Chosen persons: ";
    for (int person : chosenPersons) {
        cout << person << " ";
    }
    cout << endl;

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

Mat performMDA(const Mat& trainData, const vector<int>& trainLabels, int numComponents) {
    Mat S_W = Mat::zeros(trainData.cols, trainData.cols, CV_64F);
    Mat S_B = Mat::zeros(trainData.cols, trainData.cols, CV_64F);
    Mat mean = Mat::zeros(1, trainData.cols, CV_64F);

    // Compute class means
    vector<Mat> classMeans(numComponents, Mat::zeros(1, trainData.cols, CV_64F)); // Initialize class means
    vector<int> classCounts(numComponents, 0); // Initialize class counts

    for (int i = 0; i < trainData.rows; ++i) {
        int label = trainLabels[i] - 1; // Adjust label index (0-based index)
        if (label < 0 || label >= numComponents) {
            cerr << "Invalid label: " << label + 1 << endl;
            return Mat(); // Return empty matrix to signal error
        }
        classMeans[label] += trainData.row(i);
        classCounts[label]++;
    }

    for (int i = 0; i < numComponents; ++i) {
        if (classCounts[i] > 0) {
            classMeans[i] /= classCounts[i];
            mean += classMeans[i];
        } else {
            cerr << "Class " << i + 1 << " has no samples." << endl;
            return Mat(); // Return empty matrix to signal error
        }
    }
    mean /= numComponents;

    // Compute within-class scatter matrix
    for (int i = 0; i < trainData.rows; ++i) {
        int label = trainLabels[i] - 1; // Adjust label index (0-based index)
        if (label < 0 || label >= numComponents) {
            cerr << "Invalid label during S_W computation: " << label + 1 << endl;
            return Mat(); // Return empty matrix to signal error
        }
        Mat diff = trainData.row(i) - classMeans[label];
        S_W += diff.t() * diff;
    }

    // Compute between-class scatter matrix
    for (int i = 0; i < numComponents; ++i) {
        Mat diff = classMeans[i] - mean;
        S_B += classCounts[i] * diff.t() * diff;
    }

    // Debugging output
    cout << "S_W size: " << S_W.size() << endl;
    cout << "S_W type: " << S_W.type() << endl;
    cout << "S_B size: " << S_B.size() << endl;
    cout << "S_B type: " << S_B.type() << endl;

    // Compute eigenvectors of (S_W^-1) * S_B
    Mat eigenvalues, eigenvectors;
    eigen(S_W.inv() * S_B, eigenvalues, eigenvectors);

    // Select top numComponents eigenvectors
    Mat W = eigenvectors.rowRange(0, numComponents);

    return W;
}

Mat performPCA(const Mat& data, int numComponents) {
    PCA pca(data, Mat(), PCA::DATA_AS_ROW, numComponents);
    return pca.project(data);
}

float trainAndTestKNN(const Mat& reducedTrainData, const vector<int>& trainLabels,
                      const Mat& reducedTestData, const vector<int>& testLabels) {
    Ptr<KNearest> knn = KNearest::create();
    Mat reducedTrainData32F, reducedTestData32F;
    reducedTrainData.convertTo(reducedTrainData32F, CV_32F);
    reducedTestData.convertTo(reducedTestData32F, CV_32F);
    knn->setDefaultK(7); // 设置K值，可以根据需要进行调整
    knn->setIsClassifier(true);
    knn->train(reducedTrainData32F, ROW_SAMPLE, Mat(trainLabels));
    cout << "Trained K-Nearest Neighbors classifier" << endl;

    Mat predictions;
    knn->findNearest(reducedTestData32F, knn->getDefaultK(), predictions);
    cout << "Performed predictions" << endl;

    int correct = 0;
    for (int i = 0; i < predictions.rows; ++i) {
        if (predictions.at<float>(i) == testLabels[i]) {
            correct++;
        }
    }
    float accuracy = (float)correct / predictions.rows * 100.0;
    cout << "Accuracy: " << accuracy << "%" << endl;

    return accuracy;
}

void displayOriginalAndFlipped(const Mat& originalImage, const Mat& flippedImage, const string& windowName) {
    Mat displayImage(originalImage.rows, originalImage.cols * 2, originalImage.type());
    originalImage.copyTo(displayImage(Rect(0, 0, originalImage.cols, originalImage.rows)));
    flippedImage.copyTo(displayImage(Rect(originalImage.cols, 0, originalImage.cols, originalImage.rows)));

    namedWindow(windowName, WINDOW_AUTOSIZE);
    imshow(windowName, displayImage);
    waitKey(0);
    destroyWindow(windowName);
}

int main() {
    // Load dataset
    string datasetPath = "ORL_Faces";
    vector<Mat> images;
    vector<int> labels;
    loadImages(datasetPath, images, labels);
    cout << "Loaded images" << endl;

     // 数据增广
    int numAugmentedImagesPerPerson = 5; // 每个人增广的图像数量
    augmentData(images, labels, numAugmentedImagesPerPerson);
    cout << "Augmented data" << endl;

    // 选取一个人的原始图像和一个增广后的图像展示
    int personLabel = 1; // 选择的人的标签
    auto it = find(labels.begin(), labels.end(), personLabel);
    if (it != labels.end()) {
        size_t index = distance(labels.begin(), it); // 获取该标签的第一个出现位置
        Mat originalImage = images[index];
        Mat flippedImage;
        flip(originalImage, flippedImage, 1); // 水平翻转
        displayOriginalAndFlipped(originalImage, flippedImage, "Person " + to_string(personLabel));
    } else {
        cerr << "Person with label " << personLabel << " not found in the dataset." << endl;
    }

    // Split dataset
    vector<Mat> trainImages, testImages;
    vector<int> trainLabels, testLabels;
    int numComponents = 25; // 需要选择的标签数量
    splitData(images, labels, trainImages, trainLabels, testImages, testLabels, numComponents, 6);
    cout << "Split data" << endl;

    // Flatten images
    Mat trainData = flattenImages(trainImages);
    Mat testData = flattenImages(testImages);
    cout << "Flattened images" << endl;

    // Debugging output
    cout << "trainData size: " << trainData.size() << endl;
    cout << "trainData type: " << trainData.type() << endl;

    // Perform PCA to reduce dimensionality
    int pcaComponents = 100; // Number of components to keep after PCA
    Mat pcaTrainData = performPCA(trainData, pcaComponents);
    Mat pcaTestData = performPCA(testData, pcaComponents);
    cout << "Performed PCA" << endl;

    // Debugging output for PCA
    cout << "pcaTrainData size: " << pcaTrainData.size() << endl;
    cout << "pcaTestData size: " << pcaTestData.size() << endl;

    // Perform MDA feature extraction
    Mat W = performMDA(pcaTrainData, trainLabels, numComponents);
    if (W.empty()) {
        cerr << "Error in performMDA" << endl;
        return -1;
    }

    // Debugging output
    cout << "W size: " << W.size() << endl;

    // Apply MDA transformation
    Mat reducedTrainData = pcaTrainData * W.t();
    Mat reducedTestData = pcaTestData * W.t();

    float accuracy = trainAndTestKNN(reducedTrainData, trainLabels, reducedTestData, testLabels);

    cout << "Accuracy: " << accuracy << "%" << endl;

    return 0;
}

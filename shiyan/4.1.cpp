#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

struct Point {
    vector<double> features;
    int label;

    Point(int dims, int label) : features(dims), label(label) {}
};

void generatePoints(vector<Point> &points, int numPoints, int dims, int label) {
    for (int i = 0; i < numPoints; ++i) {
        Point p(dims, label);
        for (int j = 0; j < dims; ++j) {
            p.features[j] = static_cast<double>(rand()) / RAND_MAX;
        }
        points.push_back(p);
    }
}

double distance(const Point &a, const Point &b) {
    double dist = 0.0;
    for (size_t i = 0; i < a.features.size(); ++i) {
        dist += pow(a.features[i] - b.features[i], 2);
    }
    return sqrt(dist);
}

int knnClassify(const vector<Point> &trainSet, const Point &query, int K) {
    vector<pair<double, int>> distances;
    for (const auto &point : trainSet) {
        distances.push_back({distance(point, query), point.label});
    }
    sort(distances.begin(), distances.end());

    vector<int> classCount(2, 0); // Assuming two classes: 0 and 1
    for (int i = 0; i < K; ++i) {
        classCount[distances[i].second]++;
    }
    return (classCount[0] > classCount[1]) ? 0 : 1;
}

double calculateErrorRate(const vector<Point> &dataset, const vector<Point> &trainSet, int K) {
    int errors = 0;
    for (const auto &point : dataset) {
        if (knnClassify(trainSet, point, K) != point.label) {
            errors++;
        }
    }
    return static_cast<double>(errors) / dataset.size();
}

int findFirstLocalMinimumK(const vector<Point> &valSet, const vector<Point> &trainSet) {
    int bestK = 1;
    double minValError = calculateErrorRate(valSet, trainSet, bestK);
    for (int K = 3; K <= trainSet.size(); K += 2) {
        double valError = calculateErrorRate(valSet, trainSet, K);
        if (valError < minValError) {
            minValError = valError;
            bestK = K;
        } else {
            break;
        }
    }
    return bestK;
}

int findFirstLocalMaximumK(const vector<Point> &valSet, const vector<Point> &trainSet) {
    int bestK = 1;
    double maxValError = calculateErrorRate(valSet, trainSet, bestK);
    for (int K = 3; K <= trainSet.size(); K += 2) {
        double valError = calculateErrorRate(valSet, trainSet, K);
        if (valError > maxValError) {
            maxValError = valError;
            bestK = K;
        } else {
            break;
        }
    }
    return bestK;
}

int main() {
    srand(static_cast<unsigned>(time(0)));

    const int dims = 3;
    const int numTestPoints = 30;
    const int numTrainPointsPerClass = 60;
    const double gamma = 0.15;
    const int numTrainPoints = static_cast<int>((1 - gamma) * 2 * numTrainPointsPerClass);
    const int numValPoints = 2 * numTrainPointsPerClass - numTrainPoints;

    vector<double> testErrorsMin, testErrorsMax;

    for (int experiment = 0; experiment < 5; ++experiment) {
        vector<Point> testSet;
        generatePoints(testSet, numTestPoints / 2, dims, 0);
        generatePoints(testSet, numTestPoints / 2, dims, 1);

        vector<Point> dataset;
        generatePoints(dataset, numTrainPointsPerClass, dims, 0);
        generatePoints(dataset, numTrainPointsPerClass, dims, 1);

        random_shuffle(dataset.begin(), dataset.end());
        vector<Point> trainSet(dataset.begin(), dataset.begin() + numTrainPoints);
        vector<Point> valSet(dataset.begin() + numTrainPoints, dataset.end());

        int bestKMin = findFirstLocalMinimumK(valSet, trainSet);
        double testErrorRateMin = calculateErrorRate(testSet, trainSet, bestKMin);
        testErrorsMin.push_back(testErrorRateMin);

        int bestKMax = findFirstLocalMaximumK(valSet, trainSet);
        double testErrorRateMax = calculateErrorRate(testSet, trainSet, bestKMax);
        testErrorsMax.push_back(testErrorRateMax);

        cout << "Experiment " << experiment + 1 << " - Min K: " << bestKMin << ", Test Error Rate: " << testErrorRateMin << endl;
        cout << "Experiment " << experiment + 1 << " - Max K: " << bestKMax << ", Test Error Rate: " << testErrorRateMax << endl;
    }

    cout << "All test errors (first local minimum): ";
    for (double error : testErrorsMin) {
        cout << error << " ";
    }
    cout << endl;

    cout << "All test errors (first local maximum): ";
    for (double error : testErrorsMax) {
        cout << error << " ";
    }
    cout << endl;

    return 0;
}

#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

struct Point {
    double x, y, z;
};

double modified_distance(const Point& a, const Point& b, double beta) {
    double euclidean_dist_sq = pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2);
    return 1 - exp(-beta * euclidean_dist_sq);
}

vector<int> k_means(const vector<Point>& data, vector<Point>& centroids, double beta, int max_iters = 100) {
    int k = centroids.size();
    int n = data.size();
    vector<int> labels(n);
    int iter_count = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        iter_count = iter + 1;  
        for (int i = 0; i < n; ++i) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;
            for (int j = 0; j < k; ++j) {
                double dist = modified_distance(data[i], centroids[j], beta);
                if (dist < min_dist) {
                    min_dist = dist;
                    best_cluster = j;
                }
            }
            labels[i] = best_cluster;
        }

        vector<Point> new_centroids(k, {0, 0, 0});
        vector<int> counts(k, 0);
        for (int i = 0; i < n; ++i) {
            new_centroids[labels[i]].x += data[i].x;
            new_centroids[labels[i]].y += data[i].y;
            new_centroids[labels[i]].z += data[i].z;
            counts[labels[i]]++;
        }
        for (int j = 0; j < k; ++j) {
            if (counts[j] != 0) {
                new_centroids[j].x /= counts[j];
                new_centroids[j].y /= counts[j];
                new_centroids[j].z /= counts[j];
            }
        }

        bool converged = true;
        for (int j = 0; j < k; ++j) {
            if (modified_distance(centroids[j], new_centroids[j], beta) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        centroids = new_centroids;
    }

    cout << "K-means iterations: " << iter_count << endl;
    return labels;
}

void fuzzy_k_means(const vector<Point>& data, vector<Point>& centroids, double beta, double m = 2.0, int max_iters = 100) {
    int k = centroids.size();
    int n = data.size();
    vector<vector<double>> membership(n, vector<double>(k));
    int iter_count = 0;

    for (int iter = 0; iter < max_iters; ++iter) {
        iter_count = iter + 1; 

        for (int i = 0; i < n; ++i) {
            double sum_dist = 0.0;
            for (int j = 0; j < k; ++j) {
                double dist = modified_distance(data[i], centroids[j], beta);
                if (dist == 0) dist = 1e-10; // To avoid division by zero
                sum_dist += pow(1.0 / dist, 1.0 / (m - 1.0));
            }
            for (int j = 0; j < k; ++j) {
                double dist = modified_distance(data[i], centroids[j], beta);
                if (dist == 0) dist = 1e-10;
                membership[i][j] = pow(1.0 / dist, 1.0 / (m - 1.0)) / sum_dist;
            }
        }

        vector<Point> new_centroids(k, {0, 0, 0});
        vector<double> sum_membership(k, 0);
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                double um = pow(membership[i][j], m);
                new_centroids[j].x += um * data[i].x;
                new_centroids[j].y += um * data[i].y;
                new_centroids[j].z += um * data[i].z;
                sum_membership[j] += um;
            }
            if (sum_membership[j] != 0) {
                new_centroids[j].x /= sum_membership[j];
                new_centroids[j].y /= sum_membership[j];
                new_centroids[j].z /= sum_membership[j];
            }
        }

        bool converged = true;
        for (int j = 0; j < k; ++j) {
            if (modified_distance(centroids[j], new_centroids[j], beta) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) break;

        centroids = new_centroids;
    }

    cout << "Fuzzy K-means iterations: " << iter_count << endl;
    cout << "Final centroids:" << endl;
    for (const auto& centroid : centroids) {
        cout << "(" << centroid.x << ", " << centroid.y << ", " << centroid.z << ") ";
    }
    cout << endl;
}

int main() {
    vector<Point> data = {
        {-7.82,-4.58,-3.97},{-6.68,3.16,2.71},{4.36,-2.19,2.09},{6.72,0.88,2.8},{-8.64,3.06,3.5},
        {-6.87,0.57,-5.45},{4.47,-2.62,5.76},{6.73,-2.01,4.18},{-7.71,2.34,-6.33},{-6.91,-0.49,-5.68},
        {-6.18,2.81,5.82},{6.72,-0.93,-4.04},{-6.25,-0.26,0.56},{-6.94,-1.22,1.13},{8.09,0.2,2.25},
        {6.81,0.17,-4.15},{-5.19,4.24,4.04},{-6.38,-1.74,1.43},{4.08,1.3,5.33},{6.27,0.93,-2.78}
    };

    vector<double> betas = {0.001, 0.01, 0.1, 1, 10, 100};

    for (double beta : betas) {
        cout << "Beta: " << beta << endl;

        vector<Point> centroids_1 = {{1, 1, 1}, {-1, 1, -1}};
        vector<Point> centroids_2 = {{0, 0, 0}, {1, 1, -1}};
        
        cout << "Running K-means with initial centroids_1" << endl;
        vector<int> labels_1 = k_means(data, centroids_1, beta);
        cout << "Running K-means with initial centroids_2" << endl;
        vector<int> labels_2 = k_means(data, centroids_2, beta);
        
        cout << "Labels for centroids_1 (K-means):" << endl;
        for (const auto& label : labels_1) cout << label << " ";
        cout << endl;
        
        cout << "Labels for centroids_2 (K-means):" << endl;
        for (const auto& label : labels_2) cout << label << " ";
        cout << endl;

        cout << "Final centroids for centroids_1 (K-means):" << endl;
        for (const auto& centroid : centroids_1) cout << "(" << centroid.x << ", " << centroid.y << ", " << centroid.z << ") ";
        cout << endl;
        
        cout << "Final centroids for centroids_2 (K-means):" << endl;
        for (const auto& centroid : centroids_2) cout << "(" << centroid.x << ", " << centroid.y << ", " << centroid.z << ") ";
        cout << endl;

        cout << "Running Fuzzy K-means with initial centroids_1" << endl;
        fuzzy_k_means(data, centroids_1, beta);
                cout << "Running Fuzzy K-means with initial centroids_2" << endl;
        fuzzy_k_means(data, centroids_2, beta);

        vector<Point> centroids_3_1 = {{0, 0, 0}, {1, 1, 1}, {-1, 0, 2}};
        vector<Point> centroids_3_2 = {{-0.1, 0, 0.1}, {0, -0.1, 0.1}, {-0.1, -0.1, 0.1}};
        
        cout << "Running K-means with initial centroids_3_1" << endl;
        vector<int> labels_3_1 = k_means(data, centroids_3_1, beta);
        cout << "Running K-means with initial centroids_3_2" << endl;
        vector<int> labels_3_2 = k_means(data, centroids_3_2, beta);

        cout << "Labels for centroids_3_1 (K-means):" << endl;
        for (const auto& label : labels_3_1) cout << label << " ";
        cout << endl;
        
        cout << "Labels for centroids_3_2 (K-means):" << endl;
        for (const auto& label : labels_3_2) cout << label << " ";
        cout << endl;

        cout << "Final centroids for centroids_3_1 (K-means):" << endl;
        for (const auto& centroid : centroids_3_1) cout << "(" << centroid.x << ", " << centroid.y << ", " << centroid.z << ") ";
        cout << endl;
        
        cout << "Final centroids for centroids_3_2 (K-means):" << endl;
        for (const auto& centroid : centroids_3_2) cout << "(" << centroid.x << ", " << centroid.y << ", " << centroid.z << ") ";
        cout << endl;

        cout << "Running Fuzzy K-means with initial centroids_3_1" << endl;
        fuzzy_k_means(data, centroids_3_1, beta);
        cout << "Running Fuzzy K-means with initial centroids_3_2" << endl;
        fuzzy_k_means(data, centroids_3_2, beta);
    }

    return 0;
}


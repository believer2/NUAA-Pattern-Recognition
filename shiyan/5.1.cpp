#include <iostream>
#include <vector>
#include <cmath>
#include <limits>

using namespace std;

struct Point {
    double x, y, z;
};

double euclidean_distance(const Point& a, const Point& b) {
    return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2) + pow(a.z - b.z, 2));
}

vector<int> k_means(const vector<Point>& data, vector<Point>& centroids, int& iterations, int max_iters = 100) {
    int k = centroids.size();
    int n = data.size();
    vector<int> labels(n);
    iterations = 0;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        iterations++;
        for (int i = 0; i < n; ++i) {
            double min_dist = numeric_limits<double>::max();
            int best_cluster = 0;
            for (int j = 0; j < k; ++j) {
                double dist = euclidean_distance(data[i], centroids[j]);
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
            if (euclidean_distance(centroids[j], new_centroids[j]) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) break;
        
        centroids = new_centroids;
    }
    
    return labels;
}

double update_membership(const Point& point, const Point& centroid, const vector<Point>& centroids, double m) {
    double sum = 0.0;
    double dist_to_centroid = euclidean_distance(point, centroid);
    for (const auto& other_centroid : centroids) {
        double dist_to_other_centroid = euclidean_distance(point, other_centroid);
        if (dist_to_other_centroid == 0.0) {
            dist_to_other_centroid = 1e-10;  // Avoid division by zero
        }
        sum += pow(dist_to_centroid / dist_to_other_centroid, 2.0 / (m - 1.0));
    }
    return 1.0 / sum;
}

vector<int> fuzzy_k_means(const vector<Point>& data, vector<Point>& centroids, int& iterations, double m = 2.0, int max_iters = 100) {
    int k = centroids.size();
    int n = data.size();
    vector<vector<double>> membership(n, vector<double>(k));
    iterations = 0;
    
    for (int iter = 0; iter < max_iters; ++iter) {
        iterations++;
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
                membership[i][j] = update_membership(data[i], centroids[j], centroids, m);
            }
        }
        
        vector<Point> new_centroids(k, {0, 0, 0});
        vector<double> membership_sum(k, 0);
        
        for (int j = 0; j < k; ++j) {
            for (int i = 0; i < n; ++i) {
                double u = pow(membership[i][j], m);
                new_centroids[j].x += u * data[i].x;
                new_centroids[j].y += u * data[i].y;
                new_centroids[j].z += u * data[i].z;
                membership_sum[j] += u;
            }
            if (membership_sum[j] != 0) {
                new_centroids[j].x /= membership_sum[j];
                new_centroids[j].y /= membership_sum[j];
                new_centroids[j].z /= membership_sum[j];
            }
        }
        
        bool converged = true;
        for (int j = 0; j < k; ++j) {
            if (euclidean_distance(centroids[j], new_centroids[j]) > 1e-6) {
                converged = false;
                break;
            }
        }
        if (converged) break;
        
        centroids = new_centroids;
    }

    vector<int> labels(n);
    for (int i = 0; i < n; ++i) {
        int best_cluster = 0;
        double max_membership = 0;
        for (int j = 0; j < k; ++j) {
            if (membership[i][j] > max_membership) {
                max_membership = membership[i][j];
                best_cluster = j;
            }
        }
        labels[i] = best_cluster;
    }

    return labels;
}

int main() {
    vector<Point> data = {
        {-7.82,-4.58,-3.97},{-6.68,3.16,2.71},{4.36,-2.19,2.09},{6.72,0.88,2.8},{-8.64,3.06,3.5},
        {-6.87,0.57,-5.45},{4.47,-2.62,5.76},{6.73,-2.01,4.18},{-7.71,2.34,-6.33},{-6.91,-0.49,-5.68},
        {-6.18,2.81,5.82},{6.72,-0.93,-4.04},{-6.25,-0.26,0.56},{-6.94,-1.22,1.13},{8.09,0.2,2.25},
        {6.81,0.17,-4.15},{-5.19,4.24,4.04},{-6.38,-1.74,1.43},{4.08,1.3,5.33},{6.27,0.93,-2.78}
    };
    
    vector<Point> centroids_1 = {{1, 1, 1}, {-1, 1, -1}};
    vector<Point> centroids_2 = {{0, 0, 0}, {1, 1, -1}};
    vector<Point> centroids_3 = {{0, 0, 0}, {1, 1, 1}, {-1, 0, 2}};
    vector<Point> centroids_4 = {{-0.1, 0, 0.1}, {0, -0.1, 0.1}, {-0.1, -0.1, 0.1}};

    int iterations_1, iterations_2, iterations_3, iterations_4;

    vector<int> labels_1 = k_means(data, centroids_1, iterations_1);
    vector<int> labels_2 = k_means(data, centroids_2, iterations_2);
    vector<int> labels_3 = k_means(data, centroids_3, iterations_3);
    vector<int> labels_4 = k_means(data, centroids_4, iterations_4);
    
    cout << "Labels for centroids_1 (K-means):" << endl;
    for (const auto& label : labels_1) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_1 (K-means): " << iterations_1 << endl;
    
    cout << "Labels for centroids_2 (K-means):" << endl;
    for (const auto& label : labels_2) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_2 (K-means): " << iterations_2 << endl;
    
    cout << "Labels for centroids_3 (K-means):" << endl;
    for (const auto& label : labels_3) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_3 (K-means): " << iterations_3 << endl;
    
    cout << "Labels for centroids_4 (K-means):" << endl;
    for (const auto& label : labels_4) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_4 (K-means): " << iterations_4 << endl;

    centroids_1 = {{1, 1, 1}, {-1, 1, -1}};
    centroids_2 = {{0, 0, 0}, {1, 1, -1}};
    centroids_3 = {{0, 0, 0}, {1, 1, 1}, {-1, 0, 2}};
    centroids_4 = {{-0.1, 0, 0.1}, {0, -0.1, 0.1}, {-0.1, -0.1, 0.1}};
    
    vector<int> fuzzy_labels_1 = fuzzy_k_means(data, centroids_1, iterations_1);
    vector<int> fuzzy_labels_2 = fuzzy_k_means(data, centroids_2, iterations_2);
    vector<int> fuzzy_labels_3 = fuzzy_k_means(data, centroids_3, iterations_3);
    vector<int> fuzzy_labels_4 = fuzzy_k_means(data, centroids_4, iterations_4);

    cout << "Labels for centroids_1 (Fuzzy K-means):" << endl;
    for (const auto& label : fuzzy_labels_1) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_1 (Fuzzy K-means): " << iterations_1 << endl;
    
    cout << "Labels for centroids_2 (Fuzzy K-means):" << endl;
    for (const auto& label : fuzzy_labels_2) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_2 (Fuzzy K-means): " << iterations_2 << endl;
    
    cout << "Labels for centroids_3 (Fuzzy K-means):" << endl;
    for (const auto& label : fuzzy_labels_3) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_3 (Fuzzy K-means): " << iterations_3 << endl;
    
    cout << "Labels for centroids_4 (Fuzzy K-means):" << endl;
    for (const auto& label : fuzzy_labels_4) cout << label << " ";
    cout << endl;
    cout << "Iterations for centroids_4 (Fuzzy K-means): " << iterations_4 << endl;
    
    return 0;
}

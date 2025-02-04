/*
Name: Matej Zecic
Date: Spring 2025
Description: This program calculates different similarity measures for an image used to determine similarities between images.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <iostream>
#include "opencv2/opencv.hpp"
#include "../include/csv_util.h"
#include "../include/similarity_extraction.h"

// Function to compute Euclidean distance
float compute_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float distance = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return sqrt(distance);
}

// Function to compute Cosine Similarity
float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot_product = 0.0, norm_v1 = 0.0, norm_v2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }
    return dot_product / (sqrt(norm_v1) * sqrt(norm_v2));
}

// Function to compute Mahalanobis Distance
float mahalanobis_distance(const std::vector<float>& v1, const std::vector<float>& v2, const cv::Mat& inv_covariance) {
    cv::Mat diff = cv::Mat(v1) - cv::Mat(v2);
    cv::Mat result = diff.t() * inv_covariance * diff;
    return sqrt(result.at<float>(0, 0));
}

// Function to compute histogram intersection
float histogram_intersection(const std::vector<float>& v1, const std::vector<float>& v2) {
    float intersection = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        intersection += std::min(v1[i], v2[i]);
    }
    return intersection;
}

// Function that computes histogram intersection between texture and color equally
float histogram_intersection_2d(const std::vector<float>& v1, const std::vector<float>& v2) {
    float intersection_color = 0.0;
    float intersection_texture = 0.0;
    float sum_color = 0.0;
    float sum_texture = 0.0;

    for (size_t i = 0; i < v1.size() - 8; ++i) {
        intersection_color += std::min(v1[i], v2[i]);
        sum_color += v1[i];
    }

    for (size_t j = v1.size() - 8; j < v1.size(); ++j) {
        intersection_texture += std::min(v1[j], v2[j]);
        sum_texture += v1[j];
    }

    if (sum_color > 0) intersection_color /= sum_color;
    if (sum_texture > 0) intersection_texture /= sum_texture;

    return (intersection_color + intersection_texture) / 2;
}

// Function to extract features based on the selected method
std::vector<float> extract_features(const char* image_path, const char* feature_set) {
    if (strcmp(feature_set, "7x7") == 0) {
        return extract_7x7(image_path);
    } else if (strcmp(feature_set, "histogram_matching") == 0) {
        return compute_histogram(image_path);
    } else if (strcmp(feature_set, "multi_histogram") == 0) {
        return compute_multi_histogram(image_path);
    } else if (strcmp(feature_set, "texture_and_color") == 0) {
        return compute_texture_and_color(image_path);
    } else if (strcmp(feature_set, "banana_detection") == 0) {
        return banana_extraction(image_path);
    }

    std::cerr << "Error: Unknown feature set \"" << feature_set << "\"." << std::endl;
    return {};
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_set> <feature_vector_file> <N>" << std::endl;
        return -1;
    }

    char *target_image = argv[1];
    char *feature_set = argv[2];
    char *feature_vector_file = argv[3];
    int N = atoi(argv[4]);

    // Extract features from target image
    std::vector<float> target_feature = extract_features(target_image, feature_set);
    if (target_feature.empty()) {
        std::cerr << "Failed to extract features from the target image." << std::endl;
        return -1;
    }

    // Read feature vectors from CSV
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(feature_vector_file, filenames, data, 0) != 0) {
        std::cerr << "Failed to read the feature vector file." << std::endl;
        return -1;
    }

    // Compute distances for similarity ranking
    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        bool all_zeros = true;
        for (size_t j = 0; j < data[i].size(); ++j) {
            if (data[i][j] != 0.0) {
                all_zeros = false;
                break;
            }
        }

        if (all_zeros) {
            std::cerr << "Skipping image " << filenames[i] << " because the feature vector is all zeros." << std::endl;
            continue;
        }

        float distance;
        if (strcmp(feature_set, "dnembeddings") == 0) {
            distance = cosine_similarity(target_feature, data[i]);
        } else if (strcmp(feature_set, "histogram_matching") == 0 || strcmp(feature_set, "multi_histogram") == 0) {
            distance = 1 - histogram_intersection(target_feature, data[i]);
        } else if (strcmp(feature_set, "texture_and_color") == 0) {
            distance = histogram_intersection_2d(target_feature, data[i]);
        } else if (strcmp(feature_set, "banana_detection") == 0) {
            distance = compute_distance(target_feature, data[i]);
        } else {
            distance = compute_distance(target_feature, data[i]);
        }
        distances.emplace_back(distance, filenames[i]);
    }

    // Sort results
    if (strcmp(feature_set, "dnembeddings") == 0) {
        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;  // Higher similarity is better
        });
    } else {
        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;  // Lower distance is better
        });
    }

    // Print top N matches
    std::cout << "Top " << N << " matches (higher is better for similarity, lower for distance):" << std::endl;
    for (int i = 0; i < N && i < distances.size(); ++i) {
        std::cout << distances[i].second << " (Similarity: " << distances[i].first << ")" << std::endl;
    }

    return 0;
}

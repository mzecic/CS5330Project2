/*
Name: Matej Zecic
Date: January 2025, CS5330
Description: This program calculates the similarity between an input image and all of the images in the database using the specified criteria.
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

// Function to compute Euclidean distance between two feature vectors
float compute_distance(const std::vector<float>& v1, const std::vector<float>& v2) {
    float distance = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        distance += (v1[i] - v2[i]) * (v1[i] - v2[i]);
    }
    return distance;
}

// Function to compute Cosine Similarity
float cosine_similarity(const std::vector<float>& v1, const std::vector<float>& v2) {
    float dot_product = 0.0, norm_v1 = 0.0, norm_v2 = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        dot_product += v1[i] * v2[i];
        norm_v1 += v1[i] * v1[i];
        norm_v2 += v2[i] * v2[i];
    }
    return dot_product / (sqrt(norm_v1) * sqrt(norm_v2));  // Cosine similarity formula
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

    for (size_t i = 0; i < v1.size() - 8; ++i) {
        intersection_color += std::min(v1[i], v2[i]);
    }

    for (size_t j = v1.size() - 8; j < v1.size(); ++j) {
        intersection_texture += std::min(v1[j], v2[j]);
    }

    float intersection = (intersection_color + intersection_texture) / 2;
    return intersection;
}

// Function to select the appropriate feature extraction method
std::vector<float> extract_features(const char* image_path, const char* feature_set) {
    if (strcmp(feature_set, "dnembeddings") == 0) {
        std::string image_path_str = image_path;
        size_t last_slash_pos = image_path_str.find_last_of("/\\");
        std::string image_name = image_path_str.substr(last_slash_pos + 1);

        std::vector<char *> filenames;
        std::vector<std::vector<float>> data;
        if (read_image_data_csv("../vectors/ResNet18_olym.csv", filenames, data, 0) != 0) {
            std::cerr << "Failed to read the feature vector file." << std::endl;
            return {};
        }

        for (size_t i = 0; i < filenames.size(); i++) {
            if (strcmp(filenames[i], image_name.c_str()) == 0) {
                return data[i];
            }
        }
    }
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

    std::vector<float> target_feature = extract_features(target_image, feature_set);
    if (target_feature.empty()) {
        std::cerr << "Failed to extract features from the target image." << std::endl;
        return -1;
    }

    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(feature_vector_file, filenames, data, 0) != 0) {
        std::cerr << "Failed to read the feature vector file." << std::endl;
        return -1;
    }

    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        float distance;
        if (strcmp(feature_set, "dnembeddings") == 0) {
            distance = cosine_similarity(target_feature, data[i]);
        } else {
            distance = compute_distance(target_feature, data[i]);
        }
        distances.emplace_back(distance, filenames[i]);
    }

    // Sorting for cosine similarity (higher is better) and Euclidean distance (lower is better)
    if (strcmp(feature_set, "dnembeddings") == 0) {
        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.first > b.first;  // Higher similarity is better
        });
    } else {
        std::sort(distances.begin(), distances.end(), [](const auto& a, const auto& b) {
            return a.first < b.first;  // Lower distance is better
        });
    }

    std::cout << "Top " << N << " matches (higher is better for similarity, lower for distance):" << std::endl;
    for (int i = 0; i < N && i < distances.size(); ++i) {
        std::cout << distances[i].second << " (Similarity: " << distances[i].first << ")" << std::endl;
    }

    return 0;
}

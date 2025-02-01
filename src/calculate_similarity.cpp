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

// Function to compute histogram intersection
float histogram_intersection(const std::vector<float>& v1, const std::vector<float>& v2) {
    float intersection = 0.0;
    std::cout << "Histogram Intersection Debug: \n";
    for (size_t i = 0; i < v1.size(); ++i) {
        float min_val = std::min(v1[i], v2[i]);
        intersection += min_val;
        std::cout << "Bin " << i << ": v1=" << v1[i] << ", v2=" << v2[i] << ", min=" << min_val << "\n";
    }
    std::cout << "Final Intersection Score: " << intersection << std::endl;
    return intersection;
}

// Function to select the appropriate feature extraction method
std::vector<float> extract_features(const char* image_path, const char* feature_set) {
    if (strcmp(feature_set, "7x7") == 0) {
        return extract_7x7(image_path); // 7x7 patch extraction
    } else if (strcmp(feature_set, "histogram_matching") == 0) {
        return compute_histogram(image_path); // 2D histogram extraction
    } else {
        std::cerr << "Error: Unknown feature set \"" << feature_set << "\"." << std::endl;
        return {};
    }
}

int main(int argc, char *argv[]) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <target_image> <feature_set> <feature_vector_file> <N>" << std::endl;
        std::cerr << "Feature sets supported: 7x7 (default 7x7 patch extraction)" << std::endl;
        return -1;
    }

    char *target_image = argv[1];
    char *feature_set = argv[2];  // Used to determine the feature extraction method
    char *feature_vector_file = argv[3];
    int N = argv[4] ? atoi(argv[4]) : 3;  // Number of top matches to display

    // Step 1: Extract features from the target image based on the specified feature set
    std::vector<float> target_feature = extract_features(target_image, feature_set);
    if (target_feature.empty()) {
        std::cerr << "Failed to extract features from the target image using feature set: " << feature_set << std::endl;
        return -1;
    }

    // Step 2: Read the feature vectors from the CSV file
    std::vector<char *> filenames;
    std::vector<std::vector<float>> data;
    if (read_image_data_csv(feature_vector_file, filenames, data, 0) != 0) {
        std::cerr << "Failed to read the feature vector file." << std::endl;
        return -1;
    }

    // Step 3: Compare feature vectors and compute distances
    std::vector<std::pair<float, std::string>> distances;
    for (size_t i = 0; i < data.size(); ++i) {
        float distance;
        if (strcmp(feature_set, "7x7") == 0) {
            distance = compute_distance(target_feature, data[i]);
        } else if (strcmp(feature_set, "histogram_matching") == 0) {
            distance = histogram_intersection(target_feature, data[i]);
        }
        distances.emplace_back(distance, filenames[i]); // ✅ Fixed placement of this line
    }

    // Step 4: Sort by similarity (higher values = better match)
    std::sort(distances.rbegin(), distances.rend()); // ✅ Sorting in descending order

    // Print top N matches
    std::cout << "Top " << N << " matches (higher is better):" << std::endl;
    for (int i = 1; i <= N && i < distances.size(); ++i) {
        std::cout << distances[i].second << " (Similarity: " << distances[i].first << ")" << std::endl;
    }

    return 0;
}

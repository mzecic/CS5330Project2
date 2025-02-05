/*
Name: Matej Zecic
Date: Spring 2025
Description: This program calculates different similarity measures for an image used to determine similarities between images.
*/

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include "../include/filters.h"

/**
 * Extracts a 7x7 patch of pixel values from the center of an image.
 *
 * @param filename The path to the image file.
 * @return A vector of float values representing the pixel data of the 7x7 patch.
 *         If the image cannot be loaded or is too small, an empty vector is returned.
 *
 * The function performs the following steps:
 * 1. Loads the image from the specified file.
 * 2. Checks if the image was loaded successfully. If not, prints an error message and returns an empty vector.
 * 3. Checks if the image dimensions are at least 7x7. If not, prints an error message and returns an empty vector.
 * 4. Extracts a 7x7 patch of pixel values from the center of the image.
 *    - If the image has 3 channels (e.g., RGB), extracts each channel value for each pixel.
 *    - If the image has 1 channel (e.g., grayscale), extracts the pixel value.
 *    - If the image has an unsupported number of channels, prints an error message and returns an empty vector.
 * 5. Returns the vector of extracted pixel values.
 */
std::vector<float> extract_7x7(const char *filename) {
  cv::Mat image;
  image = cv::imread(filename);
  std::vector<float> image_data;

  if (image.empty()) {
        std::cerr << "Failed to load image: " << filename << std::endl;
        return image_data; // Return empty vector
  }

  if (image.rows < 7 || image.cols < 7) {
        std::cerr << "Image is too small for a 7x7 patch!" << std::endl;
        return image_data; // Return empty vector
  }

  for (int i = (image.rows / 2) - 3; i <= (image.rows / 2) + 3; i++) {
    for (int j = (image.rows / 2) - 3; j <= (image.rows / 2) + 3; j++) {
      if (image.channels() == 3) {
        for (int k = 0; k < 3; k++) {
          image_data.push_back(image.at<cv::Vec3b>(i, j)[k]);
        }
      } else if (image.channels() == 1) {
        image_data.push_back(image.at<uchar>(i, j));
      } else {
        std::cerr << "Image has an unsupported number of channels!" << std::endl;
        return image_data;
      }
    }
  }
  return image_data;
}

/**
 * Computes a 2D histogram for the hue and saturation channels of an image.
 *
 * @param filename The path to the image file.
 * @param hist_size The number of bins for the histogram (default is 16).
 * @return A vector of floats representing the normalized histogram.
 *
 * The function performs the following steps:
 * 1. Converts the input image from BGR to HSV color space.
 * 2. Initializes the histogram parameters, including the number of bins for hue and saturation.
 * 3. Defines the range for hue (0-180) and saturation (0-256).
 * 4. Computes the total number of pixels in the image.
 * 5. Iterates over each pixel in the HSV image:
 *    - Extracts the hue and saturation values.
 *    - Computes the bin index for the hue and saturation values.
 *    - Increments the corresponding bin in the histogram vector.
 * 6. Normalizes the histogram by dividing each bin value by the total number of pixels.
 * 7. Returns the normalized histogram vector.
 */
std::vector<float> compute_histogram(const char *filename, int hist_size = 8) {  // Reduce bins from 16 to 8
    cv::Mat hsv, image;
    image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not read image " << filename << std::endl;
        return {};
    }
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Define histogram parameters
    int h_bins = hist_size, s_bins = hist_size;
    int total_bins = h_bins * s_bins;
    std::vector<float> hist_vector(total_bins, 0.0f);

    // Define bin ranges
    float h_range = 180.0f;
    float s_range = 256.0f;

    // Compute histogram manually
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            int h = pixel[0];
            int s = pixel[1];

            // Corrected bin index computation
            int h_index = static_cast<int>((h / h_range) * h_bins);
            int s_index = static_cast<int>((s / s_range) * s_bins);
            int bin_index = h_index * s_bins + s_index;

            hist_vector[bin_index] += 1.0f;  // Count occurrences
        }
    }

    // Normalize using total sum, not total pixels
    float sum = 0.0;
    for (float val : hist_vector) {
        sum += val;
    }
    if (sum > 0) {
        for (float &val : hist_vector) {
            val /= sum;  // Normalize by total count
        }
    }

    return hist_vector;
}

/**
 * Computes histograms for the top and bottom halves of an image and combines them into a single feature vector.
 *
 * @param filename The path to the image file.
 * @param hist_size The number of bins for the histogram (default is 8).
 * @return A vector of floats representing the concatenated histograms of the top and bottom halves of the image.
 *
 * The function performs the following steps:
 * 1. Loads the image from the specified file and checks if it is empty.
 * 2. Converts the image from BGR to RGB color space.
 * 3. Splits the image into top and bottom halves.
 * 4. Splits each half into its respective RGB channels.
 * 5. Computes histograms for the R, G, and B channels of the top half.
 * 6. Computes histograms for the R, G, and B channels of the bottom half.
 * 7. Normalizes each histogram to the range [0, 1].
 * 8. Concatenates the histograms of the top and bottom halves into a single feature vector.
 * 9. Returns the concatenated feature vector.
 */
std::vector<float> compute_multi_histogram(const char* filename, int hist_size = 8) {

    cv::Mat image = cv::imread(filename);

    // check if image is empty
    if (image.empty()) {
        std::cerr << "Error: Could not read image " << filename << std::endl;
        return {};
    }

    // convert to rgb
    cv::Mat rgb_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);

    // split the image into top and bottom half
    int height = image.rows;
    int width = image.cols;
    cv::Mat top_half = rgb_image(cv::Rect(0, 0, width, height / 2));
    cv::Mat bottom_half = rgb_image(cv::Rect(0, height / 2, width, height / 2));

    std::vector<cv::Mat> channels;
    cv::split(top_half, channels);
    cv::Mat hist_top_r, hist_top_g, hist_top_b;

    // define parameters for histogram calculation
    float range[] = {0, 256};
    const float* hist_range = {range};

    // compute histograms for top half
    cv::calcHist(&channels[0], 1, 0, cv::Mat(), hist_top_r, 1, &hist_size, &hist_range);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), hist_top_g, 1, &hist_size, &hist_range);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), hist_top_b, 1, &hist_size, &hist_range);

    // compute for bottom half
    cv::split(bottom_half, channels);
    cv::Mat hist_bottom_r, hist_bottom_g, hist_bottom_b;

    cv::calcHist(&channels[0], 1, 0, cv::Mat(), hist_bottom_r, 1, &hist_size, &hist_range);
    cv::calcHist(&channels[1], 1, 0, cv::Mat(), hist_bottom_g, 1, &hist_size, &hist_range);
    cv::calcHist(&channels[2], 1, 0, cv::Mat(), hist_bottom_b, 1, &hist_size, &hist_range);

    cv::normalize(hist_top_r, hist_top_r, 0, 1, cv::NORM_MINMAX);
    cv::normalize(hist_top_g, hist_top_g, 0, 1, cv::NORM_MINMAX);
    cv::normalize(hist_top_b, hist_top_b, 0, 1, cv::NORM_MINMAX);

    cv::normalize(hist_bottom_r, hist_bottom_r, 0, 1, cv::NORM_MINMAX);
    cv::normalize(hist_bottom_g, hist_bottom_g, 0, 1, cv::NORM_MINMAX);
    cv::normalize(hist_bottom_b, hist_bottom_b, 0, 1, cv::NORM_MINMAX);

    std::vector<float> feature_vector;
    feature_vector.insert(feature_vector.end(), hist_top_r.begin<float>(), hist_top_r.end<float>());
    feature_vector.insert(feature_vector.end(), hist_top_g.begin<float>(), hist_top_g.end<float>());
    feature_vector.insert(feature_vector.end(), hist_top_b.begin<float>(), hist_top_b.end<float>());

    feature_vector.insert(feature_vector.end(), hist_bottom_r.begin<float>(), hist_bottom_r.end<float>());
    feature_vector.insert(feature_vector.end(), hist_bottom_g.begin<float>(), hist_bottom_g.end<float>());
    feature_vector.insert(feature_vector.end(), hist_bottom_b.begin<float>(), hist_bottom_b.end<float>());

    return feature_vector;
}


/**
 * Computes a combined feature vector of color and texture histograms for an image.
 *
 * @param filename The path to the image file.
 * @param hist_size The number of bins for the histograms (default is 8).
 * @return A vector of floats representing the concatenated color and texture histograms.
 *
 * The function performs the following steps:
 * 1. Loads the image from the specified file and checks if it is empty.
 * 2. Converts the image from BGR to RGB and grayscale color spaces.
 * 3. Splits the RGB image into its red, green, and blue channels.
 * 4. Initializes histograms for the color channels.
 * 5. Computes the color histograms for the red, green, and blue channels.
 * 6. Normalizes the color histograms by the total number of pixels.
 * 7. Computes the Sobel gradients for the grayscale image to obtain gradient magnitudes and directions.
 * 8. Finds the maximum gradient magnitude.
 * 9. Initializes and computes the texture histogram based on gradient magnitudes.
 * 10. Normalizes the texture histogram by the total number of pixels.
 * 11. Combines the color and texture histograms into a single feature vector.
 * 12. Returns the combined feature vector.
 */
std::vector<float> compute_texture_and_color(const char* filename, int hist_size = 8) {
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not open image " << filename << std::endl;
        return {};
    }

    cv::Mat rgb_image, grayscale_image;
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
    cv::cvtColor(image, grayscale_image, cv::COLOR_BGR2GRAY);

    // Split into color channels
    std::vector<cv::Mat> channels;
    cv::split(rgb_image, channels);
    cv::Mat red_channel = channels[0];
    cv::Mat green_channel = channels[1];
    cv::Mat blue_channel = channels[2];

    // Initialize histograms
    std::vector<float> histogram(3 * hist_size, 0.0f);
    float bin_width = 256.0 / hist_size;
    int total_pixels = image.rows * image.cols;

    // Compute color histograms
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            int red_bin = red_channel.at<uchar>(i, j) / bin_width;
            int green_bin = green_channel.at<uchar>(i, j) / bin_width;
            int blue_bin = blue_channel.at<uchar>(i, j) / bin_width;

            histogram[red_bin]++;
            histogram[green_bin + hist_size]++;
            histogram[blue_bin + 2 * hist_size]++;
        }
    }

    // Normalize color histogram
    for (int i = 0; i < 3 * hist_size; i++) {
        histogram[i] /= total_pixels;
    }

    // Compute Sobel gradients
    cv::Mat sobel_x, sobel_y, gradient_magnitude, gradient_direction;
    cv::Sobel(grayscale_image, sobel_x, CV_64F, 1, 0, 3);
    cv::Sobel(grayscale_image, sobel_y, CV_64F, 0, 1, 3);
    cv::magnitude(sobel_x, sobel_y, gradient_magnitude);
    cv::phase(sobel_x, sobel_y, gradient_direction, true);
    gradient_direction *= (180.0 / CV_PI);

    // Find max magnitude
    double minVal, max_magnitude;
    cv::minMaxLoc(gradient_magnitude, &minVal, &max_magnitude);

    if (max_magnitude == 0) {
        return {};  // No texture info available
    }

    float texture_bin_width = max_magnitude / hist_size;
    std::vector<float> texture_histogram(hist_size, 0.0f);

    // Compute texture histogram
    for (int i = 0; i < grayscale_image.rows; i++) {
        for (int j = 0; j < grayscale_image.cols; j++) {
            int bin = std::min(static_cast<int>(gradient_magnitude.at<double>(i, j) / texture_bin_width), hist_size - 1);
            texture_histogram[bin]++;
        }
    }

    // Normalize texture histogram
    for (int i = 0; i < hist_size; i++) {
        texture_histogram[i] /= total_pixels;
    }

    // Combine feature vectors
    std::vector<float> feature_vector;
    feature_vector.insert(feature_vector.end(), histogram.begin(), histogram.end());
    feature_vector.insert(feature_vector.end(), texture_histogram.begin(), texture_histogram.end());

    return feature_vector;
}

/**
 * Extracts features from an image to identify banana-like objects.
 *
 * @param filename The path to the image file.
 * @return A vector of floats representing the extracted features.
 *
 * The function performs the following steps:
 * 1. Loads the image from the specified file and checks if it is empty.
 * 2. Converts the image to HSV color space and thresholds for yellow color to create a mask.
 * 3. Applies morphological operations to reduce noise in the mask.
 * 4. Finds contours in the mask.
 * 5. Initializes feature values including total area, largest aspect ratio, Hu moments, mean HSV, and number of contours.
 * 6. Processes each contour to compute area, aspect ratio, and convexity defects.
 * 7. Computes Hu moments for valid contours.
 * 8. Computes texture score using the Laplacian variance.
 * 9. Applies a log transformation to the Hu moments while preserving their sign.
 * 10. Normalizes the features.
 * 11. If no valid banana-like contours are found, returns a default feature vector.
 * 12. Constructs and returns the feature vector.
 */
std::vector<float> banana_extraction(const char* filename) {
    // Load the image
    cv::Mat image = cv::imread(filename);
    if (image.empty()) {
        printf("Error: Could not load image %s\n", filename);
        return {};
    }

    // Convert to HSV and threshold for yellow (banana) color
    cv::Mat hsv, mask;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);
    cv::Scalar lower_yellow(20, 110, 90);
    cv::Scalar upper_yellow(40, 255, 255);
    cv::inRange(hsv, lower_yellow, upper_yellow, mask);

    // Apply morphological operations to reduce noise
    cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::Mat::ones(5, 5, CV_8U));
    cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::Mat::ones(7, 7, CV_8U));

    // Find contours in the mask
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(mask, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Initialize feature values
    float total_area = 0.0f;
    float largest_aspect_ratio = 0.0f;
    std::vector<double> hu_moments(7, 0.0);
    cv::Scalar mean_hsv = cv::mean(hsv, mask);
    int num_contours = 0;

    // Process each contour
    for (const auto& contour : contours) {
        float area = cv::contourArea(contour);

        // Discard small contours (noise)
        if (area < 1200.0f)
            continue;

        // Use a rotated rectangle to obtain a more robust aspect ratio (handles rotation)
        cv::RotatedRect rotated_rect = cv::minAreaRect(contour);
        float width = rotated_rect.size.width;
        float height = rotated_rect.size.height;
        if (height == 0) height = 1;  // avoid division by zero

        // Compute aspect ratio as the ratio of the larger to the smaller side
        float aspect_ratio = (width > height) ? width / height : height / width;

        // Only consider contours with aspect ratios typical of a banana
        if (aspect_ratio < 1.5 || aspect_ratio > 7)
            continue;

        total_area += area;
        num_contours++;
        largest_aspect_ratio = std::max(largest_aspect_ratio, aspect_ratio);

        // Check for convexity defects (bananas should have a relatively smooth outline)
        std::vector<int> hull;
        cv::convexHull(contour, hull, false, false);
        if (hull.size() > 3) {
            std::vector<cv::Vec4i> defects;
            cv::convexityDefects(contour, hull, defects);
            if (defects.size() > 3)
                continue;  // Skip contours with many defects
        }

        // Compute Hu Moments from the contour's moments
        cv::Moments moments = cv::moments(contour);
        if (moments.m00 > 0) {
            std::vector<double> tempHu(7, 0.0);
            cv::HuMoments(moments, tempHu.data());
            hu_moments = tempHu;  // In this example, we simply take the last valid contour's Hu Moments.
        }
    }

    // Compute texture score using the Laplacian variance
    cv::Mat gray, laplacian;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    cv::Laplacian(gray, laplacian, CV_64F);
    cv::Scalar mean_lap, stddev_lap;
    cv::meanStdDev(laplacian, mean_lap, stddev_lap);
    float texture_score = std::min(static_cast<float>(stddev_lap[0]) / 100.0f, 1.0f);

    // Apply a log transformation to the Hu Moments while preserving their sign
    for (int i = 0; i < 7; i++) {
        double sign = (hu_moments[i] < 0) ? -1.0 : 1.0;
        hu_moments[i] = -sign * log10(std::abs(hu_moments[i]) + 1e-6);
    }

    // Normalize the features
    total_area = std::min(total_area / 50000.0f, 1.0f);
    largest_aspect_ratio = std::min(largest_aspect_ratio / 10.0f, 1.0f);
    float norm_num_contours = std::min(num_contours / 10.0f, 1.0f);

    // If no valid banana-like contours are found, return a default feature vector (14 elements)
    if (num_contours == 0) {
        printf("Warning: No banana-like contours detected in %s\n", filename);
        return { 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f, 0.01f,
                 0.5f, 0.5f, 0.5f, 0.01f, 0.01f };
    }

    // Construct the feature vector
    std::vector<float> feature_vector = {
        total_area,
        largest_aspect_ratio,
        static_cast<float>(hu_moments[0]),
        static_cast<float>(hu_moments[1]),
        static_cast<float>(hu_moments[2]),
        static_cast<float>(hu_moments[3]),
        static_cast<float>(hu_moments[4]),
        static_cast<float>(hu_moments[5]),
        static_cast<float>(hu_moments[6]),
        static_cast<float>(mean_hsv[0] / 180.0f),
        static_cast<float>(mean_hsv[1] / 255.0f),
        static_cast<float>(mean_hsv[2] / 255.0f),
        norm_num_contours,
        texture_score
    };

    return feature_vector;
}

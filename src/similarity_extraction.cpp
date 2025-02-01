#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include <vector>

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
 * @param histSize The number of bins for the histogram (default is 16).
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
std::vector<float> compute_histogram(const char *filename, int histSize = 8) {  // Reduce bins from 16 to 8
    cv::Mat hsv, image;
    image = cv::imread(filename);
    if (image.empty()) {
        std::cerr << "Error: Could not read image " << filename << std::endl;
        return {};
    }
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Define histogram parameters
    int hBins = histSize, sBins = histSize;
    int totalBins = hBins * sBins;
    std::vector<float> histVector(totalBins, 0.0f);

    // Define bin ranges
    float hRange = 180.0f;
    float sRange = 256.0f;

    // Compute histogram manually
    for (int y = 0; y < hsv.rows; ++y) {
        for (int x = 0; x < hsv.cols; ++x) {
            cv::Vec3b pixel = hsv.at<cv::Vec3b>(y, x);
            int h = pixel[0];
            int s = pixel[1];

            // Corrected bin index computation
            int hIndex = static_cast<int>((h / hRange) * hBins);
            int sIndex = static_cast<int>((s / sRange) * sBins);
            int binIndex = hIndex * sBins + sIndex;

            histVector[binIndex] += 1.0f;  // Count occurrences
        }
    }

    // Normalize using total sum, not total pixels
    float sum = 0.0;
    for (float val : histVector) {
        sum += val;
    }
    if (sum > 0) {
        for (float &val : histVector) {
            val /= sum;  // Normalize by total count
        }
    }

    return histVector;
}

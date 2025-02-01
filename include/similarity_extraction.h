/*
Name: Matej Zecic
Date: Spring 2025
Description: This program calculates different similarity measures for an image used to determine similarities between images.
*/

#include <iostream>
#include <vector>
#include <cstring>
#include <cstdlib>
#include "opencv2/opencv.hpp"


// Function to extract a 7x7 patch of pixel values from the center of an image
std::vector<float> extract_7x7(const char *filename);

// Function to compute a 2D histogram for the hue and saturation channels of an image
std::vector<float> compute_histogram(const char *filename, int histSize = 16);

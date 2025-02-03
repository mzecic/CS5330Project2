/*
Name: Matej Zecic
Date: Spring 2025
Description: This program holds the sobel filter functions.
*/

#include <opencv2/opencv.hpp>

// Function that applies sobel x filter to an image
int sobelX3x3(cv::Mat& src, cv::Mat& dst);

// Function that applies sobel y filter to an image
int sobelY3x3(cv::Mat& src, cv::Mat& dst);

// Function that applies magnitude filter to an image
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst);

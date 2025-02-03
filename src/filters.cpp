/*
Name: Matej Zecic
Date: Spring 2025
Description: This program holds the sobel filter functions.
*/

#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "../include/filters.h"


// Function that applies sobel x filter to an image
// Parameters:
//      src: input image
//      dst: destination image
// Returns:
//      int
int sobelX3x3(cv::Mat& src, cv::Mat& dst) {
    // creating an image of the same size as the source image, but with 16-bit signed channels
    dst.create(src.size(), CV_16SC3);
    cv::Mat paddedSrc;
    cv::copyMakeBorder(src, paddedSrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    int horizontalKernel[3] = {-1, 0, +1};

    // horizontal pass
    for (int i = 0; i < paddedSrc.rows - 1; i++) {
        for (int j = 1; j < paddedSrc.cols - 1; j++) {
            short sumBlue = 0, sumGreen = 0, sumRed = 0;

            for (int k = -1; k <= 1; k++) {
                cv::Vec3b intensity = src.at<cv::Vec3b>(i, j + k);
                sumBlue += intensity[0] * horizontalKernel[k + 1];
                sumGreen += intensity[1] * horizontalKernel[k + 1];
                sumRed += intensity[2] * horizontalKernel[k + 1];
            }

            temp.at<cv::Vec3s>(i, j) = cv::Vec3s(sumBlue, sumGreen, sumRed);
        }
    }

    int verticalKernel[3] = {1, 2, 1};

    // vertical pass
    for (int i = 1; i < paddedSrc.rows; i++) {
        for (int j = 0; j < paddedSrc.cols; j++) {
            short sumBlue = 0, sumGreen = 0, sumRed = 0;

            for (int k = -1; k <= 1; k++) {
                cv::Vec3s intensity = temp.at<cv::Vec3s>(i + k, j);
                sumBlue += intensity[0] * verticalKernel[k + 1];
                sumGreen += intensity[1] * verticalKernel[k + 1];
                sumRed += intensity[2] * verticalKernel[k + 1];
            }

            dst.at<cv::Vec3s>(i, j) = cv::Vec3s(sumBlue, sumGreen, sumRed);
        }
    }

    return 0;
}


// Function that applies sobel y filter to an image
// Parameters:
//      src: input image
//      dst: destination image
// Returns:
//      int
int sobelY3x3(cv::Mat& src, cv::Mat& dst) {
    // creating an image of the same size as the source image, but with 16-bit signed channels
    dst.create(src.size(), CV_16SC3);
    cv::Mat paddedSrc;
    cv::copyMakeBorder(src, paddedSrc, 1, 1, 1, 1, cv::BORDER_REPLICATE);

    cv::Mat temp;
    temp.create(src.size(), CV_16SC3);

    int horizontalKernel[3] = {1, 2, 1};

    // horizontal pass
    for (int i = 0; i < src.rows; i++) {
        for (int j = 1; j < src.cols - 1; j++) {
            short sumBlue = 0, sumGreen = 0, sumRed = 0;

            for (int k = -1; k <= 1; k++) {
                cv::Vec3b intensity = src.at<cv::Vec3b>(i, j + k);
                sumBlue += intensity[0] * horizontalKernel[k + 1];
                sumGreen += intensity[1] * horizontalKernel[k + 1];
                sumRed += intensity[2] * horizontalKernel[k + 1];
            }

            temp.at<cv::Vec3s>(i, j) = cv::Vec3s(sumBlue, sumGreen, sumRed);
        }
    }

    int verticalKernel[3] = {-1, 0, +1};

    // vertical pass
    for (int i = 1; i < src.rows - 1; i++) {
        for (int j = 0; j < src.cols; j++) {
            short sumBlue = 0, sumGreen = 0, sumRed = 0;

            for (int k = -1; k <= 1; k++) {
                cv::Vec3s intensity = temp.at<cv::Vec3s>(i + k, j);
                sumBlue += intensity[0] * verticalKernel[k + 1];
                sumGreen += intensity[1] * verticalKernel[k + 1];
                sumRed += intensity[2] * verticalKernel[k + 1];
            }

            dst.at<cv::Vec3s>(i, j) = cv::Vec3s(sumBlue, sumGreen, sumRed);
        }
    }

    return 0;
}


// Function that applies magnitude filter to an image
// Parameters:
//      sx: unsigned short 3 channel input image
//      sy: unsigned short 3 channel input image
//      dst: destination image
// Returns:
//      int
int magnitude(cv::Mat &sx, cv::Mat &sy, cv::Mat &dst) {
    // check if input is valid
    if (sx.empty() || sy.empty() || sx.size() != sy.size() || sx.type() != sy.type()) {
        return -1;
    }

    dst = cv::Mat::zeros(sx.size(), CV_8UC3); // Unsigned char 3-channel image

    // main loop, iterate through rows, cols, and channels
    for (int row = 0; row < sx.rows; ++row) {
        for (int col = 0; col < sx.cols; ++col) {
            for (int ch = 0; ch < 3; ++ch) {
                short val_sx = sx.at<cv::Vec3s>(row, col)[ch];
                short val_sy = sy.at<cv::Vec3s>(row, col)[ch];

                // compute gradient magnitude
                float magnitude = std::sqrt(val_sx * val_sx + val_sy * val_sy);

                // cormalize and convert to uchar
                dst.at<cv::Vec3b>(row, col)[ch] = static_cast<uchar>(
                    std::min(255.0f, magnitude));
            }
        }
    }
    return 0;
}

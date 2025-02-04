/*
  Bruce A. Maxwell
  S21

  Sample code to identify image fils in a directory
*/
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <dirent.h>
#include "opencv2/opencv.hpp"
#include <vector>
#include "../include/csv_util.h"
#include "../include/similarity_extraction.h"


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
// std::vector<float> extract_7x7(char *filename) {
//   cv::Mat image;
//   image = cv::imread(filename);
//   std::vector<float> image_data;

//   if (image.empty()) {
//         std::cerr << "Failed to load image: " << filename << std::endl;
//         return image_data; // Return empty vector
//   }

//   if (image.rows < 7 || image.cols < 7) {
//         std::cerr << "Image is too small for a 7x7 patch!" << std::endl;
//         return image_data; // Return empty vector
//   }

//   for (int i = (image.rows / 2) - 3; i <= (image.rows / 2) + 3; i++) {
//     for (int j = (image.rows / 2) - 3; j <= (image.rows / 2) + 3; j++) {
//       if (image.channels() == 3) {
//         for (int k = 0; k < 3; k++) {
//           image_data.push_back(image.at<cv::Vec3b>(i, j)[k]);
//         }
//       } else if (image.channels() == 1) {
//         image_data.push_back(image.at<uchar>(i, j));
//       } else {
//         std::cerr << "Image has an unsupported number of channels!" << std::endl;
//         return image_data;
//       }
//     }
//   }
//   return image_data;
// }


/*
  Given a directory on the command line, scans through the directory for image files.

  Prints out the full path name for each file.  This can be used as an argument to fopen or to cv::imread.
 */
int main(int argc, char *argv[]) {
  char dirname[256];
  char buffer[256];
  FILE *fp;
  DIR *dirp;
  struct dirent *dp;
  int i;

  // check for sufficient arguments
  if( argc < 2) {
    printf("usage: %s <directory path>\n", argv[0]);
    exit(-1);
  }

  // get the directory path
  strcpy(dirname, argv[1]);
  printf("Processing directory %s\n", dirname );

  // open the directory
  dirp = opendir( dirname );
  if( dirp == NULL) {
    printf("Cannot open directory %s\n", dirname);
    exit(-1);
  }

  // loop over all the files in the image file listing
  while( (dp = readdir(dirp)) != NULL ) {

    // check if the file is an image
    if( strstr(dp->d_name, ".jpg") ||
	strstr(dp->d_name, ".png") ||
	strstr(dp->d_name, ".ppm") ||
	strstr(dp->d_name, ".tif") ) {

      printf("processing image file: %s\n", dp->d_name);

      // build the overall filename
      strcpy(buffer, dirname);
      strcat(buffer, "/");
      strcat(buffer, dp->d_name);

      printf("full path name: %s\n", buffer);

      // extract feature set from each image based on the argument
      char *feature_set = argv[2];
      std::vector<float> feature;

      if (strcmp(feature_set, "7x7") == 0) {
        feature = extract_7x7(buffer); // extract 7x7 patch in the middle and save to a vector
      } else if (strcmp(feature_set, "histogram_matching") == 0) {
        feature = compute_histogram(buffer); // compute 2d histogram and save to a vector
      } else if (strcmp(feature_set, "multi_histogram") == 0) {
        feature = compute_multi_histogram(buffer); // compute multi-histogram and save to a vector
      } else if (strcmp(feature_set, "texture_and_color") == 0) {
        feature = compute_texture_and_color(buffer); // compute texture and color features and save to a vector
      } else if (strcmp(feature_set, "banana_detection") == 0) {
        feature = banana_extraction(buffer);
      }
      else {
        std::cerr << "Error: Unknown feature set \"" << feature_set << "\"." << std::endl;
        return -1;
      }

      // append the feature vector to a csv file
      char final_path[256];
      snprintf(final_path, sizeof(final_path), "../vectors/%s.csv", feature_set);
      append_image_data_csv(final_path, buffer, feature);
    }
  }

  printf("Terminating\n");

  return(0);
}

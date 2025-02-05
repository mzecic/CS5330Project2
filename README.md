# Image Similarity Project

### Contributions

- Developers: Matej Zecic

## Overview

This project calculates different similarity measures for images to determine similarities between them. It includes functionalities for reading and writing CSV files, extracting image features, and computing various similarity metrics.

#### This program was compiled and ran on MacOS Sequoia 15.3 using Vs Code as IDE

## Features

- **CSV Utility**: Functions for reading and writing CSV files with image filenames and feature vectors.
- **Similarity Extraction**: Functions to compute different similarity measures such as Euclidean distance, Cosine similarity, Mahalanobis distance, and histogram intersection.
- **Feature Extraction**: Functions to extract features from images based on different methods.

## Dependencies

- OpenCV
- Protobuf
- Homebrew (for managing dependencies on macOS)

## Installation

1. **Clone the repository**:

   ```sh
   git clone https://github.com/yourusername/cs5330_project2.git
   cd cs5330_project2
   ```

2. **Install dependencies**:

   ```sh
   brew install opencv protobuf
   ```

3. **Build the project**:
   ```sh
   cd src
   make
   ```

## Usage

### Calculate Similarity

To calculate the similarity between images, use the `calculate_similarity` executable:

```sh
../bin/calculate_similarity <target_image> <feature_set> <feature_vector_file> <N>
```

- `<target_image>`: Path to the target image.
- `<feature_set>`: Feature extraction method (e.g., `7x7`, `histogram_matching`, `multi_histogram`, `texture_and_color`, `banana_detection`).
- `<feature_vector_file>`: Path to the CSV file containing feature vectors.
- `<N>`: Number of top matches to display.

### Example

```sh
../bin/calculate_similarity ../olympus/<target>.jpg histogram_matching ../vectors/histogram_matching.csv 5
```

## File Structure

- `src/`: Source code files.
- `include/`: Header files.
- `bin/`: Compiled executables.
- `images/`: Sample images.
- `features.csv`: Example CSV file with feature vectors.

## Contributing

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

This project is open source and available for both personal and commercial use under the [Northeastern University License](LICENSE). Feel free to use, modify, and distribute the code as long as the original license is included in any distributions.

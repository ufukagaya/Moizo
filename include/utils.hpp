#ifndef UTILS_HPP
#define UTILS_HPP

#include <opencv2/opencv.hpp>

// Function prototypes for utility functions
cv::Mat applyGaussianBlur(const cv::Mat& inputImage, int kernelSize);
cv::Mat convertToGrayscale(const cv::Mat& inputImage);
void displayImage(const std::string& windowName, const cv::Mat& image);

#endif // UTILS_HPP
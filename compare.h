#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat addSaltPepperNoise(const cv::Mat& src, double density);
cv::Mat addGaussianNoise(const cv::Mat& src, double sigma);
cv::Mat averageFilter(const cv::Mat& src, int winSize);
double calculatePSNR(const cv::Mat& orig, const cv::Mat& proc);
void runNoiseComparison(const cv::Mat& original, const std::string& outDir);
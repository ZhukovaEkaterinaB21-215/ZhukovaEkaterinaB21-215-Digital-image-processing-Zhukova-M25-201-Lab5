#pragma once
#include <opencv2/opencv.hpp>

static void manualSort(std::vector<uchar>& arr);
static void manualNthElement(std::vector<uchar>& arr, int k);

cv::Mat rankFilter(const cv::Mat& src, int k, const cv::Mat& mask);

cv::Mat medianFilter(const cv::Mat& src, const cv::Mat& mask);

cv::Mat createCrossMask(); 
cv::Mat createRectMask(int size);
cv::Mat createDiamondMask();
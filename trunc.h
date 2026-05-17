#pragma once
#include <opencv2/opencv.hpp>

static void manualSort(std::vector<uchar>& arr);
cv::Mat truncatedMeanFilter(const cv::Mat& src, int winSize, int d);
#pragma once
#include <opencv2/opencv.hpp>
#include <string>

cv::Mat padImageReplicate(const cv::Mat& src, int pad);
void ensureOutputDir(const std::string& dir);
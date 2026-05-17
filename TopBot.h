#pragma once
#include <opencv2/opencv.hpp>

cv::Mat morphTopHat(const cv::Mat& src, int winSize);
cv::Mat morphBotHat(const cv::Mat& src, int winSize);
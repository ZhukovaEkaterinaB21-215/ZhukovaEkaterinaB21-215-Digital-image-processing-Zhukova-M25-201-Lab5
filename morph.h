#pragma once
#include <opencv2/opencv.hpp>

static uchar getMinWindow(const cv::Mat& padded, int py, int px, int pad);
static uchar getMaxWindow(const cv::Mat& padded, int py, int px, int pad);

cv::Mat morphErosion(const cv::Mat& src, int winSize);
cv::Mat morphDilation(const cv::Mat& src, int winSize);
cv::Mat morphOpening(const cv::Mat& src, int winSize);
cv::Mat morphClosing(const cv::Mat& src, int winSize);
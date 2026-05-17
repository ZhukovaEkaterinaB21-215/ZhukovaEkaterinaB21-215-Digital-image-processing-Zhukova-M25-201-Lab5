#include "morph.h"
#include "utils.h"
#include <iostream>
#include "TopBot.h"


cv::Mat morphTopHat(const cv::Mat& src, int winSize) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }

    cv::Mat opened = morphOpening(src, winSize);
    if (opened.empty()) {
        return cv::Mat();
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        const uchar* o = opened.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            d[x] = s[x] - o[x];
        }
    }
    return dst;
}

cv::Mat morphBotHat(const cv::Mat& src, int winSize) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }

    cv::Mat closed = morphClosing(src, winSize);
    if (closed.empty()) {
        return cv::Mat();
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        const uchar* c = closed.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            d[x] = c[x] - s[x];
        }
    }
    return dst;
}
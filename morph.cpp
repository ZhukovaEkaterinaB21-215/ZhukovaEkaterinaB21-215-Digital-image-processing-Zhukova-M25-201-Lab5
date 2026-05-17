#include "morph.h"
#include "utils.h"

static uchar getMinWindow(const cv::Mat& padded, int py, int px, int pad) {
    uchar mn = 255;
    for (int dy = -pad; dy <= pad; ++dy) {
        const uchar* row = padded.ptr<uchar>(py + dy);
        for (int dx = -pad; dx <= pad; ++dx) {
            if (row[px + dx] < mn) {
                mn = row[px + dx];
            }
        }
    }
    return mn;
}

static uchar getMaxWindow(const cv::Mat& padded, int py, int px, int pad) {
    uchar mx = 0;
    for (int dy = -pad; dy <= pad; ++dy) {
        const uchar* row = padded.ptr<uchar>(py + dy);
        for (int dx = -pad; dx <= pad; ++dx) {
            if (row[px + dx] > mx) {
                mx = row[px + dx];
            }
        }
    }
    return mx;
}

cv::Mat morphErosion(const cv::Mat& src, int winSize) {
    if (winSize % 2 == 0) {
        std::cerr << "Размер апертуры должен быть нечётным\n";
        return cv::Mat();
    }
    cv::Mat dst(src.size(), CV_8UC1, cv::Scalar(255));
    int pad = winSize / 2;
    cv::Mat p = padImageReplicate(src, pad);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            dst.at<uchar>(y, x) = getMinWindow(p, y + pad, x + pad, pad);
        }
    }
    return dst;
}

cv::Mat morphDilation(const cv::Mat& src, int winSize) {
    if (winSize % 2 == 0) {
        std::cerr << "Размер апертуры должен быть нечётным\n";
        return cv::Mat();
    }
    cv::Mat dst(src.size(), CV_8UC1, cv::Scalar(0));
    int pad = winSize / 2;
    cv::Mat p = padImageReplicate(src, pad);
    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            dst.at<uchar>(y, x) = getMaxWindow(p, y + pad, x + pad, pad);
        }
    }
    return dst;
}

cv::Mat morphOpening(const cv::Mat& src, int winSize) { 
    return morphDilation(morphErosion(src, winSize), winSize); 
}

cv::Mat morphClosing(const cv::Mat& src, int winSize) { 
    return morphErosion(morphDilation(src, winSize), winSize); 
}
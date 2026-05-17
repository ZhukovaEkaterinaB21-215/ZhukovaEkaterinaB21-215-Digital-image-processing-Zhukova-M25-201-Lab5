#include "trunc.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <cmath>

static void manualSort(std::vector<uchar>& arr) {
    for (size_t i = 1; i < arr.size(); ++i) {
        uchar key = arr[i];
        int j = static_cast<int>(i) - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            --j;
        }
        arr[j + 1] = key;
    }
}

cv::Mat truncatedMeanFilter(const cv::Mat& src, int winSize, int d) {
    if (winSize < 3 || winSize % 2 == 0) {
        std::cerr << "Апертура должна быть нечётной и >= 3\n";
        return cv::Mat();
    }
    if (d < 0 || d >= winSize * winSize) {
        std::cerr << " d должно быть в диапазоне [0, " << (winSize * winSize - 1) << "]\n";
        return cv::Mat();
    }
    if (d % 2 != 0) {
        std::cerr << "d должно быть чётным (для симметричного усечения d/2)\n";
        return cv::Mat();
    }

    cv::Mat dst(src.size(), CV_8UC1);
    int pad = winSize / 2;
    cv::Mat padded = padImageReplicate(src, pad);
    if (padded.empty()) {
        return cv::Mat();
    }

    std::vector<uchar> window(winSize * winSize);
    int KL = winSize * winSize;
    int halfD = d / 2;

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            int idx = 0;
            for (int dy = -pad; dy <= pad; ++dy) {
                const uchar* row = padded.ptr<uchar>(y + pad + dy);
                for (int dx = -pad; dx <= pad; ++dx) {
                    window[idx++] = row[x + pad + dx];
                }
            }

            manualSort(window);

            double sum = 0.0;
            for (int i = halfD; i < KL - halfD; ++i) {
                sum += window[i];
            }

            dst.at<uchar>(y, x) = static_cast<uchar>(std::round(sum / (KL - d)));
        }
    }
    return dst;
}
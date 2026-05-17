#include "rank.h"
#include "utils.h"
#include <vector>
#include <iostream>
#include <algorithm>

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


static void manualNthElement(std::vector<uchar>& arr, int k) {
    if (k < 0 || k >= static_cast<int>(arr.size())) {
        return;
    }

    int left = 0;
    int right = static_cast<int>(arr.size()) - 1;

    while (left < right) {
        int pivotIdx = left + (right - left) / 2;
        uchar pivotVal = arr[pivotIdx];

        uchar tmp = arr[pivotIdx];
        arr[pivotIdx] = arr[right];
        arr[right] = tmp;

        int storeIdx = left;
        for (int i = left; i < right; ++i) {
            if (arr[i] <= pivotVal) {
                tmp = arr[storeIdx];
                arr[storeIdx] = arr[i];
                arr[i] = tmp;
                ++storeIdx;
            }
        }

        tmp = arr[storeIdx];
        arr[storeIdx] = arr[right];
        arr[right] = tmp;

        if (k == storeIdx) {
            return;
        }
        else {
            if (k < storeIdx) {
                right = storeIdx - 1;
            }
            else {
                left = storeIdx + 1;
            }
        }
    }
}


cv::Mat createCrossMask() {
    cv::Mat mask(3, 3, CV_8UC1, uchar(0));
    mask.at<uchar>(0, 1) = 1;
    mask.at<uchar>(1, 0) = 1;
    mask.at<uchar>(1, 1) = 1;
    mask.at<uchar>(1, 2) = 1;
    mask.at<uchar>(2, 1) = 1;
    return mask;
}

cv::Mat createRectMask(int size) {
    if (size < 3 || size % 2 == 0) {
        std::cerr << "Размер маски должен быть нечётным и >= 3\n";
        return cv::Mat();
    }
    return cv::Mat(size, size, CV_8UC1, uchar(1));
}

cv::Mat createDiamondMask() {
    cv::Mat mask(5, 5, CV_8UC1, uchar(0));
    mask.at<uchar>(0, 2) = 1;
    mask.at<uchar>(1, 1) = 1;
    mask.at<uchar>(1, 2) = 1;
    mask.at<uchar>(1, 3) = 1;
    mask.at<uchar>(2, 0) = 1;
    mask.at<uchar>(2, 1) = 1;
    mask.at<uchar>(2, 2) = 1;
    mask.at<uchar>(2, 3) = 1;
    mask.at<uchar>(2, 4) = 1;
    mask.at<uchar>(3, 1) = 1;
    mask.at<uchar>(3, 2) = 1;
    mask.at<uchar>(3, 3) = 1;
    mask.at<uchar>(4, 2) = 1;
    return mask;
}

cv::Mat rankFilter(const cv::Mat& src, int k, const cv::Mat& mask) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }
    if (mask.empty() || mask.channels() != 1 || mask.type() != CV_8UC1) {
        std::cerr << "Маска должна быть CV_8UC1.\n";
        return cv::Mat();
    }

    int activePixels = 0;
    for (int my = 0; my < mask.rows; ++my) {
        for (int mx = 0; mx < mask.cols; ++mx) {
            if (mask.at<uchar>(my, mx) != 0) ++activePixels;
        }
    }

    if (activePixels == 0) {
        std::cerr << "Маска не содержит активных пикселей.\n";
        return cv::Mat();
    }
    if (k < 0 || k >= activePixels) {
        std::cerr << "Ранг k=" << k << " вне диапазона [0, " << (activePixels - 1) << "].\n";
        return cv::Mat();
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) d[x] = s[x];
    }

    int padY = mask.rows / 2;
    int padX = mask.cols / 2;
    int pad = (padY > padX) ? padY : padX;

    cv::Mat padded = padImageReplicate(src, pad);
    std::vector<uchar> window;

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            while (!window.empty()) {
                window.pop_back();
            }

            for (int my = 0; my < mask.rows; ++my) {
                const uchar* maskRow = mask.ptr<uchar>(my);
                
                const uchar* srcRow = padded.ptr<uchar>(y + pad + my - padY);

                for (int mx = 0; mx < mask.cols; ++mx) {
                    if (maskRow[mx] != 0) {
                        
                        window.push_back(srcRow[x + pad + mx - padX]);
                    }
                }
            }

            manualNthElement(window, k);
            dst.at<uchar>(y, x) = window[k];
        }
    }
    return dst;
}

cv::Mat medianFilter(const cv::Mat& src, const cv::Mat& mask) {
    int activePixels = 0;
    for (int my = 0; my < mask.rows; ++my) {
        for (int mx = 0; mx < mask.cols; ++mx) {
            if (mask.at<uchar>(my, mx) != 0) {
                ++activePixels;
            }
        }
    }
    return rankFilter(src, activePixels / 2, mask);
}


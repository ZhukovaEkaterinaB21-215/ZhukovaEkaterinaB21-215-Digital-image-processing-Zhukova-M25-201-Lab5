#include "utils.h"
#include <filesystem>

cv::Mat padImageReplicate(const cv::Mat& src, int pad) {
    if (pad < 0) {
        std::cerr << "pad должен быть >= 0\n";
        return cv::Mat();
    }
    if (src.depth() != CV_8U || src.channels() != 1) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }
    int h = src.rows + 2 * pad;
    int w = src.cols + 2 * pad;
    cv::Mat dst(h, w, CV_8UC1);

    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y + pad) + pad;
        for (int x = 0; x < src.cols; ++x) {
            d[x] = s[x];
        }
    }

    for (int y = 0; y < h; ++y) {
        uchar* row = dst.ptr<uchar>(y);
        int sy = std::clamp(y - pad, 0, src.rows - 1);
        const uchar* srcRow = src.ptr<uchar>(sy);
        for (int x = 0; x < pad; ++x) {
            row[x] = srcRow[0];
        }
        for (int x = 0; x < pad; ++x) {
            row[w - 1 - x] = srcRow[src.cols - 1];
        }
    }
    for (int y = 0; y < pad; ++y) {
        const uchar* refTop = dst.ptr<uchar>(pad);
        const uchar* refBot = dst.ptr<uchar>(dst.rows - pad - 1);
        uchar* rowTop = dst.ptr<uchar>(y);
        uchar* rowBot = dst.ptr<uchar>(dst.rows - 1 - y);
        for (int x = 0; x < w; ++x) { 
            rowTop[x] = refTop[x]; 
            rowBot[x] = refBot[x]; }
    }
    return dst;
}

void ensureOutputDir(const std::string& dir) {
    std::filesystem::create_directories(dir);
}
#include "compare.h"
#include "utils.h"
#include "rank.h"
#include "trunc.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>

const double PI = 3.14159265358979323846;

cv::Mat addSaltPepperNoise(const cv::Mat& src, double density) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC1);

    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            d[x] = s[x];
        }
    }

    srand(42);
    for (int y = 0; y < dst.rows; ++y) {
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < dst.cols; ++x) {
            double r = rand() / (RAND_MAX + 1.0);
            if (r < density) {
                d[x] = (rand() & 1) ? 255 : 0;
            }
        }
    }
    return dst;
}

cv::Mat addGaussianNoise(const cv::Mat& src, double sigma) {
    if (src.empty() || src.channels() != 1 || src.depth() != CV_8U) {
        std::cerr << "Требуется полутоновое изображение 8 бит\n";
        return cv::Mat();
    }

    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    srand(42);


    for (int y = 0; y < src.rows; ++y) {
        const uchar* s = src.ptr<uchar>(y);
        uchar* d = dst.ptr<uchar>(y);
        for (int x = 0; x < src.cols; ++x) {
            double u1 = rand() / (RAND_MAX + 1.0);
            double u2 = rand() / (RAND_MAX + 1.0);
            if (u1 < 1e-10) {
                u1 = 1e-10;
            }

            double z = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2);
            double val = static_cast<double>(s[x]) + sigma * z;

            
            if (val < 0.0) {
                val = 0.0;
            }
            if (val > 255.0) {
                val = 255.0;
            }

            d[x] = static_cast<uchar>(val);
        }
    }
    return dst;
}

cv::Mat averageFilter(const cv::Mat& src, int winSize) {
    if (winSize < 3 || winSize % 2 == 0) {
        std::cerr << "Апертура должна быть нечётной и >= 3\n";
        return cv::Mat();
    }
    cv::Mat dst(src.rows, src.cols, CV_8UC1);
    int pad = winSize / 2;
    cv::Mat padded = padImageReplicate(src, pad);
    if (padded.empty()) {
        return cv::Mat();
    }

    for (int y = 0; y < src.rows; ++y) {
        for (int x = 0; x < src.cols; ++x) {
            double sum = 0.0;
            for (int dy = -pad; dy <= pad; ++dy) {
                const uchar* row = padded.ptr<uchar>(y + pad + dy);
                for (int dx = -pad; dx <= pad; ++dx) {
                    sum += row[x + pad + dx];
                }
            }
            dst.at<uchar>(y, x) = static_cast<uchar>(sum / (winSize * winSize) + 0.5);
        }
    }
    return dst;
}

double calculatePSNR(const cv::Mat& orig, const cv::Mat& proc) {
    if (orig.empty() || proc.empty() || orig.rows != proc.rows || orig.cols != proc.cols || orig.type() != proc.type()) {
        std::cerr << "Некорректные входные данные\n";
        return 0.0;
    }
    double mse = 0.0;
    int total = orig.rows * orig.cols;
    for (int y = 0; y < orig.rows; ++y) {
        const uchar* o = orig.ptr<uchar>(y);
        const uchar* p = proc.ptr<uchar>(y);
        for (int x = 0; x < orig.cols; ++x) {
            double d = o[x] - p[x];
            mse += d * d;
        }
    }
    mse /= total;
    return mse < 1e-9 ? 100.0 : 10.0 * log10((255.0 * 255.0) / mse);
}

void runNoiseComparison(const cv::Mat& original, const std::string& outDir) {
    const int win = 5;
    const int d = 6;
    std::cout << "\nСравнение фильтров\n";

    auto runTest = [&](const cv::Mat& noisy, const std::string& name) {
        std::cout << "Шум: " << name << "\n";
        cv::imwrite(outDir + "noisy_" + name + ".png", noisy);

        cv::Mat resAvg = averageFilter(noisy, win);
        cv::Mat resMed = medianFilter(noisy, createRectMask(win));
        cv::Mat resTrunc = truncatedMeanFilter(noisy, win, d);

        double psnrAvg = calculatePSNR(original, resAvg);
        double psnrMed = calculatePSNR(original, resMed);
        double psnrTrunc = calculatePSNR(original, resTrunc);

        std::cout << std::left << std::setw(15) << "  Фильтр" << std::right << std::setw(10) << "PSNR\n";
        std::cout << std::string(25, '-') << "\n";
        std::cout << std::setw(15) << "  Усредняющий" << std::fixed << std::setw(10) << std::setprecision(2) << psnrAvg << " dB\n";
        std::cout << std::setw(15) << "  Медианный" << std::setw(10) << psnrMed << " dB\n";
        std::cout << std::setw(15) << "  Усеч.среднее" << std::setw(10) << psnrTrunc << " dB\n";

        cv::imwrite(outDir + "res_" + name + "_avg.png", resAvg);
        cv::imwrite(outDir + "res_" + name + "_median.png", resMed);
        cv::imwrite(outDir + "res_" + name + "_trunc.png", resTrunc);
        };

    runTest(addSaltPepperNoise(original, 0.1), "impulse");
    runTest(addGaussianNoise(original, 25.0), "gauss");
}

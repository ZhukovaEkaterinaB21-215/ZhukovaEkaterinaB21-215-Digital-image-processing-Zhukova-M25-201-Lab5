#include "test.h"
#include "utils.h"
#include "rank.h"
#include "trunc.h"
#include "compare.h"
#include "morph.h"
#include "TopBot.h"

#include <iostream>
#include <stdexcept>
#include <cmath>
#include <opencv2/opencv.hpp>


static void saveTestImage(const cv::Mat& img, const std::string& filename) {
    if (!img.empty()) {
        if (cv::imwrite(filename, img)) {
            std::cout << "  [SAVE] " << filename << "\n";
        }
        else {
            std::cerr << "  [WARN] Не удалось сохранить " << filename << "\n";
        }
    }
}


void runTests() {
    const std::string dir = "output_tests/";
    ensureOutputDir(dir);
  
    {
        std::cout << "Проверка ранговой фильтрации...\n";

        cv::Mat test_img(256, 256, CV_8UC1);
        for (int y = 0; y < test_img.rows; ++y)
            for (int x = 0; x < test_img.cols; ++x)
                test_img.at<uchar>(y, x) = static_cast<uchar>(x + y);

        cv::Mat mask = createRectMask(3);

        cv::Mat min_res = rankFilter(test_img, 0, mask);
        cv::Mat med_res = medianFilter(test_img, mask);
        cv::Mat max_res = rankFilter(test_img, 8, mask);

        if (min_res.empty() || med_res.empty() || max_res.empty()) {
            throw std::runtime_error("Ранговая фильтрация вернула пустое изображение");
        }


        bool order_ok = true;
        for (int y = 1; y < 255 && order_ok; ++y) {
            for (int x = 1; x < 255 && order_ok; ++x) {
                uchar mn = min_res.at<uchar>(y, x);
                uchar md = med_res.at<uchar>(y, x);
                uchar mx = max_res.at<uchar>(y, x);
                if (!(mn <= md && md <= mx)) {
                    order_ok = false;
                }
            }
        }
        if (!order_ok) {
            throw std::runtime_error("Нарушено неравенство: min <= median <= max");
        }

        std::cout << "Ранговая фильтрация работает корректно\n";

        saveTestImage(test_img, dir + "test1_rank_input.png");
        saveTestImage(min_res, dir + "test1_rank_min.png");
        saveTestImage(med_res, dir + "test1_rank_median.png");
        saveTestImage(max_res, dir + "test1_rank_max.png");
    }

    {
        std::cout << "Проверка масок апертуры...\n";

        cv::Mat test_img(256, 256, CV_8UC1);
        for (int y = 0; y < test_img.rows; ++y)
            for (int x = 0; x < test_img.cols; ++x)
                test_img.at<uchar>(y, x) = static_cast<uchar>(x + y);


        cv::Mat cross_res = medianFilter(test_img, createCrossMask());
        cv::Mat diamond_res = medianFilter(test_img, createDiamondMask());
        cv::Mat rect5_res = medianFilter(test_img, createRectMask(5));

        if (cross_res.empty() || diamond_res.empty() || rect5_res.empty()) {
            throw std::runtime_error("Фильтрация с нестандартной маской вернула пустой результат");
        }

        std::cout << "Маски крест/ромб/прямоугольник работают корректно.\n";

        saveTestImage(cross_res, dir + "test2_mask_cross.png");
        saveTestImage(diamond_res, dir + "test2_mask_diamond.png");
        saveTestImage(rect5_res, dir + "test2_mask_rect5.png");
    }


    {
        std::cout << "Проверка обработки ошибок (ранговая фильтрация)...\n";

        cv::Mat test_img(256, 256, CV_8UC1);
        for (int y = 0; y < test_img.rows; ++y)
            for (int x = 0; x < test_img.cols; ++x)
                test_img.at<uchar>(y, x) = static_cast<uchar>(x + y);
        cv::Mat mask = createRectMask(3);

        bool err_caught = true;

        if (!rankFilter(test_img, -1, mask).empty() || !rankFilter(test_img, 100, mask).empty() || !rankFilter(cv::Mat(), 0, mask).empty() || !rankFilter(test_img, 0, cv::Mat()).empty()) {
            err_caught = false;
        }


        cv::Mat zero_mask(3, 3, CV_8UC1, uchar(0));
        if (!rankFilter(test_img, 0, zero_mask).empty()) {
            err_caught = false;
        }

        if (!err_caught) {
            throw std::runtime_error("Ранговая фильтрация не отклонила некорректные параметры");
        }

        std::cout << "Обработка ошибок в ранговой фильтрации работает корректно.\n";
    }

    {
        std::cout << "Проверка фильтра усечённого среднего...\n";

        cv::Mat test_img(256, 256, CV_8UC1);
        for (int y = 0; y < test_img.rows; ++y)
            for (int x = 0; x < test_img.cols; ++x)
                test_img.at<uchar>(y, x) = static_cast<uchar>(x + y);

        cv::Mat trunc0 = truncatedMeanFilter(test_img, 3, 0);
        cv::Mat avg = averageFilter(test_img, 3);

        bool d0_ok = true;
        for (int y = 0; y < test_img.rows && d0_ok; ++y) {
            for (int x = 0; x < test_img.cols && d0_ok; ++x) {
                if (std::abs(trunc0.at<uchar>(y, x) - avg.at<uchar>(y, x)) > 1) {
                    d0_ok = false;
                }
            }
        }
        if (!d0_ok) {
            throw std::runtime_error("truncatedMeanFilter(d=0) не совпадает с averageFilter");
        }

        // Проверка ошибок
        bool err_ok = true;
        if (!truncatedMeanFilter(test_img, 3, -1).empty() || !truncatedMeanFilter(test_img, 3, 9).empty() || !truncatedMeanFilter(test_img, 3, 5).empty() || !truncatedMeanFilter(test_img, 4, 2).empty()) {
            err_ok = false;
        }

        if (!err_ok) {
            throw std::runtime_error("truncatedMeanFilter не отклоняет некорректные параметры");
        }

        std::cout << "Фильтр усечённого среднего работает корректно.\n";

        saveTestImage(test_img, dir + "test4_trunc_input.png");
        saveTestImage(trunc0, dir + "test4_trunc_d0.png");
        saveTestImage(truncatedMeanFilter(test_img, 5, 6), dir + "test4_trunc_d6.png");
    }


    {
        std::cout << "Проверка генерации шума...\n";

        cv::Mat test_img(256, 256, CV_8UC1, cv::Scalar(128));

        cv::Mat sp1 = addSaltPepperNoise(test_img, 0.1);
        cv::Mat sp2 = addSaltPepperNoise(test_img, 0.1);

        bool reproducible = true;
        for (int y = 0; y < test_img.rows && reproducible; ++y) {
            for (int x = 0; x < test_img.cols && reproducible; ++x) {
                if (sp1.at<uchar>(y, x) != sp2.at<uchar>(y, x)) {
                    reproducible = false;
                }
            }
        }
        if (!reproducible) {
            throw std::runtime_error("Генерация шума не воспроизводима");
        }


        int changed = 0;
        for (int y = 0; y < test_img.rows; ++y) {
            for (int x = 0; x < test_img.cols; ++x) {
                if (test_img.at<uchar>(y, x) != sp1.at<uchar>(y, x)) {
                    ++changed;
                }
            }
        }

        double ratio = static_cast<double>(changed) / (test_img.rows * test_img.cols);
        if (ratio < 0.05 || ratio > 0.20) {
            throw std::runtime_error("Доля зашумлённых пикселей не соответствует вероятности");
        }

        std::cout << "Генерация шума работает корректно.\n";

        saveTestImage(test_img, dir + "test5_noise_clean.png");
        saveTestImage(sp1, dir + "test5_noise_sp.png");
        saveTestImage(addGaussianNoise(test_img, 25.0), dir + "test5_noise_gauss.png");
    }

    {
        std::cout << "Сравнение фильтров: импульсный шум...\n";

        cv::Mat clean(256, 256, CV_8UC1, cv::Scalar(128));
        cv::Mat noisy = addSaltPepperNoise(clean, 0.1);

        cv::Mat res_avg = averageFilter(noisy, 5);
        cv::Mat res_med = medianFilter(noisy, createRectMask(5));
        cv::Mat res_trunc = truncatedMeanFilter(noisy, 5, 6);

        double psnr_avg = calculatePSNR(clean, res_avg);
        double psnr_med = calculatePSNR(clean, res_med);
        double psnr_trunc = calculatePSNR(clean, res_trunc);

        std::cout << "  PSNR Average: " << psnr_avg << " dB\n";
        std::cout << "  PSNR Median:  " << psnr_med << " dB\n";
        std::cout << "  PSNR Trunc:   " << psnr_trunc << " dB\n";

        if (psnr_med <= psnr_avg) {
            throw std::runtime_error("Медианный фильтр не превзошёл усредняющий для импульсного шума");
        }

        std::cout << "Медианный фильтр эффективнее усредняющего для импульсного шума.\n";

        saveTestImage(clean, dir + "test6_sp_clean.png");
        saveTestImage(noisy, dir + "test6_sp_noisy.png");
        saveTestImage(res_avg, dir + "test6_sp_avg.png");
        saveTestImage(res_med, dir + "test6_sp_median.png");
        saveTestImage(res_trunc, dir + "test6_sp_trunc.png");
    }

    {
        std::cout << "Сравнение фильтров: гауссов шум...\n";

        cv::Mat clean(256, 256, CV_8UC1, cv::Scalar(128));
        cv::Mat noisy = addGaussianNoise(clean, 25.0);

        cv::Mat res_avg = averageFilter(noisy, 5);
        cv::Mat res_med = medianFilter(noisy, createRectMask(5));
        cv::Mat res_trunc = truncatedMeanFilter(noisy, 5, 6);

        double psnr_avg = calculatePSNR(clean, res_avg);
        double psnr_med = calculatePSNR(clean, res_med);
        double psnr_trunc = calculatePSNR(clean, res_trunc);

        std::cout << "  PSNR Average: " << psnr_avg << " dB\n";
        std::cout << "  PSNR Median:  " << psnr_med << " dB\n";
        std::cout << "  PSNR Trunc:   " << psnr_trunc << " dB\n";

        if (psnr_avg < 10.0 || psnr_trunc < 10.0) {
            throw std::runtime_error("Фильтры не улучшили изображение с гауссовым шумом");
        }

        std::cout << "Фильтры эффективно обрабатывают гауссов шум.\n";

        saveTestImage(clean, dir + "test7_gauss_clean.png");
        saveTestImage(noisy, dir + "test7_gauss_noisy.png");
        saveTestImage(res_avg, dir + "test7_gauss_avg.png");
        saveTestImage(res_med, dir + "test7_gauss_median.png");
        saveTestImage(res_trunc, dir + "test7_gauss_trunc.png");
    }

    
    {
        std::cout << "Проверка морфологических операций...\n";

        cv::Mat test_img(256, 256, CV_8UC1, uchar(0));
        for (int y = 80; y < 176; ++y) {
            for (int x = 80; x < 176; ++x) {
                test_img.at<uchar>(y, x) = 200;
            }
        }

        cv::Mat eroded = morphErosion(test_img, 3);
        cv::Mat dilated = morphDilation(test_img, 3);
        cv::Mat opened = morphOpening(test_img, 3);
        cv::Mat closed = morphClosing(test_img, 3);

        if (eroded.empty() || dilated.empty() || opened.empty() || closed.empty()) {
            throw std::runtime_error("Морфологическая операция вернула пустой результат");
        }

        int bright_orig = 0, bright_er = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (test_img.at<uchar>(y, x) > 100) {
                    ++bright_orig;
                }
                if (eroded.at<uchar>(y, x) > 100) {
                    ++bright_er;
                }
            }
        }
        if (bright_er >= bright_orig) {
            throw std::runtime_error("Эрозия не уменьшила яркий объект");
        }

        int bright_di = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (dilated.at<uchar>(y, x) > 100) ++bright_di;
            }
        }
        if (bright_di <= bright_orig) {
            throw std::runtime_error("Дилатация не увеличила яркий объект");
        }

        bool borders_ok = true;
        for (int y = 0; y < 256 && borders_ok; ++y) {
            for (int x = 0; x < 256 && borders_ok; ++x) {
                if (eroded.at<uchar>(y, x) > 100 && test_img.at<uchar>(y, x) <= 100) {
                    borders_ok = false; 
                }
            }
        }
        if (!borders_ok) {
            throw std::runtime_error("Эрозия нарушила границы объекта");
        }

        std::cout << "Морфологические операции работают корректно.\n";

        saveTestImage(test_img, dir + "test8_morph_input.png");
        saveTestImage(eroded, dir + "test8_morph_erosion.png");
        saveTestImage(dilated, dir + "test8_morph_dilation.png");
        saveTestImage(opened, dir + "test8_morph_opening.png");
        saveTestImage(closed, dir + "test8_morph_closing.png");
    }

    {
        std::cout << "Проверка операций TopHat и BotHat...\n";

        cv::Mat test_img(256, 256, CV_8UC1, cv::Scalar(100));

        for (int i = 0; i < 10; ++i) {
            test_img.at<uchar>(40 + i, 40 + i) = 255;
            test_img.at<uchar>(200 - i, 200 - i) = 0;
        }

        cv::Mat tophat = morphTopHat(test_img, 5);
        cv::Mat bothat = morphBotHat(test_img, 5);

        if (tophat.empty() || bothat.empty()) {
            throw std::runtime_error("TopHat/BotHat вернули пустой результат");
        }

        int bright_tophat = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (tophat.at<uchar>(y, x) > 50) {
                    ++bright_tophat;
                }
            }
        }

        if (bright_tophat < 5) {
            throw std::runtime_error("TopHat не выделил яркие детали");
        }

        
        int dark_bothat = 0;
        for (int y = 0; y < 256; ++y) {
            for (int x = 0; x < 256; ++x) {
                if (bothat.at<uchar>(y, x) > 50) {
                    ++dark_bothat;
                }
            }
        }

        if (dark_bothat < 5) {
            throw std::runtime_error("BotHat не выделил тёмные детали");
        }

        std::cout << "TopHat и BotHat работают корректно.\n";

        saveTestImage(test_img, dir + "test10_tophat_input.png");
        saveTestImage(tophat, dir + "test10_tophat_result.png");
        saveTestImage(bothat, dir + "test10_bothat_result.png");
    }

}
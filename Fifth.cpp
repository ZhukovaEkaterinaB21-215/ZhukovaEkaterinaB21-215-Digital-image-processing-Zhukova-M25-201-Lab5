#include <iostream>
#include <string>
#include <filesystem>
#include <opencv2/opencv.hpp>
#include "utils.h"
#include "rank.h"
#include "trunc.h"
#include "compare.h"
#include "morph.h"
#include "test.h"
#include "TopBot.h"

int main(int argc, char** argv) {
    #ifdef _WIN32
        system("chcp 65001 >nul");
    #endif

    runTests();

     
    const std::string dir = "output_results/";
    ensureOutputDir(dir);

    cv::Mat original;
    std::string img_source;

    img_source = argv[1];
    original = cv::imread(img_source, cv::IMREAD_GRAYSCALE);

    cv::imwrite(dir + "original.png", original);

    const int WIN = 5; 
    
    cv::Mat crossMask = createCrossMask();
    cv::imwrite(dir + "rank_cross_median.png", medianFilter(original, crossMask));

    cv::Mat rect3_3 = createRectMask(3);
    cv::imwrite(dir + "rank_3_3_median.png", medianFilter(original, rect3_3));

    cv::Mat diamond = createDiamondMask();
    cv::imwrite(dir + "rank_diamond_median.png", medianFilter(original, diamond));

    cv::Mat rect5_5 = createRectMask(5);
    cv::imwrite(dir + "rank_5_5_min.png", rankFilter(original, 0, rect5_5));
    cv::imwrite(dir + "rank_5_5_q25.png", rankFilter(original, 6, rect5_5));
    cv::imwrite(dir + "rank_5_5_median.png", rankFilter(original, 12, rect5_5));
    cv::imwrite(dir + "rank_5_5_max.png", rankFilter(original, 24, rect5_5));    
    

    cv::imwrite(dir + "trunc_d0.png", truncatedMeanFilter(original, WIN, 0));
    cv::imwrite(dir + "trunc_d6.png", truncatedMeanFilter(original, WIN, 6));
    cv::imwrite(dir + "trunc_d12.png", truncatedMeanFilter(original, WIN, 12));
    cv::imwrite(dir + "trunc_d24.png", truncatedMeanFilter(original, WIN, 24));

    runNoiseComparison(original, dir);

    cv::imwrite(dir + "erosion.png", morphErosion(original, WIN));
    cv::imwrite(dir + "dilation.png", morphDilation(original, WIN));
    cv::imwrite(dir + "opening.png", morphOpening(original, WIN));
    cv::imwrite(dir + "closing.png", morphClosing(original, WIN));


    cv::Mat tophat = morphTopHat(original, WIN);
    cv::Mat bothat = morphBotHat(original, WIN);

    cv::imwrite(dir + "morph_tophat.png", tophat);
    cv::imwrite(dir + "morph_bothat.png", bothat);

    cv::waitKey(0);
    cv::destroyAllWindows();
    
    return 0;
}
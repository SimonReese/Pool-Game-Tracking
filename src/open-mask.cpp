/**
 * @author Simone Peraro.
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

/**
 * Open a mask remapping its values to better show segmented pixels
 */
void openMask(std::string maskPath, std::string winname);

int main(int argc, char* argv[]){
    if (argc < 3){
        std::cerr << "specify true mask path and predicted mask path" << std::endl;
    }

    std::string maskPath = argv[1];
    std::string maskPredicted = argv[2];
    openMask(maskPath, "True");
    openMask(maskPredicted, "Predicted");
    cv::waitKey(0);
}

void openMask(std::string maskPath, std::string winname){
    cv::Mat mask = cv::imread(maskPath);
    std::vector<uchar> conversionTable(256);
    conversionTable[0] = 0;
    conversionTable[1] = 100;
    conversionTable[2] = 128;
    conversionTable[3] = 150;
    conversionTable[4] = 200;
    conversionTable[5] = 255;
    cv::Mat remapped;
    cv::LUT(mask,conversionTable,remapped);
    cv::imshow(winname, remapped);
}

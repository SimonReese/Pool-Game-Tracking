#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <BallsDetection.h>
#include <FieldGeometryAndMask.h>
#include <Ball.h>
#include <filesystem>

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

int value = 0;


std::string OUTPUT_DATASET = "../res/predictions";
std::string OUTPUT_CLIP = "/game1_clip1";


const std::string FRAMES_DIR = "frames";
const std::string BOX_DIR = "bounding_boxes";
const std::string MASK_DIR = "masks";
const std::string CUTOUT_DIR = "cutouts";


enum class BallClass {
    FULL,
    HALF
};

void showHsvChannelsandBinary(const std::vector<cv::Mat> &channelsHsvImage, const cv::Mat &binaryImage, std::string windowName){

    int k = 13;
    cv::Mat H = channelsHsvImage[0].clone();
    cv::Mat S = channelsHsvImage[1].clone();
    cv::Mat V = channelsHsvImage[2].clone();

    cv::Mat binaryClone = binaryImage.clone();


    // cv::resize(H, H, cv::Size(H.rows*k, H.cols*k));
    // cv::resize(S, S, cv::Size(S.rows*k, S.cols*k));
    cv::resize(V, V, cv::Size(V.rows*k, V.cols*k));
    cv::resize(binaryClone, binaryClone, cv::Size(binaryClone.rows*k, binaryClone.cols*k));

    // cv::imshow("H", H);
    // cv::imshow("S", S);
    cv::imshow(windowName, V);
    cv::imshow(windowName + "_bin", binaryClone);

}

void showHlsChannelsandBinary(const std::vector<cv::Mat> &channelsHlsImage, const cv::Mat &binaryImage, std::string windowName){

    int k = 13;
    cv::Mat H = channelsHlsImage[0].clone();
    cv::Mat L = channelsHlsImage[1].clone();
    cv::Mat S = channelsHlsImage[2].clone();

    cv::Mat binaryClone = binaryImage.clone();


    // cv::resize(H, H, cv::Size(H.rows*k, H.cols*k));
    cv::resize(S, S, cv::Size(S.rows*k, S.cols*k));
    cv::resize(L, L, cv::Size(L.rows*k, L.cols*k));
    cv::resize(binaryClone, binaryClone, cv::Size(binaryClone.rows*k, binaryClone.cols*k));

    //cv::imshow("H", H);
    cv::imshow(windowName + "S", S);
    cv::imshow(windowName + "L", L);
    cv::imshow(windowName + "_bin", binaryClone);

}

float whitePixelsRatio(const cv::Mat &image){
    int whitePixels = cv::countNonZero(image == 255);
    int totalPixels = image.rows * image.cols;
    return static_cast<float>(whitePixels) / totalPixels;
}


BallClass ballClassifier(cv::Mat &image, std::string name, BallClass expected){
    cv::Mat hsvImage;
    // cv::cvtColor(image,hsvImage,cv::COLOR_BGR2HLS);
    //cv::Mat normalizedImage;
    // cv::normalize(image, image, 0, 255, cv::NORM_MINMAX);
    cv::Mat smoothedImage;
    int kernelSize = 11;
    double sigmaX = 3; // Standard deviation in X direction. If it is zero, it is computed from ksize.
    // cv::GaussianBlur(image, smoothedImage, cv::Size(kernelSize, kernelSize), sigmaX);
    cv::cvtColor(image, hsvImage, cv::COLOR_BGR2HLS);


    std::vector<cv::Mat> channelsHsvImage;
    cv::split(hsvImage,channelsHsvImage);

    // cv::Mat normalizedImage(channelsHsvImage[1].size(), CV_16FC1);

    cv::Mat binaryImage;
    double thresholdValue = 150;  // Threshold value
    double maxBinaryValue = 255; // Maximum value to use with the THRESH_BINARY type
    // cv::normalize(channelsHsvImage[1], normalizedImage, 0, 255, cv::NORM_L1);

    // cv::equalizeHist(channelsHsvImage[1], channelsHsvImage[1]);

    int low_H = 0;
    int high_H = 180;

    int low_L = 150;
    int high_L = 255;
    
    int low_S = 30;
    int high_S = 255;

    cv::inRange(hsvImage, cv::Scalar(low_H, low_L, low_S), cv::Scalar(high_H, high_L, high_S), binaryImage);
    // cv::threshold(channelsHsvImage[1], binaryImage, thresholdValue, maxBinaryValue, cv::THRESH_BINARY );

    float whiteRatio = whitePixelsRatio(binaryImage);

    std::cout << name + "White pixels ratio: " << whiteRatio << std::endl;

    float classificationThresh = 0.099;

    if (whiteRatio > classificationThresh) {
        if(expected != BallClass::HALF){
            showHlsChannelsandBinary(channelsHsvImage, binaryImage, name);
            // cv::imshow(name + "_normed", image);
        } 
        return BallClass::HALF;
    } else {
        if(expected != BallClass::FULL){
            showHlsChannelsandBinary(channelsHsvImage, binaryImage, name);
            // cv::imshow(name + "_normed", image);
        }
        return BallClass::FULL;
    }
}

void saveTofile(const cv::Mat &image, std::string imageName, std::string outputFolder){

    // Define the root folder
    std::string rootFolderPath = "../" + CUTOUT_DIR;

    // Check if the folder exists, if not, create its
    if (!cv::utils::fs::exists(rootFolderPath)) {
        if (cv::utils::fs::createDirectory(rootFolderPath)) {
            std::cout << "Directory created successfully: " << rootFolderPath << std::endl;
        } else {
            std::cout << "Failed to create directory: " << rootFolderPath << std::endl;
            // return -1;
        }
    }

    // Check if the folder exists, if not, create its
    if (!cv::utils::fs::exists(outputFolder)) {
        if (cv::utils::fs::createDirectory(outputFolder)) {
            std::cout << "Directory created successfully: " << outputFolder << std::endl;
        } else {
            std::cout << "Failed to create directory: " << outputFolder << std::endl;
            // return -1;
        }
    }

    // Define the output file path
    std::string outputPath = outputFolder + imageName + ".png";

    // Save the image
    if (cv::imwrite(outputPath, image)) {
        std::cout << "Image saved successfully to " << outputPath << std::endl;
    } else {
        std::cout << "Failed to save the image" << std::endl;
    }
}


void cutOutBalls(const cv::Mat &image, const vector<Ball>& balls, const std::string &gameFolder){


    int i=0;
    for (Ball ball : balls) {

        cv::Mat boundingBoxCutOut = image(ball.getBoundingBox());

        // maybe for balls better to have a tuple for center and a variable just for the radius
        cv::Mat circleMask = cv::Mat::zeros(boundingBoxCutOut.size(), CV_8UC1);

        cv::Point2i center(circleMask.cols /2, circleMask.rows /2);
        circle(circleMask, center , 9, cv::Scalar(255), -1);
        // Copy the original ROI to the ball cutout, but only where the mask is white (the ball is present)
        cv::Mat circleCutOut;

        // cv::bitwise_and(boundingBoxCutOut, boundingBoxCutOut, circleCutOut, circleMask);
        boundingBoxCutOut.copyTo(circleCutOut, circleMask); 

        // std::cout << "Ball center: " << ball.getBallCenterInBoundingBox() << boundingBoxCutOut.rows << std::endl;

        saveTofile(circleCutOut, "ball_cutout" + std::to_string(i), "../" + CUTOUT_DIR + "/" + gameFolder + "/" ); 
        // cv:imshow("Ball Detection" + std::to_string(i), circleMask);
        i++;
    }
    

}


void my_HSV_callback2(int event, int x, int y, int flags, void* userdata, std::string bboxFileName, std::string maskFileName, std::string gameFolder){
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat image2 = *(cv::Mat*) userdata;
        cv::Mat no_mod_image = image2.clone();
        cv::Mat image; 

        cv::cvtColor(image2,image,cv::COLOR_BGR2HSV);

        cv::Mat less_blur_image = image.clone();

        cv::Mat no_blur_image = image.clone();

        cv::GaussianBlur(less_blur_image,less_blur_image,cv::Size(3,3),0,0); // used to find balls later on

        cv::GaussianBlur(image,image,cv::Size(7,7),0,0); // used to find field mask
        
        cv::Vec3b mean_color = fieldMeanColor(image,11);

        cv::Mat filled_field_contour = computeFieldMask(image,mean_color);
        
        cv::Mat approximate_field_lines = findFieldLines(filled_field_contour);

        vector<cv::Point2i> sorted_corners = findFieldCorners(approximate_field_lines);

        vector<cv::Point> boundaries_contours_poly = defineBoundingPolygon(sorted_corners,approximate_field_lines);

        cv::Mat only_table_image = no_blur_image.clone();

        //removes everything from the initial image apart from the pixels defined by the mask that segments the field
        for (int i = 0; i < image.size().height; i++){
            for (int j = 0; j < image.size().width; j++){
                if( filled_field_contour.at<uchar>(i,j) == 0){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                }
            }
        }

        // cv::imshow("filled_field_contour", filled_field_contour);

        vector<Ball> balls = findBalls(only_table_image, filled_field_contour, boundaries_contours_poly, sorted_corners);

        drawBallsHSVChannels(balls, no_mod_image);

        cv::Mat field_and_balls_mask = drawBallsOnFieldMask(filled_field_contour,balls);

        vector<cv::Rect> boundRect = findBoundingRectangles(field_and_balls_mask);

        /*updates the bounding box value of each ball by assigning the bounding box that has center closest to the center of the circle that defines the ball*/
        int offset_thres = 1;
        for (int h = 0; h < boundRect.size(); h++){
            for (int b = 0; b < balls.size(); b++){
                if((boundRect[h].x+boundRect[h].width/2) >= balls[b].getBallPosition()[0]-offset_thres && (boundRect[h].x+boundRect[h].width/2) <= balls[b].getBallPosition()[0]+offset_thres){
                    if((boundRect[h].y+boundRect[h].height/2) >= balls[b].getBallPosition()[1]-offset_thres && (boundRect[h].y+boundRect[h].height/2) <= balls[b].getBallPosition()[1]+offset_thres){
                        balls[b].setBoundingBox(boundRect[h]);
                    }
                }
            }   
        }

        // -> here try the algorgthmt to classify balls

        // find bounding boxes of all circles and draws them
        
        drawBoundingBoxesHSVChannels(balls, no_mod_image);
        cutOutBalls(no_mod_image, balls, gameFolder);



        // writeBboxToFile(bboxFileName, balls);

        // cv::namedWindow("mask display");
        // cv::imshow("mask display", mask);
        // cv::namedWindow("edges display");
        // cv::imshow("edges display", edges);
        // cv::namedWindow("field display");
        // cv::imshow("field display", field_contour);
        // cv::namedWindow("cropped field display");
        // cv::imshow("cropped field display", exit1);
        // cv::namedWindow("cropped field display");
        // cv::cvtColor(only_table_image,only_table_image,cv::COLOR_HSV2BGR);
        // cv::imshow("cropped field display", only_table_image);

        // cv::cvtColor(image,image,cv::COLOR_HSV2BGR);
        

        // cv::imshow("boh",exit1);

        // cv::imwrite(maskFileName,field_and_balls_mask); //salva mask del campo e palline --> background = 0 campo = 255 palline = 127


    }
}

void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
    my_HSV_callback2(event, x, y, flags, userdata, "bounding_box_output.txt", "maschera.jpeg", "okok");
}

std::vector<cv::String> listGameDirectories(std::string datasetPath){
    std::vector<cv::String> files;
    std::vector<cv::String> gameFolders;
    cv::utils::fs::glob_relative(datasetPath, "", files, false, true);
    for (cv::String file : files){
        // Check if it is a directory
        if (cv::utils::fs::isDirectory(datasetPath + "/" + file)){
            gameFolders.push_back(file);
        }
    }

    return gameFolders;
}

std::vector<cv::String> listFrames(std::string datasetPath, std::string gamePath, std::string frameFolderName){
    std::vector<cv::String> frameFullNames;
    std::string fullPath = datasetPath + "/" + gamePath + "/" + frameFolderName;
    cv::utils::fs::glob_relative(fullPath, "", frameFullNames);
    return frameFullNames;
}

int evaluateBallsSet(const std::string datasetFolder, const std::string gameFolder, const std::string ballClassFolder){

    std::vector<cv::String> cutOutList = listFrames("../" + datasetFolder, gameFolder, ballClassFolder);

    BallClass setClass;
    int wrongClassCounter = 0;

    (ballClassFolder == "full") ? setClass = BallClass::FULL
    : (ballClassFolder == "half") ? setClass = BallClass::HALF
    : throw std::invalid_argument("Invalid ball class folder: " + ballClassFolder);


    for (cv::String cutOutName : cutOutList){

        std::string imgPath = "../" + datasetFolder + "/" + gameFolder + "/" + ballClassFolder + "/" + cutOutName;
        cv::Mat cutout = cv::imread( imgPath.c_str() );
        BallClass assignedClass = ballClassifier(cutout, cutOutName, setClass);

        (assignedClass != setClass) ? wrongClassCounter++ : 0;

    }


    return wrongClassCounter;

}

void evaluteGames(std::string datasetFolder){

    std::string datasetPath = "../" + datasetFolder;

    std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);

    int wrongClassifiedFull;
    int wrongClassifiedHalf;

    int wrongClassifiedSum = 0;

    for(cv::String game : gameFolders){

        std::string ballClassFolder = "full";
        wrongClassifiedFull = evaluateBallsSet(datasetFolder, game, ballClassFolder);   

        std::cout << "================" << std::endl;

        ballClassFolder = "half";
        wrongClassifiedHalf = evaluateBallsSet(datasetFolder, game, ballClassFolder); 

        std::cout << "Wrongly classified full balls in " << game << ": " << wrongClassifiedFull << std::endl;
        std::cout << "Wrongly classified half balls in " << game << ": " << wrongClassifiedHalf << "\n"<< std::endl;
        //std::cout << "================" << std::endl;
        // std::cout << "Total wrong classified balls in " << gameFolder << ": " << wrongClassifiedFull + wrongClassifiedHalf << std::endl;

        wrongClassifiedSum += wrongClassifiedFull + wrongClassifiedHalf;
    }


    std::cout << "Total wrong classified balls in all games: " << wrongClassifiedSum << std::endl;
    
}


int main(int argc, char* argv[])
{
    if (argc < 3){
        std::cout << "Please enter dataset path (usually in (root)/res/Dataset/) and predictions path (usually in (root)/res/predictions/)" << std::endl;
    }
    std::string datasetPath = argv[1];
    // std::string predictionsPath = argv[2];
    std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);


//the following lines are to extract singular bboxes from the dataset to cutout the balls
    //given dummy value just to use function my_HSV_callback2()
    std::string bboxFileName = "none"; // NOT USED
    std::string maskFileName = "none"; // NOT USED

    //using gamefolders computed before with listGameDirectories()

    //ANNOTATION: function my_HSV_callback2() has imwrite error in game2_clip2 and not running in game3_clip2
    /* for(cv::String game : gameFolders)
    for(int i = 6; i < gameFolders.size(); i++){

        std::string game = gameFolders[i];

        //game = "game4_clip2";

        // Open frame
        std::string fullFramePath = datasetPath + "/" + game + "/" + FRAMES_DIR + "/" + "frame_first.png";
        cv::Mat image = cv::imread(fullFramePath);

        my_HSV_callback2(cv::EVENT_LBUTTONDOWN, 0, 0, 0, &image, bboxFileName, maskFileName, game);
        
        //break; //used to test the function with a single frame
    } */

    std::string datasetFolder = "balls_cutout";
    

    evaluteGames(datasetFolder);

    cv::waitKey(0);

    return 0;
}

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <BallsDetection.h>
#include <FieldGeometryAndMask.h>
#include <Ball.h>

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

std::string OUTPUT_DATASET = "../res/predictions";
std::string OUTPUT_CLIP = "/game1_clip1";




void my_HSV_callback2(int event, int x, int y, int flags, void* userdata, std::string bboxFileName, std::string maskFileName){
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat image = *(cv::Mat*) userdata;

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

        vector<Ball> balls = findBalls(only_table_image, filled_field_contour, boundaries_contours_poly, sorted_corners);

        drawBallsHSVChannels(balls, image);

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
        // find bounding boxes of all circles and draws them
        
        drawBoundingBoxesHSVChannels(balls, image);

        writeBboxToFile(bboxFileName, balls);

        // cv::namedWindow("mask display");
        // cv::imshow("mask display", mask);
        // cv::namedWindow("edges display");
        // cv::imshow("edges display", edges);
        // cv::namedWindow("field display");
        // cv::imshow("field display", field_contour);
        // cv::namedWindow("cropped field display");
        // cv::imshow("cropped field display", exit1);
        // cv::namedWindow("cropped field display");
        cv::cvtColor(only_table_image,only_table_image,cv::COLOR_HSV2BGR);
        cv::imshow("cropped field display", only_table_image);

        cv::cvtColor(image,image,cv::COLOR_HSV2BGR);
        cv::imshow("angoli",image);

        // cv::imshow("boh",exit1);

        cv::imwrite(maskFileName,field_and_balls_mask); //salva mask del campo e palline --> background = 0 campo = 255 palline = 127


    }
}

void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
    my_HSV_callback2(event, x, y, flags, userdata, "bounding_box_output.txt", "maschera.jpeg");
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

const std::string FRAMES_DIR = "frames";
const std::string BOX_DIR = "bounding_boxes";
const std::string MASK_DIR = "masks";

int main(int argc, char* argv[])
{
    if (argc < 3){
        std::cout << "Please enter dataset path (usually in (root)/res/Dataset/) and predictions path (usually in (root)/res/predictions/)" << std::endl;
    }
    std::string datasetPath = argv[1];
    std::string predictionsPath = argv[2];
    std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);
    for(cv::String game : gameFolders){
        // Create folders for predictions
        cv::utils::fs::createDirectories(predictionsPath + "/" + game + "/" + BOX_DIR);
        cv::utils::fs::createDirectories(predictionsPath + "/" + game + "/" + MASK_DIR);
        // Get all frames in game folder
        std::vector<cv::String> frames = listFrames(datasetPath, game, FRAMES_DIR);
        // For each frame
        for (cv::String frame : frames){
            // Open frame
            std::string fullFramePath = datasetPath + "/" + game + "/" + "/" + FRAMES_DIR + "/" + frame;
            cv::Mat image = cv::imread(fullFramePath);
            cv::Mat imageHSV;
            cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);
            // Compute boundig box and mask file name
            std::string bboxFileName = frame;
            int index = bboxFileName.rfind(".");
            bboxFileName = bboxFileName.erase(index, std::string::npos);
            std::string maskFileName = bboxFileName;
            bboxFileName = predictionsPath + "/" + game + "/" + BOX_DIR + "/" + bboxFileName + "_bbox" +".txt";
            maskFileName = predictionsPath + "/" + game + "/" + MASK_DIR + "/" + maskFileName + ".png";
            my_HSV_callback2(cv::EVENT_LBUTTONDOWN, 0, 0, 0, &imageHSV, bboxFileName, maskFileName);
        }
    }
    std::string clip_path = "/game1_clip1";

    cv::Mat image = cv::imread( datasetPath + clip_path + "/frames/frame_first.png");
    cv::Mat image2;
    cv::Mat image3 = cv::imread( datasetPath + clip_path + "/frames/frame_first.png");
    cv::Mat image4;
    cv::cvtColor(image,image2,cv::COLOR_BGR2HSV);
    cv::cvtColor(image3,image4,cv::COLOR_BGR2HSV);
    cv::namedWindow("hsv window");
    cv::imshow("hsv window",image);
    cv::setMouseCallback("hsv window", my_HSV_callback2, &image2);
    //cv::namedWindow("hsv window2");
    //cv::imshow("hsv window2",image3);
    //cv::setMouseCallback("hsv window2", my_HSV_callback2, &image4);
    cv::waitKey(0);

    return 0;
}

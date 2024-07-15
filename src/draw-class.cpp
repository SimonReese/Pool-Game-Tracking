/**
 * @author Simone Peraro.
 */

#include <iostream>
#include <tuple>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Draw.h"
#include "TableSegmenter.h"
#include "BallClassifier.h"
#include "Ball.h"

int main(int argc, char* argv[]){
    
    if (argc < 2){
        std::cerr << "Please provide a video path." << std::endl;
        return -1;
    }

    std::string inputFile = argv[1];

    cv::VideoCapture video(inputFile);
    if(!video.isOpened()){
        std::cerr << "Error. Unable to open video " << inputFile << std::endl;
        return -1;
    }

    // Sart reading video
    cv::Mat frame;
    TableSegmenter segmenter;
    Draw draw;
    for( video >> frame; !frame.empty(); video >> frame){
        cv::imshow("Video", frame);

        // 1. Get table mask
        cv::Mat mask = segmenter.getTableMask(frame);
        // Show masked frame
        cv::Mat maskedFrame = segmenter.getMaskedImage(frame, mask);
        cv::imshow("Masked frame", maskedFrame);
        // 2. Get table corners
        std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);        

        // 3. Detect balls
        std::vector<Ball> balls;
        std::vector<std::tuple<cv::Point2f, cv::Point2f>> points;
        for(auto corner: corners){
            Ball ball(1, cv::Point(500, 250));
            balls.push_back(ball);
            std::cout << corner << std::endl;
            points.push_back(std::make_tuple(corner, cv::Point(500, 250)));
        }

        // DEBUG
        draw.computePrespective(corners);
        cv::Mat drawing = draw.updateDrawing(balls, points);
        cv::imshow("Draw", drawing);
        std::cout << "Endframe" << std::endl;
        //4. Associate class to balls
        //BallClassifier::classify(balls, frame);

        cv::waitKey(25);
    }
    video.release(); 
    cv::waitKey(0);
    


    return 0;
    /*
    // Read a single frame image
    cv::Mat frame = cv::imread(inputFile);
    // Construct Draw object
    Draw drawing("");
    // Set frame
    drawing.setCurrentFrame(frame);
    // Get drawing
    cv::Mat result;
    drawing.getGameDraw(result);
    cv::imshow("Correction", result);
    cv::imshow("Original", frame);
    cv::waitKey(0);
    return 0;
    // Testing overlapping function
    cv::Mat spiderMan = cv::imread("../res/spider-man-3.png");
    cv::Mat testImage = cv::imread("../res/opencv-test-image.png");
    cv::resize(spiderMan, spiderMan, cv::Size(300, 300));
    cv::Mat over = drawing.drawOver(testImage, spiderMan, cv::Point(testImage.cols / 2, testImage.rows / 2));
    cv::imshow("Overlap", over);
    //animation();
    cv::waitKey(0);
    return 0; 
    */
}
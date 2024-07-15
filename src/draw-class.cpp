/**
 * @author Simone Peraro.
 */

#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Draw.h"
#include "TableSegmenter.h"
#include "BallDetector.h"
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
    BallDetector ballDetector;
    for( video >> frame; !frame.empty(); video >> frame){
        cv::imshow("Video", frame);

        // 1. Get table mask
        cv::Mat mask = segmenter.getTableMask(frame);
        // Show masked frame
        cv::Mat maskedFrame = segmenter.getMaskedImage(frame, mask);
        

        // 2. Get table corners
        std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);      

        // 3. Detect balls
        std::vector<Ball> balls = ballDetector.detectBalls(frame, mask, segmenter.getTableContours(), corners); 
        std::cout << "Detected balls: " << balls.size() << std::endl;    


        // Draw balls over image
        for (Ball ball : balls){
            cv::Vec3f pos = ball.getBallPosition();
            cv::Vec2i center(pos[0], pos[1]);
            cv::circle(maskedFrame, center, ball.getBallRadius(), cv::Scalar(0, 255, 128));
        }
        cv::imshow("Masked frame", maskedFrame);
        cv::waitKey(25);
    }
    video.release(); \
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
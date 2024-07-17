/**
 * @author Simone Peraro.
 */

#include <iostream>
#include <tuple>

//just to see how fast the tracker runs, can then be removed
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "Draw.h"
#include "TableSegmenter.h"
#include "BallDetector.h"
#include "BallClassifier.h"
#include "Ball.h"
#include "BallTracker.h"

int main(int argc, char* argv[]){

    //just for chrono can then be removed
    using namespace std::chrono;
    
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
    cv::Mat firstFrame;
    video >> firstFrame;

    TableSegmenter segmenter;
    Draw draw;
    BallDetector ballDetector;

    // 1. Get table mask
    cv::Mat mask = segmenter.getTableMask(firstFrame);
    // Show masked frame
    cv::Mat maskedFrame = segmenter.getMaskedImage(firstFrame, mask);

    // 2. Get table corners
    std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);

    // 3. Detect balls
    std::vector<Ball> balls = ballDetector.detectBalls(firstFrame, mask, segmenter.getTableContours(), corners);

    // DEBUG
    draw.computePrespective(corners);

    
    BallClassifier ballClassifier;
    balls = ballClassifier.classify(balls, firstFrame);

    BallTracker tracker(firstFrame, balls);
    
    // for(Ball ball : balls){

    //     switch (ball.getBallType())
    //         {
    //         case Ball::BallType::FULL:
    //             cv::circle(firstFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 0, 0), 3);
    //             break;
    //         case Ball::BallType::HALF:
    //             cv::circle(firstFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 255, 0), 2);
    //             break;
    //         case Ball::BallType::WHITE:
    //             cv::circle(firstFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(255, 255 , 255), 2);
    //             break;
    //         case Ball::BallType::BLACK:
    //             cv::circle(firstFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 0, 255), 3);
    //             break;
    //         case Ball::BallType::UNKNOWN:
    //             break;
    //         default:
    //             break;
    //         }
    //     std::cout << "Ball pos: " << ball.getWhiteRatio() << "Ball class: " << ball.typeToString()  << std::endl;
    // }

    // cv::imshow("n",firstFrame);

    // cv::waitKey(0);

    cv::imshow("n",firstFrame);

    //============================================================================

    // Get the starting timepoint
    auto start = high_resolution_clock::now();

    
    //============================================================================

    for( video >> frame; !frame.empty(); video >> frame){

        bool allBallsFound = tracker.update(frame, balls);
        if(!allBallsFound){

            balls = ballDetector.detectballsAlt(frame);
            balls = ballClassifier.classify(balls, frame);
            tracker = BallTracker(frame, balls);
            tracker.update(frame, balls);
            
        }

        for(Ball ball : balls){
            cv::circle(frame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 255, 128));
        }

        cv::imshow("Masked frame", frame);

        cv::Mat drawing = draw.updateDrawing(balls);
        cv::imshow("Draw", drawing);
        // 4. Associate class to balls
        // BallClassifier::classify(balls, frame);

        cv::waitKey(1);
    }

    // Get the ending timepoint
    auto end = high_resolution_clock::now();

    // Calculate the duration
    auto duration = duration_cast<milliseconds>(end - start);

    // Output the duration
    std::cout << "someFunction() took " << duration.count() << " milliseconds." << std::endl;

    video.release();
    cv::waitKey(0);
    


    // return 0;




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
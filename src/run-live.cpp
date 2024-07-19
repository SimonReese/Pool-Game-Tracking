/**
 * @author Simone Peraro.
 * 
 * This file is the main runner for the real time pool tracking system.
 */

#include <iostream>
#include <chrono>

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "TableSegmenter.h"
#include "Ball.h"
#include "BallDetector.h"
#include "BallClassifier.h"
#include "BallTracker.h"
#include "Draw.h"

/**
 * Main runner needs an argument with a path to the video which needs to be analyzed and a path to save the video file.
 */
int main(int argc, char* argv[]){
    // Check if video path is privided
    if (argc < 3){
        std::cerr << "Please provide an input video path and a output folder path for output video." << std::endl;
        return -1;
    }
    // Read path string
    std::string inputFile = argv[1];
    std::string outputDir = argv[2];
    // Compute output video and drawing name
    std::string clipName = inputFile.substr(inputFile.find_last_of('/')+1);
    std::string videoName = cv::utils::fs::join(outputDir ,"output-video-" + clipName);
    std::string drawName = cv::utils::fs::join(outputDir, "drawing-" + clipName.substr(0, clipName.find_last_of('.'))) + ".png";
    
    std::cout << "Saving output video to " << videoName << std::endl;
    std::cout << "Final drawing will be saved to " << drawName << std::endl;

    // Try to open input video
    cv::VideoCapture video(inputFile);
    if(!video.isOpened()){  // Return error if cannot open
        std::cerr << "Error. Unable to open video " << inputFile << std::endl;
        return -1;
    }

    // Get video fps and compute milliseconds interval between each frame
    double fps = video.get(cv::CAP_PROP_FPS);
    int milliseconds = 1000/fps;
    // Set target frametime
    std::chrono::duration<int, std::milli> targetFrameTIme(milliseconds); 
    // Prepare output video
    cv::VideoWriter outVideo(
        videoName, 
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
        fps, 
        cv::Size(video.get(cv::CAP_PROP_FRAME_WIDTH), video.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    std::cout << cv::utils::fs::join(outputDir, videoName) << std::endl;
    if(!outVideo.isOpened()){
        std::cerr << "Error. Unable to write video " << videoName << std::endl;
        return -1;
    }

    // Declare used objects
    TableSegmenter segmenter;
    Draw draw;
    BallDetector ballDetector;
    // Setup to record starting time
    std::chrono::system_clock::time_point startTime, endTime; // Global start, end time
    std::chrono::system_clock::time_point frameStartTime, frameEndTime; // Frame start, end time
    std::chrono::system_clock::duration elapsedTime, remainingTime; // Frame elapsed and remaining time
    double meanFrameTime = 0; // To compute mean frame time


    // Sart reading video and extract first frame
    cv::Mat firstFrame;
    video >> firstFrame;
    // Record start time
    startTime = std::chrono::system_clock::now();

    // 1. Get table mask
    cv::Mat mask = segmenter.getTableMask(firstFrame);

    // 2. Get table corners
    std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);

    // 3. Detect balls
    //std::vector<Ball> balls = ballDetector.detectBalls(firstFrame, mask, corners);
    std::vector<Ball> balls = ballDetector.detectballsAlt(firstFrame);

    // 4. Compute perspective effect for drawing
    draw.computePrespective(corners);

    // 5. Classify balls
    BallClassifier ballClassifier;
    balls = ballClassifier.classify(balls, firstFrame);

    // 6. Start tracking the balls
    BallTracker tracker(firstFrame, balls);

    // 7. Draw first frame
    cv::Mat drawing = draw.updateDrawing(balls);
    //cv::imshow("Draw", drawing);
    cv::Mat overlay = Draw::displayOverlay(firstFrame, drawing);
    cv::imshow("Overlay", overlay);
    outVideo << overlay;
    cv::Mat frame;
    // Looping to read all frames
    for( video >> frame; !frame.empty(); video >> frame){
        // Store current time
        frameStartTime = std::chrono::system_clock::now();
        
        // cv::imshow("Video", frame);  // Show video frame
        
        /* Show masked frame
        cv::Mat maskedFrame = segmenter.getMaskedImage(frame, mask);
        cv::imshow("Masked video", maskedFrame);
        */

        // Update ball tracking
        bool allBallsFound = tracker.update(frame, balls);
        // Check if we found all balls
        if(!allBallsFound){
            // If some balls got lost, try to detect them againq
            balls = ballDetector.detectballsAlt(frame); // detect
            balls = ballClassifier.classify(balls, frame); // classify
            tracker = BallTracker(frame, balls); // create new tracker with the detected balls
            tracker.update(frame, balls); // update tracker with current frames
        }

        /* Draw circles over detected balls
        cv::Mat circlesFrame = frame.clone();
        for(Ball ball : balls){
            cv::circle(circlesFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 255, 128));
        }
        cv::imshow("CircleFrame", circlesFrame);
        */
        
        // Update current game status drawing
        drawing = draw.updateDrawing(balls); // last frame drawing will be saved along with the video
        // cv::imshow("Draw", drawing);     // Show update drawing
        // Update overlay drawing
        overlay = Draw::displayOverlay(frame, drawing);
        cv::imshow("Overlay", overlay);

        // Write overlay frame to video
        outVideo << overlay;

        // Record ending time
        frameEndTime = std::chrono::system_clock::now();
        // Compute time remaining to display next frame at target framerate
        elapsedTime = frameEndTime - frameStartTime;  // time spent tracking and drawing
        remainingTime = targetFrameTIme - elapsedTime; // time remaining
        meanFrameTime += std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count(); // record time required to compute each frame
        // Check if we need to wait a few milliseconds  before moving on to the next frame
        if (std::chrono::duration_cast<std::chrono::milliseconds>(remainingTime).count() <= 0){
            // If remaining time is negative, we are running late, we need to move to next frame as early as possible
            cv::waitKey(1);
        } else {
            // Otherwise, we wait the remaining amout of time to keep the target framerate
            cv::waitKey(std::chrono::duration_cast<std::chrono::milliseconds>(remainingTime).count());
        }
        
    }

    // Save end time
    endTime = std::chrono::system_clock::now();
    // Close video resource
    video.release();
    outVideo.release();
    // Save last drawing image
    cv::imwrite(drawName, drawing);
    // Print time statistics 
    meanFrameTime /= (video.get(cv::CAP_PROP_FRAME_COUNT) -1); // mean elaboration time
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime); //
    std::cout << "Mean frame time: " << meanFrameTime << "ms" << std::endl;
    

    // Output the duration
    std::cout << "Runner took " << duration.count() << " ms." << std::endl;

    cv::waitKey(0);
}
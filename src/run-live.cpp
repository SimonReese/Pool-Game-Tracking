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

const bool DEBUG_MODE = true; // Change this to run in debug mode

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
    cv::VideoCapture inVideo(inputFile);
    if(!inVideo.isOpened()){  // Return error if cannot open
        std::cerr << "Error. Unable to open video " << inputFile << std::endl;
        return -1;
    }

    // Get video fps and compute milliseconds interval between each frame
    double fps = inVideo.get(cv::CAP_PROP_FPS);
    int milliseconds = 1000/fps;
    double frameNumber = inVideo.get(cv::CAP_PROP_FRAME_COUNT);
    // Set target frametime
    std::chrono::duration<int, std::milli> targetFrameTime(milliseconds);

    // Prepare output video
    cv::VideoWriter outVideo(
        videoName, 
        cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 
        fps, 
        cv::Size(inVideo.get(cv::CAP_PROP_FRAME_WIDTH), inVideo.get(cv::CAP_PROP_FRAME_HEIGHT))
    );
    // Check output video is open
    if(!outVideo.isOpened()){
        std::cerr << "Error. Unable to write video " << videoName << std::endl;
        return -1;
    }

    // Declare used objects
    TableSegmenter segmenter;
    Draw draw;
    BallDetector ballDetector;

    // Start reading video and extract first frame
    cv::Mat firstFrame;
    inVideo >> firstFrame;
    // Record start time
    auto startTime = std::chrono::system_clock::now();

    // 1. Get table mask
    cv::Mat mask = segmenter.getTableMask(firstFrame);

    // 2. Get table corners
    std::vector<cv::Point2i> corners = segmenter.getFieldCorners(mask);

    // 3. Detect balls
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
    if(DEBUG_MODE){cv::imshow("Draw", drawing);} // Show drawing

    // 8. Show overlay and write it to output video
    cv::Mat overlay = Draw::displayOverlay(firstFrame, drawing);
    cv::imshow("Overlay", overlay);
    outVideo << overlay;

    
    // Looping to read all frames
    cv::Mat frame; // Current frame
    for( inVideo >> frame; !frame.empty(); inVideo >> frame){
        // Record current time
        auto frameStartTime = std::chrono::system_clock::now();        

        // Update ball tracking
        bool allBallsFound = tracker.update(frame, balls);
        if(!allBallsFound){
            // If some balls got lost, try to detect them againq
            balls = ballDetector.detectballsAlt(frame); // detect
            balls = ballClassifier.classify(balls, frame); // classify
            tracker = BallTracker(frame, balls); // create new tracker with the detected balls
            tracker.update(frame, balls); // update tracker with current frames
        }

        // Update current game status drawing
        drawing = draw.updateDrawing(balls); // last frame drawing will be saved along with the video

        if (DEBUG_MODE) {
            // Show masked frame
            cv::Mat maskedFrame = segmenter.getMaskedImage(frame, mask);
            cv::imshow("Masked video", maskedFrame);
            // Draw bounding boxes over detected balls
            cv::Mat bboxes = frame.clone();
            for(Ball ball : balls){
                //cv::circle(circlesFrame, ball.getBallCenter(), ball.getBallRadius(), cv::Scalar(0, 255, 128));
                cv::rectangle(bboxes, ball.getBoundingBox(), cv::Scalar(51, 255, 255));
            }
            // Show Bounding boxes and drawing drawing
            cv::imshow("Bounding boxes", bboxes);
            cv::imshow("Draw", drawing);     
        }
        
        // Update overlay drawing
        overlay = Draw::displayOverlay(frame, drawing);
        cv::imshow("Overlay", overlay);

        // Write overlay frame to video
        outVideo << overlay;

        // Compute time remaining to display next frame at target framerate
        auto frameEndTime = std::chrono::system_clock::now(); // ending time
        auto frameTime = frameEndTime - frameStartTime;  // time spent tracking and drawing
        auto remainingTime = targetFrameTime - frameTime; // time remaining
        // Check if we need to wait a few milliseconds  before moving on to the next frame
        if (remainingTime <= std::chrono::milliseconds(0)){
            // If remaining time is negative, we are running late, we need to move to next frame as early as possible
            cv::waitKey(1);
        } else {
            // Otherwise, we wait the remaining amout of time to keep the target framerate
            cv::waitKey(remainingTime.count());
        }        
    }
    // Save end time
    auto endTime = std::chrono::system_clock::now();

    // Close video resources
    inVideo.release();
    outVideo.release();
    // Save last drawing image
    cv::imwrite(drawName, drawing);
    
    // Output the mean frame elaboration time
    auto duration = (endTime - startTime);
    double meanFrameTime = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count() / frameNumber;
    std::cout << "Mean frame elaboration time is: " << meanFrameTime << " ms" << std::endl;

    cv::waitKey(0);
}
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <BallsDetection.h>
#include <FieldGeometryAndMask.h>
#include <Ball.h>
#include <chrono>

#include "BallTracker.h"

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

std::string OUTPUT_DATASET = "../res/predictions";
std::string OUTPUT_CLIP = "/game1_clip1";




// void my_HSV_callback2(int event, int x, int y, int flags, void* userdata, std::string bboxFileName, std::string maskFileName){
//     if(event == cv::EVENT_LBUTTONDOWN){
//         cv::Mat image = *(cv::Mat*) userdata;

//         cv::Mat less_blur_image = image.clone();

//         cv::Mat no_blur_image = image.clone();

//         cv::GaussianBlur(less_blur_image,less_blur_image,cv::Size(3,3),0,0); // used to find balls later on

//         cv::GaussianBlur(image,image,cv::Size(7,7),0,0); // used to find field mask
        
//         cv::Vec3b mean_color = fieldMeanColor(image,11);

//         cv::Mat filled_field_contour = computeFieldMask(image,mean_color);

//         cv::Mat approximate_field_lines = findFieldLines(filled_field_contour);

//         vector<cv::Point2i> sorted_corners = findFieldCorners(approximate_field_lines);

//         vector<cv::Point> boundaries_contours_poly = defineBoundingPolygon(sorted_corners,approximate_field_lines);

//         cv::Mat only_table_image = no_blur_image.clone();

//         //removes everything from the initial image apart from the pixels defined by the mask that segments the field
//         for (int i = 0; i < image.size().height; i++){
//             for (int j = 0; j < image.size().width; j++){
//                 if( filled_field_contour.at<uchar>(i,j) == 0){
//                     only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
//                 }
//             }
//         }

//         vector<Ball> balls = findBalls(only_table_image, filled_field_contour, boundaries_contours_poly, sorted_corners);

//         drawBallsHSVChannels(balls, image);

//         cv::Mat field_and_balls_mask = drawBallsOnFieldMask(filled_field_contour,balls);

//         vector<cv::Rect> boundRect = findBoundingRectangles(field_and_balls_mask);

//         /*updates the bounding box value of each ball by assigning the bounding box that has center closest to the center of the circle that defines the ball*/
//         int offset_thres = 1;
//         for (int h = 0; h < boundRect.size(); h++){
//             for (int b = 0; b < balls.size(); b++){
//                 if((boundRect[h].x+boundRect[h].width/2) >= balls[b].getBallPosition()[0]-offset_thres && (boundRect[h].x+boundRect[h].width/2) <= balls[b].getBallPosition()[0]+offset_thres){
//                     if((boundRect[h].y+boundRect[h].height/2) >= balls[b].getBallPosition()[1]-offset_thres && (boundRect[h].y+boundRect[h].height/2) <= balls[b].getBallPosition()[1]+offset_thres){
//                         balls[b].setBoundingBox(boundRect[h]);
//                     }
//                 }
//             }   
//         }
//         // find bounding boxes of all circles and draws them
        
//         drawBoundingBoxesHSVChannels(balls, image);

//         writeBboxToFile(bboxFileName, balls);

//         cv::imwrite(maskFileName,field_and_balls_mask); //salva mask del campo e palline --> background = 0 campo = 255 palline = 127


//     }
// }

// void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
//     my_HSV_callback2(event, x, y, flags, userdata, "bounding_box_output.txt", "maschera.png");
// }

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
const std::string INITIAL_BBOX_FILENAME = "first_frame_bounding_box_output.txt";
const std::string FINAL_BBOX_FILENAME = "last_frame_bounding_box_output.txt";
const std::string INITIAL_MASK_FILENAME = "first_frame_mask.png";
const std::string FINAL_MASK_FILENAME = "last_frame_mask.png";

int main(int argc, char* argv[])
{


    using namespace std::chrono;

    


    // if (argc < 3){
    //     std::cout << "Please enter dataset path (usually in (root)/res/Dataset/) and predictions path (usually in (root)/res/predictions/)" << std::endl;
    // }
    // std::string datasetPath = argv[1];
    // std::string predictionsPath = argv[2];
    // std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);
    // for(cv::String game : gameFolders){
    //     // Create folders for predictions
    //     cv::utils::fs::createDirectories(predictionsPath + "/" + game + "/" + BOX_DIR);
    //     cv::utils::fs::createDirectories(predictionsPath + "/" + game + "/" + MASK_DIR);
    //     // Get all frames in game folder
    //     std::vector<cv::String> frames = listFrames(datasetPath, game, FRAMES_DIR);
    //     // For each frame
    //     for (cv::String frame : frames){
    //         // Open frame
    //         std::string fullFramePath = datasetPath + "/" + game + "/" + "/" + FRAMES_DIR + "/" + frame;
    //         cv::Mat image = cv::imread(fullFramePath);
    //         cv::Mat imageHSV;
    //         cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);
    //         // Compute boundig box and mask file name
    //         std::string bboxFileName = frame;
    //         int index = bboxFileName.rfind(".");
    //         bboxFileName = bboxFileName.erase(index, std::string::npos);
    //         std::string maskFileName = bboxFileName;
    //         bboxFileName = predictionsPath + "/" + game + "/" + BOX_DIR + "/" + bboxFileName + "_bbox" +".txt";
    //         maskFileName = predictionsPath + "/" + game + "/" + MASK_DIR + "/" + maskFileName + ".png";
    //         my_HSV_callback2(cv::EVENT_LBUTTONDOWN, 0, 0, 0, &imageHSV, bboxFileName, maskFileName);
    //     }
    // }
    // std::string clip_path = "/game1_clip1";

    std::string video_clip_path = argv[1];

    cv::VideoCapture cap(video_clip_path);
 
    cv::Mat frame;

    cap >> frame;

    cv::Mat frame_copy; /*copy of the frame used for later processing and keeping intact the original frame*/
    cv::cvtColor(frame,frame_copy,cv::COLOR_BGR2HSV);

    cv::Mat less_blur_image = frame_copy.clone(); /*copy of the frame that will be blurred*/

    cv::Mat no_blur_image = frame_copy.clone(); /*copy of the frame without any kind of blur*/

    cv::GaussianBlur(less_blur_image,less_blur_image,cv::Size(3,3),0,0); // used to find balls later on

    cv::GaussianBlur(frame_copy,frame_copy,cv::Size(7,7),0,0); // used to find field mask
        
    cv::Vec3b mean_color = fieldMeanColor(frame_copy,11);

    cv::Mat filled_field_contour = computeFieldMask(frame_copy,mean_color);
    cv::imshow("Maskera", filled_field_contour);
    // cv::waitKey(0);

    cv::Mat approximate_field_lines = findFieldLines(filled_field_contour);

    vector<cv::Point2i> sorted_corners = findFieldCorners(approximate_field_lines);

    vector<cv::Point> boundaries_contours_poly = defineBoundingPolygon(sorted_corners,approximate_field_lines);

    cv::Mat only_table_image = no_blur_image.clone(); /*copy of the frame without any operation performed on it*/

    //removes everything from the initial image apart from the pixels defined by the mask that segments the field
    for (int i = 0; i < frame_copy.size().height; i++){
        for (int j = 0; j < frame_copy.size().width; j++){
            if( filled_field_contour.at<uchar>(i,j) == 0){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
    }

    vector<Ball> balls = findBalls(only_table_image, filled_field_contour, boundaries_contours_poly, sorted_corners);

    drawBallsHSVChannels(balls, frame_copy);

    std::cout << "halo" << std::endl;

    cv::Mat field_and_balls_mask = drawBallsOnFieldMask(filled_field_contour,balls);  

    vector<cv::Rect> boundRect = findBoundingRectangles(field_and_balls_mask);

    

    for (int h = 0; h < boundRect.size(); h++){
        for (int b = 0; b < balls.size(); b++){
            if((boundRect[h].x+boundRect[h].width/2) >= balls[b].getBallPosition()[0]-1 && (boundRect[h].x+boundRect[h].width/2) <= balls[b].getBallPosition()[0]+1){
                if((boundRect[h].y+boundRect[h].height/2) >= balls[b].getBallPosition()[1]-1 && (boundRect[h].y+boundRect[h].height/2) <= balls[b].getBallPosition()[1]+1){
                    balls[b].setBoundingBox(boundRect[h]);
                }
            }
        }   
    }

    // find bounding boxes of all circles and draws them
        
    drawBoundingBoxesHSVChannels(balls, frame_copy);

    writeBboxToFile(INITIAL_BBOX_FILENAME, balls);    

    std::cout << "halo" << std::endl;

    cv::imwrite(INITIAL_MASK_FILENAME,field_and_balls_mask);


    //============================================================================

    // Get the starting timepoint
    auto start = high_resolution_clock::now();

    
    //============================================================================


    
    std::cout <<"halloz"<< std::endl;

    std::vector<cv::Ptr<cv::Tracker>> trackers = createTrackers(balls,frame);
    std::cout <<"halloz"<< std::endl;

    // cv::VideoWriter output("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

    std::vector<cv::Rect> rois;
    
    std::vector<cv::Point> ballMovement;
    // cv::cvtColor(frame,frame,cv::COLOR_HSV2BGR);

    for ( ;; ){

        
        // get frame from the video
        cap >> frame;
        
        // stop the program if no more images
        if(frame.rows==0 || frame.cols==0)
        break;
        
        rois.clear();
        ballMovement.clear();

        //update the tracking result
        for (int h = 0; h < trackers.size(); h++){
            cv::Rect roi;
            trackers[h]->update(frame,roi);
            rois.push_back(roi);
        }

        ballMovement = computeBallMovement(balls,rois); /*balls still have the position relative to the previous frame*/ /*this vector contains pair of point that represent the ball movement from a frame to the next*/

        updateBallValues(balls,rois);

        //debug test to see if bounding box and center of circle that defines ball gets updated by the function
        for (int g = 0; g < balls.size(); g++){
            cv::rectangle(frame, balls[g].getBoundingBox(), cv::Scalar(255,255,255),1,cv::LINE_AA);
            //cv::circle(frame,cv::Point(static_cast<int>(balls[g].getBallPosition()[0]),static_cast<int>((balls[g].getBallPosition()[1]))),static_cast<int>(balls[g].getBallPosition()[2]), cv::Scalar(255,255,255),-1);
        }

        // show image with the tracked object
        cv::imshow("tracker",frame);

        //quit on ESC button
        if(cv::waitKey(1)==27)break;

        
        //output.write(frame);
    }

    // Get the ending timepoint
    auto end = high_resolution_clock::now();

    // Calculate the duration
    auto duration = duration_cast<milliseconds>(end - start);

    // Output the duration
    std::cout << "someFunction() took " << duration.count() << " milliseconds." << std::endl;


    writeBboxToFile(FINAL_BBOX_FILENAME, balls);

    cv::Mat final_field_mask = drawBallsOnFieldMask(filled_field_contour,balls);

    cv::imwrite(FINAL_MASK_FILENAME,final_field_mask);

    /*release resources used for reading the frames of the input video and for saving the output video*/
    // output.release(); 
	cap.release();
    return 0;
}

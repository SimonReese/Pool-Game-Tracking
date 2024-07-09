#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include "BallsDetection.h" 
#include "FieldGeometryAndMask.h"


std::vector<Ball> findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point> boundaries_contours_poly, std::vector<cv::Point2i> sorted_corners){

        //find the circles around the balls --> parameters of hough_circles found after some manual testing to find a trade-off between all initial frames of the 10 clips
        std::vector<cv::Mat> channels_masked_table;
        cv::split(only_table_image,channels_masked_table);
        ///
        cv::Mat exit1(only_table_image.size().height,only_table_image.size().width,CV_8UC3); // Mat that contains circles that will be used to find bounding boxes
        ///
        std::vector<cv::Vec3f> circles;
        HoughCircles(channels_masked_table[1], circles, cv::HOUGH_GRADIENT, 1.2, channels_masked_table[1].rows/27, 160, 14.5, 6, 13);

        //////////

        cv::Vec3b mean_color2 = fieldMeanColor(only_table_image,11);

        /////////

        //////
        uchar h_thr = 10;
        uchar s_thr = 255;  //parameters were obtained by various manual tries 
        uchar v_thr = 125;
        uchar h_l,s_l,v_l,h_h,s_h,v_h;

        h_l = (mean_color2[0]-h_thr < 0) ? 0 : mean_color2[0]-h_thr;
        h_h = (mean_color2[0]+h_thr > 179) ? 179 : mean_color2[0]+h_thr;

        s_l = (mean_color2[1]-s_thr < 0) ? 0 : mean_color2[1]-s_thr;
        s_h = (mean_color2[1]+s_thr > 255) ? 255 : mean_color2[1]+s_thr;

        v_l = (mean_color2[2]-v_thr < 0) ? 0 : mean_color2[2]-v_thr;
        v_h = (mean_color2[2]+v_thr > 255) ? 255 : mean_color2[2]+v_thr;
        //////

        //draws the circles around the balls
        for( size_t i = 0; i < circles.size(); i++ ){
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);
            int radius = c[2];

            if(field_contour.at<uchar>(c[1],c[0]) == 255){
                if(channels_masked_table[0].at<uchar>(c[1],c[0]) < h_h && channels_masked_table[0].at<uchar>(c[1],c[0]) > h_l){
                    if(channels_masked_table[1].at<uchar>(c[1],c[0]) < s_h && channels_masked_table[1].at<uchar>(c[1],c[0]) > s_l){
                        if(channels_masked_table[2].at<uchar>(c[1],c[0]) < v_h && channels_masked_table[2].at<uchar>(c[1],c[0]) > v_l){
                            circles.erase(circles.begin()+i);
                            i--;
                            continue;
                        }   
                    }
                }
                
                if (cv::pointPolygonTest(boundaries_contours_poly,center,true) < 8.2){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }

                bool remove = false; //to try to remove outliers circles found at field corners holes
                for (int j = 0; j < sorted_corners.size(); j++){
                   if (sqrt(pow(sorted_corners[j].x-center.x,2)+pow(sorted_corners[j].y-center.y,2)) < 31.0){
                        remove = true;
                    } 
                }
                
                if(remove){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }
                

                
            }else{
                circles.erase(circles.begin()+i);
                i--;
                continue;
            }
        }

    std::vector<Ball> balls;
    for (int h = 0; h < circles.size(); h++){
        balls.push_back(Ball(circles[h]));
    }
    

    return balls;
}

void drawBallsHSVChannels(std::vector<Ball> balls, cv::Mat &image){
        for (int i = 0; i < balls.size(); i++){
            cv::circle(image, cv::Point2i(static_cast<int>(balls[i].getBallPosition()[0]), static_cast<int>(balls[i].getBallPosition()[1])), static_cast<int>(balls[i].getBallPosition()[2]), cv::Scalar(45, 255, 255), 1, cv::LINE_AA);
        }
}

std::vector<cv::Rect> findBoundingRectangles(const cv::Mat field_mask_and_balls){

        cv::Mat bbox_edges(field_mask_and_balls.size(),CV_8U);
        cv::Canny(field_mask_and_balls,bbox_edges,100,400);
        std::vector<std::vector<cv::Point> > contours;
        findContours( bbox_edges, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

        std::vector<cv::Rect> boundRect;
        cv::Rect rectHolder;
        
        for( size_t i = 0; i < contours.size(); i++ ){

            if(rectHolder == cv::boundingRect(contours[i])){ //check to remove multiple bounding boxes exaclty stacked one over the other
                contours.erase(contours.begin()+i);
                i--;
                continue;
            }
            rectHolder = boundingRect( contours[i] );

            if(field_mask_and_balls.at<uchar>(rectHolder.y+rectHolder.height/2,rectHolder.x+rectHolder.width/2) != 127){ //removes unwanted bounding box that do not frame any ball
                contours.erase(contours.begin()+i);
                i--;
                continue;
            }

            if(rectHolder.height*rectHolder.width > 1000 || rectHolder.height*rectHolder.width < 100){ //removes to small bounding boxes and too large bounding boxes 
                contours.erase(contours.begin()+i);
                i--;
                continue;
            }

            boundRect.push_back(rectHolder);
        }

    return boundRect;
}

void drawBoundingBoxesHSVChannels(std::vector<Ball> balls, cv::Mat &image){

        for (int i = 0; i < balls.size(); i++){
            cv::rectangle( image, balls[i].getBoundingBox().tl(), balls[i].getBoundingBox().br(), cv::Scalar(45, 255, 255), 1 );
        }

}
//balls have all the same class, when complete it will write also the correct class of the ball
void writeBboxToFile(std::string bboxFileName, std::vector<Ball> balls){

        std::ofstream outfile(bboxFileName);

        for( size_t i = 0; i < balls.size(); i++ ){
            //saves to file all the bounding boxes --> top left x coord top left y coord width height class=1 because classification is not implemented yet
            outfile << std::to_string(balls[i].getBoundingBox().tl().x) << " " << std::to_string(balls[i].getBoundingBox().tl().y) << " " << std::to_string(balls[i].getBoundingBox().width) << " " << std::to_string(balls[i].getBoundingBox().height) << " " << 1 << std::endl;
        }

        outfile.close();

}
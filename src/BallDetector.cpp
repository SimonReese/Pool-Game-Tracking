/**
 * @author Alessandro Bozzon
 */
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fstream>
#include "BallDetector.h"
#include "TableSegmenter.h"
#include "Ball.h"

cv::Vec3b BallDetector::fieldMeanColor(const cv::Mat& image, int kernel_size)const{

    int x = image.size().width/2; /*defines the coordinates of the center of the image*/
    int y = image.size().height/2;

     std::vector<cv::Vec3b> vec; /*retrieve the values of all the pixels inside the kernel window and stores them in a vector*/
        for (int i = y-kernel_size/2; i <= y+kernel_size/2 && i < image.size().height; i++)
        {
            for (int j = x-kernel_size/2; j <= x+kernel_size/2 && j < image.size().width; j++)
            {
                if(i < 0 || j < 0){
                    continue;
                }else{
                    vec.push_back(image.at<cv::Vec3b>(i,j));
                }
            }
            
        }
        

        /*evaluates average value for h,s,v channels*/
        uint32_t h = 0;
        uint32_t s = 0;
        uint32_t v = 0;
        uchar k = 0;
        for (k; k < vec.size(); k++)
        {
            h = h + (uint32_t)(vec[k].val[0]);
            s = s + (uint32_t)(vec[k].val[1]);
            v = v + (uint32_t)(vec[k].val[2]); 
        }
    cv::Vec3b mean_color(h/k,s/k,v/k); /*stores the average value of the 3 channels in a vector*/
    return mean_color;

}

std::vector<Ball> BallDetector::findBalls(const cv::Mat only_table_image, const cv::Mat field_contour, std::vector<cv::Point2i> sorted_corners){

        
        std::vector<cv::Mat> channels_masked_table;
        cv::split(only_table_image,channels_masked_table); /*splits the image into the 3 individual channels H,S,V*/

        std::vector<cv::Vec3f> circles;
        //find the circles around the balls --> parameters of hough_circles found after some manual testing to find a trade-off between all initial frames of the 10 clips
        /*uses only the S channel of the image*/
        HoughCircles(channels_masked_table[1], circles, cv::HOUGH_GRADIENT, 1.2, channels_masked_table[1].rows/27, 160, 14.5, 6, 13);

        cv::Vec3b mean_color2 = fieldMeanColor(only_table_image,11);

        /*values to define upper and lower thresholds on the HSV channels for later removal of false positive circles*/
        /*values have been fixed after performing some tests on the first frame of each of the 10 video clips and averaging the obtained results*/
        uchar h_thr = 10;
        uchar s_thr = 255;  
        uchar v_thr = 125;
        uchar h_l,s_l,v_l,h_h,s_h,v_h;

        h_l = (mean_color2[0]-h_thr < 0) ? 0 : mean_color2[0]-h_thr;
        h_h = (mean_color2[0]+h_thr > 179) ? 179 : mean_color2[0]+h_thr;

        s_l = (mean_color2[1]-s_thr < 0) ? 0 : mean_color2[1]-s_thr;
        s_h = (mean_color2[1]+s_thr > 255) ? 255 : mean_color2[1]+s_thr;

        v_l = (mean_color2[2]-v_thr < 0) ? 0 : mean_color2[2]-v_thr;
        v_h = (mean_color2[2]+v_thr > 255) ? 255 : mean_color2[2]+v_thr;

        /*draws circles detected earlier with HoughCircles function only if the satisfy some constraints*/
        for( size_t i = 0; i < circles.size(); i++ ){
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);
            int radius = c[2];

            /*checks if the center of the circle is located inside the playing field and the color of the pixel corresponding to the center of the circle is not similar to the one of the field within a given range*/
            if(field_contour.at<uchar>(c[1],c[0]) == 255){
                if(channels_masked_table[0].at<uchar>(c[1],c[0]) < h_h && channels_masked_table[0].at<uchar>(c[1],c[0]) > h_l){
                    if(channels_masked_table[1].at<uchar>(c[1],c[0]) < s_h && channels_masked_table[1].at<uchar>(c[1],c[0]) > s_l){
                        if(channels_masked_table[2].at<uchar>(c[1],c[0]) < v_h && channels_masked_table[2].at<uchar>(c[1],c[0]) > v_l){
                            circles.erase(circles.begin()+i); /*erase the circle from the vector containing them if it does not satisfy the constraints to be considered a real circle corresponding to a ball*/
                            i--;
                            continue;
                        }   
                    }
                }
                
                /*erases all the circles with a distance less than min_dist from the polygon that defines the field boundaries*/
                /*it erases most of the false positive cirlces that are located on the railings of the playing field*/
                double min_dist = 8.2;
                if (cv::pointPolygonTest(getTableContours(),center,true) < min_dist){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }

                /*tries to remove all the false positive circles that can be generated by the pot holes of the field by looking at the distance of the center of the circle from the corners of the field*/
                bool remove = false; //to try to remove outliers circles found at field corners holes
                double min_distance_from_corner = 31.0;
                for (int j = 0; j < sorted_corners.size(); j++){
                   if (sqrt(pow(sorted_corners[j].x-center.x,2)+pow(sorted_corners[j].y-center.y,2)) < min_distance_from_corner){
                        remove = true;
                    } 
                }
                
                if(remove){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }
                

            /*removes all the circles that falls outside the playing field*/    
            }else{
                circles.erase(circles.begin()+i);
                i--;
                continue;
            }
        }

    /*creates the final vector containing all the balls, initialized with only the center and the radius of the circle that identifies the single ball*/
    std::vector<Ball> balls;
    for (int h = 0; h < circles.size(); h++){
        balls.push_back(Ball(circles[h]));
    }
    

    return balls;
}

std::vector<Ball> BallDetector::detectBalls(const cv::Mat &image, const cv::Mat &tableMask, std::vector<cv::Point2i> tableCorners){
    
    cv::Mat hsvImage; 
    cv::cvtColor(image,hsvImage,cv::COLOR_BGR2HSV);

    //removes everything from the initial image apart from the pixels defined by the mask that segments the field
    for (int i = 0; i < hsvImage.size().height; i++){
        for (int j = 0; j < hsvImage.size().width; j++){
            if( (tableMask.at<uchar>(i,j)) == 0){
                hsvImage.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
            }
        }
    }

    defineBoundingPolygon(tableCorners,image);

    std::vector<Ball> balls = findBalls(image, tableMask, tableCorners);

    cv::Mat field_and_balls_mask = drawBallsOnFieldMask(tableMask,balls);
    std::vector<cv::Rect> boundRect = this->findBoundingRectangles(field_and_balls_mask);

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

    /*removes balls that still have the bounding box with default values, it removes wrongly detected circles that do not correspond to balls*/
    for (int i = 0; i < balls.size(); i++){
        if (balls[i].getBoundingBox().x == 0 && balls[i].getBoundingBox().y == 0
            && balls[i].getBoundingBox().height == 2 && balls[i].getBoundingBox().width == 2){
                balls.erase(balls.begin() + i);
                i--;
        }
    }

    return balls;
}

cv::Mat BallDetector::drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls){/*should draw all the class correctly on the mask*/

        /*draws each individual ball of the balls vector on the image that contains the mask of the field only without the balls*/
        cv::Mat field_mask_and_balls = field_mask.clone();

        Ball::BallType type;

        for (int i = 0; i < balls.size(); i++){

            /*gets ball class*/
            type = balls[i].getBallType(); 
            /*draws single ball on the mask using info about position and class*/
            cv::circle(field_mask_and_balls, cv::Point2i(balls[i].getBallPosition()[0], balls[i].getBallPosition()[1]), balls[i].getBallPosition()[2], static_cast<int>(type), cv::FILLED, cv::LINE_8);
        }

    return field_mask_and_balls;
}

std::vector<cv::Rect> BallDetector::findBoundingRectangles(const cv::Mat field_mask_and_balls){ /*modified to consider all the class color for the balls*/

        cv::Mat bbox_edges = cv::Mat::zeros(field_mask_and_balls.size(),CV_8U);
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
            rectHolder = boundingRect( contours[i] ); /*computes the bounding box of the single contour*/

            /*removes unwanted bounding box that do not frame any ball by looking at the color of the pixel the mask that correspond to the center of the bounding box*/
            /*gets color of the central pixel of the bounding box*/
            uchar pixel_color = field_mask_and_balls.at<uchar>(rectHolder.y+rectHolder.height/2,rectHolder.x+rectHolder.width/2);
            /*1: graylevel of white ball on mask, 2:graylevel of black ball on mask, 3:graylevel of solid color balls on mask, 4:graylevel of balls with stripes on mask, 6: graylevel of unknown class balls*/
            if(pixel_color != 1 && pixel_color != 2 && pixel_color != 3 && pixel_color != 4 && pixel_color != 6){
                contours.erase(contours.begin()+i);
                i--;
                continue;
            }

            /*removes all the bounding boxes that are too small or too large to represent a bounding box of a ball*/
            if(rectHolder.height*rectHolder.width > 1000 || rectHolder.height*rectHolder.width < 100){ 
                contours.erase(contours.begin()+i);
                i--;
                continue;
            }

            boundRect.push_back(rectHolder); /*stores the found bounding box in the vector that will be returned by the function*/
        }

    return boundRect;
}

/*funtion used to re-detect balls when the tracker fails to update*/
std::vector<Ball> BallDetector::detectballsAlt(cv::Mat frame){

    TableSegmenter t;
    
    cv::Mat frame_copy = frame.clone(); /*copy of the frame used for later processing and keeping intact the original frame*/
    cv::cvtColor(frame,frame_copy,cv::COLOR_BGR2HSV); /*color conversion to HSV because it is better suited for working with colors*/

    cv::Mat no_blur_image = frame_copy.clone(); /*copy of the frame without any kind of blur*/

    cv::GaussianBlur(frame_copy,frame_copy,cv::Size(7,7),0,0); // used to find field mask 
        
    cv::Mat filled_field_contour = t.getTableMask(frame); /*extracts the table mask without the balls on top of it*/

    std::vector<cv::Point2i> sorted_corners = t.getFieldCorners(filled_field_contour); /*computes the 4 corners of the field*/

    defineBoundingPolygon(sorted_corners,frame); /*defines the bounding polygon of the field that will be used later for the removal of some wrongly detected cirlces that do not correspond to the balls*/

    cv::Mat only_table_image = t.getMaskedImage(no_blur_image,filled_field_contour); /*generate an image that has all the pixels set to 0 except the ones that lies underneath the mask that represent the field*/

    std::vector<Ball> balls = this->findBalls(only_table_image, filled_field_contour, sorted_corners); /*finds the balls on the table for the given frame*/

    cv::Mat field_and_balls_mask = drawBallsOnFieldMask(filled_field_contour,balls); /*computes the mask of the field and the balls that will be used to later compute the bounding boxes of the detected balls*/

    std::vector<cv::Rect> boundRect = this->findBoundingRectangles(field_and_balls_mask); /*computes the bounding boxes of all the detected balls on the field*/

    
    /*maps the found bounding boxes to the correct balls by looking at the position of the center of the bounding box and the center of the ball*/
    for (int h = 0; h < boundRect.size(); h++){
        for (int b = 0; b < balls.size(); b++){
            if((boundRect[h].x+boundRect[h].width/2) >= balls[b].getBallPosition()[0]-1 && (boundRect[h].x+boundRect[h].width/2) <= balls[b].getBallPosition()[0]+1){
                if((boundRect[h].y+boundRect[h].height/2) >= balls[b].getBallPosition()[1]-1 && (boundRect[h].y+boundRect[h].height/2) <= balls[b].getBallPosition()[1]+1){
                    balls[b].setBoundingBox(boundRect[h]);
                }
            }
        }   
    }

    /*removes all the circles from the vector of balls that did not receive a bounding box, it removes wrongly detected cirlces*/
    for (int i = 0; i < balls.size(); i++){
        /*checks if a ball still has the bounding box setted to the default values*/
        if (balls[i].getBoundingBox().x == 0 && balls[i].getBoundingBox().y == 0
            && balls[i].getBoundingBox().height == 2 && balls[i].getBoundingBox().width == 2){
                balls.erase(balls.begin() + i);
                i--;
        }
    }

    return balls;
}

void BallDetector::defineBoundingPolygon(std::vector<cv::Point2i> sorted_corners, const cv::Mat frame){

        cv::Mat boundaries = cv::Mat::zeros(frame.size(),CV_8U);

        /*draws the 4 lines that delimits the playing field based on the 4 sorted corners found before*/
        for (int i = 0; i < sorted_corners.size(); i++){
            cv::line(boundaries,sorted_corners[i%sorted_corners.size()],sorted_corners[(i+1)%sorted_corners.size()],255,1);
        }

        /*finds the contours of the playing field based on the 4 drawed lines*/
        std::vector<std::vector<cv::Point>> boundaries_contours;
        findContours( boundaries, boundaries_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

        /*removes all the contours found except the one with the largest area*/
        double contour_max_area = 0;
        for (int i = 0; i < boundaries_contours.size(); i++){
            if(cv::contourArea(boundaries_contours[i]) > contour_max_area){  
                contour_max_area = cv::contourArea(boundaries_contours[i]);
            }
        }

        for (int i = 0; i < boundaries_contours.size(); i++){
            if(cv::contourArea(boundaries_contours[i]) != contour_max_area){  
                boundaries_contours.erase(boundaries_contours.begin()+i);
                i--;
            }
        }
        
        
        /*defines the bounding polygon that delimits the playing field*/
        std::vector<cv::Point> boundaries_contours_poly( boundaries_contours.size() );
        
        for( size_t i = 0; i < boundaries_contours.size(); i++ )
        {
            approxPolyDP( boundaries_contours[i], boundaries_contours_poly, 1, true );
        }

    /*sets table contours for this object*/
    this->tableContours = boundaries_contours_poly;
}


    void BallDetector::saveMaskToFile(const cv::Mat &mask, std::vector<Ball> balls, std::string predictedMaskPath){
        cv::Mat complete_mask = cv::Mat::zeros(mask.size(),CV_8U);
        complete_mask = drawBallsOnFieldMask(mask,balls);
        /*converts the pixels of the mask that represent the field from the value of 255 to the value of 5, all the other colors are kept the same */
        std::vector<uchar> conversionTable(256);
        for (int i = 0; i < conversionTable.size(); i++){
            if(i == 255){
                conversionTable[i] = 5;
            }else{
                conversionTable[i] = i;
            }  
        }
        cv::LUT(complete_mask,conversionTable,complete_mask);
        /*saves the mask on the disk at the given location*/
        cv::imwrite(predictedMaskPath,complete_mask);
    }

    void BallDetector::saveBoxesToFile(std::vector<Ball> balls, std::string predictedBBoxPath){

        std::ofstream myfile;
        myfile.open (predictedBBoxPath);
        /*retireves bounding box and class information for each ball and saves them in a file*/
        for(int i = 0; i < balls.size(); i++){
            cv::Rect bbox = balls[i].getBoundingBox();
            int top_left_x = bbox.x;
            int top_left_y = bbox.y;
            int width = bbox.width;
            int height = bbox.height;
            int ball_class = static_cast<int>(balls[i].getBallType());
            std::string ball_output;
            myfile << top_left_x << " " << top_left_y << " " << width << " " << height << " " << ball_class << std::endl;
        }
        myfile.close();
    }

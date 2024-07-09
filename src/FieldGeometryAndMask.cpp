#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "FieldGeometryAndMask.h"
#include "Ball.h"

cv::Vec3b fieldMeanColor(const cv::Mat image, int kernel_size){

    int x = image.size().width/2;
    int y = image.size().height/2;

     std::vector<cv::Vec3b> vec;
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
        
        /*MASK1 FIELD CONTOUR*/ 
        //evaluates average value for h,s,v and range of value to consider to form the mask
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
    cv::Vec3b mean_color(h/k,s/k,v/k);
    return mean_color;

}


cv::Mat computeFieldMask(const cv::Mat image, cv::Vec3b mean_color){

        uchar h_threshold = 14;
        uchar s_threshold = 80;  //parameters were obtained by various manual tries
        uchar v_threshold = 137;
        cv::Mat mask(image.size().height,image.size().width,CV_8U);
        uchar h_low,s_low,v_low,h_high,s_high,v_high;

        h_low = (mean_color[0]-h_threshold < 0) ? 0 : mean_color[0]-h_threshold;
        h_high = (mean_color[0]+h_threshold > 179) ? 179 : mean_color[0]+h_threshold;

        s_low = (mean_color[1]-s_threshold < 0) ? 0 : mean_color[1]-s_threshold;
        s_high = (mean_color[1]+s_threshold > 255) ? 255 : mean_color[1]+s_threshold;

        v_low = (mean_color[2]-v_threshold < 0) ? 0 : mean_color[2]-v_threshold;
        v_high = (mean_color[2]+v_threshold > 255) ? 255 : mean_color[2]+v_threshold;

        // creates a mask of the field
        for (int i = 0; i < mask.size().height; i++)
        {
            for (int j = 0; j < mask.size().width; j++)
            {
                if(image.at<cv::Vec3b>(i,j)[0] < h_low || image.at<cv::Vec3b>(i,j)[0] > h_high){
                    mask.at<uchar>(i,j) = 0;
                }else if(image.at<cv::Vec3b>(i,j)[1] < s_low || image.at<cv::Vec3b>(i,j)[1] > s_high){
                    mask.at<uchar>(i,j) = 0;
                }else if(image.at<cv::Vec3b>(i,j)[2] < v_low || image.at<cv::Vec3b>(i,j)[2] > v_high){
                    mask.at<uchar>(i,j) = 0;
                }else{
                    mask.at<uchar>(i,j) = 255;
                }
            }
            
        }

        // evaluates the contour of the field
        cv::Mat field_contour = cv::Mat::zeros(image.size().height,image.size().width,CV_8U);
        std::vector<std::vector<cv::Point> > contours;
        std::vector<cv::Vec4i> hierarchy;

        cv::findContours(mask,contours,hierarchy,cv::RETR_TREE,cv::CHAIN_APPROX_SIMPLE);

        // find the contour with the highest area value
        double max_area = 0.0;
        for (int i = 0; i < contours.size(); i++){
            if(max_area < cv::contourArea(contours[i])){
                max_area = cv::contourArea(contours[i]);
            }
        }

        //removes all contours with an area different from the max area value
        for (int i = 0; i < contours.size(); i++){
            if(max_area != cv::contourArea(contours[i])){
                contours.erase(contours.begin()+i);
                hierarchy.erase(hierarchy.begin()+i);
                i--;
            } 
        }
        
        //draws the contour with the max area and fills it <--- THIS ONE IS THE BEST MASK OF THE FIELD ALONE WITHOUT CONSIDERING THE BALLS
        cv::drawContours(field_contour,contours,-1,255,cv::FILLED,cv::LINE_8,hierarchy,0);

    return field_contour;

}


cv::Mat findFieldLines(const cv::Mat field_contour){

//computes edges of the contour to help find the 4 lines that delimit the field
        cv::Mat edges(field_contour.size().height,field_contour.size().width,CV_8U);

        cv::Canny(field_contour,edges,127,127);

        // finds the lines from the edges
        std::vector<cv::Vec2f> lines; // will hold the results of the detection
        cv::HoughLines(edges, lines, 1.15, CV_PI/180, 125, 0, 0); // runs the actual detection
        std::vector<cv::Point2f> pts; //will contain only 2 lines points
        cv::Mat only_lines(field_contour.size().height,field_contour.size().width,CV_8U);

        //removes all lines with a similar rho value --> erases close lines
        for(int i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0], theta = lines[i][1];  
            for (int j = i+1; j < lines.size(); j++){
                if((lines[j][0] <= rho+40.0) && (lines[j][0] >= rho-40.0)){
                    lines.erase(lines.begin()+j);
                    j--;
                }
            }
        }

        //draws the lines that delimit the field
        for( size_t i = 0; i < lines.size(); i++ ){
            float rho = lines[i][0], theta = lines[i][1];  
            cv::Point2i pt1, pt2;
            double a = cos(theta), b = sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 15000*(-b));
            pt1.y = cvRound(y0 + 15000*(a));
            pt2.x = cvRound(x0 - 15000*(-b));
            pt2.y = cvRound(y0 - 15000*(a));
            cv::line(only_lines, pt1, pt2, 255, 1, cv::LINE_AA);
	    }
    return only_lines;
}


std::vector<cv::Point2i> findFieldCorners(const cv::Mat approximate_field_lines){

 /* NEED TO WRITE CODE TO FIND CORNER POINTS FROM INTERSECTION OF 4 LINES --> USE HARRIS CORNER DETECTOR OR SOMETHING SIMILAR*/
        /*Shi-Tomasi corner to find 4 points --> sort them as top_left, top_right, bottom_right, bottom_left*/

        std::vector<cv::Point2i> corners;
        std::vector<cv::Point2i> sorted_corners;

        cv::goodFeaturesToTrack(approximate_field_lines,corners, 4, 0.01, 10, cv::noArray(), 5);

        int y_min1 = INT16_MAX, y_min2 = INT16_MAX;
        int index1 = 0, index2 = 0, index3 = 0, index4 = 0;
        int y_max1 = 0, y_max2= 0;

        for (int i = 0; i < corners.size(); i++){
            if(corners[i].y < y_min1 && corners[i].y < y_min2){
                y_min2 = y_min1;
                y_min1 = corners[i].y;
                index2 = index1;
                index1 = i;
            }else if(corners[i].y >= y_min1 && corners[i].y <= y_min2){
                y_min2 = corners[i].y;
                index2 = i;
            }

            if(corners[i].y > y_max1 && corners[i].y > y_max2){
                y_max2 = y_max1;
                y_max1 = corners[i].y;
                index3 = index4;
                index4 = i;
            }else if(corners[i].y <= y_max1 && corners[i].y >= y_max2){
                y_max2 = corners[i].y;
                index3 = i;
            }
        }
        
        if(corners[index1].x <= corners[index2].x){
            sorted_corners.push_back(corners[index1]);
            sorted_corners.push_back(corners[index2]);
        }else{
            sorted_corners.push_back(corners[index2]);
            sorted_corners.push_back(corners[index1]);
        }

        if(corners[index4].x >= corners[index3].x){
            sorted_corners.push_back(corners[index4]);
            sorted_corners.push_back(corners[index3]);
        }else{
            sorted_corners.push_back(corners[index3]);
            sorted_corners.push_back(corners[index4]);
        }

    return sorted_corners;

}


std::vector<cv::Point> defineBoundingPolygon(std::vector<cv::Point2i> sorted_corners, const cv::Mat approximate_field_lines){

        cv::Mat boundaries(approximate_field_lines.size(),CV_8U);

        for (int i = 0; i < sorted_corners.size(); i++){
            cv::line(boundaries,sorted_corners[i%sorted_corners.size()],sorted_corners[(i+1)%sorted_corners.size()],255,1);
        }
        
        //////////

        std::vector<std::vector<cv::Point>> boundaries_contours;
        findContours( boundaries, boundaries_contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

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
        
        
        
        std::vector<cv::Point> boundaries_contours_poly( boundaries_contours.size() );
        
        for( size_t i = 0; i < boundaries_contours.size(); i++ )
        {
            approxPolyDP( boundaries_contours[i], boundaries_contours_poly, 1, true );
        }

    return boundaries_contours_poly;
}

//currently all balls are of the same class!!!
cv::Mat drawBallsOnFieldMask(const cv::Mat field_mask, std::vector<Ball> balls){

        cv::Mat field_mask_and_balls = field_mask.clone();
        for (int i = 0; i < balls.size(); i++){
            cv::circle(field_mask_and_balls, cv::Point2i(static_cast<int>(balls[i].getBallPosition()[0]), static_cast<int>(balls[i].getBallPosition()[1])), static_cast<int>(balls[i].getBallPosition()[2]), 127, cv::FILLED, cv::LINE_AA);
        }
    return field_mask_and_balls;
}



    
    
    










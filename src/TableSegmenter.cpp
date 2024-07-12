/**
 * @author .
 * TODO: who will maintain this file?
 */

#include "TableSegmenter.h"

#include <iostream>

#include <opencv2/imgproc.hpp>

cv::Vec3b TableSegmenter::fieldMeanColor(const cv::Mat& image, int kernel_size) const{

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

cv::Mat TableSegmenter::computeFieldMask(const cv::Mat image, cv::Vec3b mean_color) const{
    uchar h_threshold = 14;
    uchar s_threshold = 80;  /*parameters to compute upper and lower threshold for each channel. Each value was obtained by averaging the quality of the mask obtained by this function*/ 
    uchar v_threshold = 137; /*on the first frame of each video clip */

    cv::Mat mask(image.size().height,image.size().width,CV_8U);
    uchar h_low,s_low,v_low,h_high,s_high,v_high;

    /*definition of the upper and lower threshold for each channel based on the 3 fixed values and the mean color received as input of the function*/
    h_low = (mean_color[0]-h_threshold < 0) ? 0 : mean_color[0]-h_threshold;
    h_high = (mean_color[0]+h_threshold > 179) ? 179 : mean_color[0]+h_threshold;

    s_low = (mean_color[1]-s_threshold < 0) ? 0 : mean_color[1]-s_threshold;
    s_high = (mean_color[1]+s_threshold > 255) ? 255 : mean_color[1]+s_threshold;

    v_low = (mean_color[2]-v_threshold < 0) ? 0 : mean_color[2]-v_threshold;
    v_high = (mean_color[2]+v_threshold > 255) ? 255 : mean_color[2]+v_threshold;

    /*creates a rough mask of the playing field by setting to zero all the pixels that have at least one of the 3 values of the h,s,v channels outside of the ranges defined above*/
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

    /*computes the contours of the mask found in order to find the contour with the largest area that corresponds to the playing field in order to compute the final field mask without considering the balls*/
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

    // removes all contours with an area different from the max area value
    for (int i = 0; i < contours.size(); i++){
        if(max_area != cv::contourArea(contours[i])){
            contours.erase(contours.begin()+i);
            hierarchy.erase(hierarchy.begin()+i);
            i--;
        } 
    }
    
    //draws the contour with the max area and fills it. The result correspond to the final field mask without considering the balls
    cv::drawContours(field_contour,contours,-1,255,cv::FILLED,cv::LINE_8,hierarchy,0);

    return field_contour;
}

cv::Mat TableSegmenter::findFieldLines(const cv::Mat field_contour)const{
    //computes edges of the contour of the field mask to help find the 4 lines that delimit the field
    cv::Mat edges(field_contour.size().height,field_contour.size().width,CV_8U);

    cv::Canny(field_contour,edges,127,127);

    // finds the lines from the edges
    std::vector<cv::Vec2f> lines; // will hold the results of the detection
    cv::HoughLines(edges, lines, 1.15, CV_PI/180, 125, 0, 0); // runs the actual detection of the lines
    std::vector<cv::Point2f> pts; //will contain only 2 lines points
    cv::Mat only_lines(field_contour.size().height,field_contour.size().width,CV_8U); /*cv::Mat that will contain the 4 drawed lines corresponding to the 4 lines delimiting the field*/

    //removes all lines with a similar rho value --> erases close lines
    float rho_thres = 40.0; /*value to define the range of lines with similar rho to eliminate. Value fixed after looking at the average result obtained on the first frame of each clip*/
    for(int i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0], theta = lines[i][1];  
        for (int j = i+1; j < lines.size(); j++){
            if((lines[j][0] <= rho+rho_thres) && (lines[j][0] >= rho-rho_thres)){  
                lines.erase(lines.begin()+j);
                j--;
            }
        }
    }
    
    /*draws the 4 lines that delimit the field; it is a rough draw of the lines that goues across the entire image*/
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

std::vector<cv::Point2i> TableSegmenter::findFieldCorners(const cv::Mat approximate_field_lines) const{
    std::vector<cv::Point2i> corners;
    std::vector<cv::Point2i> sorted_corners;

    /*Shi-Tomasi corner detection to find 4 points that will corespond to the 4 corners of the playing field*/
    cv::goodFeaturesToTrack(approximate_field_lines,corners, 4, 0.01, 10, cv::noArray(), 5);

    int y_min1 = INT16_MAX, y_min2 = INT16_MAX;
    int index1 = 0, index2 = 0, index3 = 0, index4 = 0;
    int y_max1 = 0, y_max2= 0;
    
    /*sorting of the 4 found corners in order to have them stored as "top_left, top_right, bottom_right, bottom_left" in the final vector that is returned by this function*/
    /*find the 2 corners with the smallest y value and the 2 corners with the highest y value*/
    /*index1 contains the index of the lowest y value corner, index2 contains the index of the second lowest y value corner*/
    /*index4 contains the index fo the highets y value corner, index3 contains the index of the second highest y value corner*/
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
    /*defines the final sorting of the corners based on the comparison of the x value of the two lowest y value corners*/
    if(corners[index1].x <= corners[index2].x){
        sorted_corners.push_back(corners[index1]);
        sorted_corners.push_back(corners[index2]);
    }else{
        sorted_corners.push_back(corners[index2]);
        sorted_corners.push_back(corners[index1]);
    }

    /*defines the final sorting of the corners based on the comparison of the x value of the two highest y value corners*/
    if(corners[index4].x >= corners[index3].x){
        sorted_corners.push_back(corners[index4]);
        sorted_corners.push_back(corners[index3]);
    }else{
        sorted_corners.push_back(corners[index3]);
        sorted_corners.push_back(corners[index4]);
    }
    return sorted_corners;
}

cv::Mat TableSegmenter::getTableMask(const cv::Mat &frame) const{
    cv::Mat hsv;
    cv::cvtColor(frame,hsv,cv::COLOR_BGR2HSV);
    cv::Mat blurred;
    cv::GaussianBlur(hsv,blurred,cv::Size(7,7),0,0); // used to find field mask
    cv::Vec3b mean_color = fieldMeanColor(blurred,11);
    cv::Mat filled_field_contour = computeFieldMask(blurred,mean_color);
    
    return filled_field_contour;
}

std::vector<cv::Point2i> TableSegmenter::getFieldCorners(const cv::Mat &mask) const{
    cv::Mat approximate_field_lines = findFieldLines(mask);
    std::vector<cv::Point2i> sorted_corners = findFieldCorners(approximate_field_lines);
    return sorted_corners;
}
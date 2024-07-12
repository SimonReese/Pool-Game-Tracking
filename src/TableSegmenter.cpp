/**
 * @author .
 * TODO: who will maintain this file?
 */

#include "TableSegmenter.h"

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

cv::Mat TableSegmenter::getTableMask(const cv::Mat &frame) const{

    cv::Mat blurred;
    cv::GaussianBlur(frame,blurred,cv::Size(7,7),0,0); // used to find field mask
    cv::Vec3b mean_color = fieldMeanColor(blurred,11);
    cv::Mat filled_field_contour = computeFieldMask(blurred,mean_color);
    
    return filled_field_contour;
}

std::vector<cv::Point2i> TableSegmenter::findFieldCorners(){

    return std::vector<cv::Point2i>();
}
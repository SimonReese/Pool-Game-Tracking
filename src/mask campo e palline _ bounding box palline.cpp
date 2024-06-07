#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat image = *(cv::Mat*) userdata;

        // takes central pixel of the image
        x = image.size().width/2;
        y = image.size().height/2;
        cout << "pixel position: " << "(" << x << "," << y << ") "<< "HSV values: " << cv::Vec3b(image.at<cv::Vec3b>(x,y)[0],image.at<cv::Vec3b>(x,y)[1],image.at<cv::Vec3b>(x,y)[2]) << "\n";

        // takes all pixels colors in a kernel of size 11x11 centered on the center of the image
        vector<cv::Vec3b> vec;
        for (int i = y-Y_VALUE/2; i <= y+Y_VALUE/2 && i < image.size().height; i++)
        {
            for (int j = x-X_VALUE/2; j <= x+X_VALUE/2 && j < image.size().width; j++)
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
        
        uchar h_threshold = 35;
        uchar s_threshold = 60;  //parameters were obtained by various manual tries
        uchar v_threshold = 255;
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

        // region grow algorithm to find a mask to help find balls later, 

        vector<pair<int, int>> seed_set; // Stores the seeds
        seed_set.push_back(pair<int, int>(x, y)); //center of image

        cv::Mat visited_matrix_h = cv::Mat::zeros(image.size().height,image.size().width,CV_8UC1);
        cv::Mat hsv_image_help(image.size().height,image.size().width,CV_8UC3);
        cv::cvtColor(image,hsv_image_help,cv::COLOR_BGR2HSV);
        vector<cv::Mat> channels;
        cv::split(hsv_image_help,channels);

        cv::Mat visited_matrix_v = cv::Mat::zeros(image.size().height,image.size().width,CV_8UC1);

        while ( ! seed_set.empty() ){
            // Get a point from the list
            pair<int, int> this_point = seed_set.back();
            seed_set.pop_back();
            
            int x = this_point.first;
            int y = this_point.second;
            unsigned char pixel_value = channels[2].at<unsigned char>(cv::Point(x,y));
                                                                                                                                
            // Visit the point
            visited_matrix_v.at<unsigned char>(cv::Point(x, y)) = 255;

            // for each neighbour of this_point
            for (int j = y - 1; j <= y + 1; ++j)
            {
                // vertical index is valid
                if (0 <= j && j < channels[2].rows)
                {
                    for (int i = x - 1; i <= x + 1; ++i)
                    {
                        // hozirontal index is valid
                        if (0 <= i && i < channels[2].cols)
                        {
                            unsigned char neighbour_value = channels[2].at<unsigned char>(cv::Point(i, j));
                            unsigned char neighbour_visited = visited_matrix_v.at<unsigned char>(cv::Point(i, j));
                            
                            if (!neighbour_visited &&
                                fabs(neighbour_value - pixel_value) <= (0.9 / 100.0 * 255.0)) // neighbour is similar to this_point
                            {
                                seed_set.push_back(pair<int, int>(i, j));
                            }
                        }
                    }
                }
            }
        }

        // evaluates the contour of the field
        cv::Mat field_contour(image.size().height,image.size().width,CV_8U);
        vector<vector<cv::Point> > contours;
        vector<cv::Vec4i> hierarchy;

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

        //computes edges of the contour to help find the 4 lines that delimit the field
        cv::Mat edges(image.size().height,image.size().width,CV_8U);

        cv::Canny(field_contour,edges,127,127);

        // finds the lines from the edges
        std::vector<cv::Vec2f> lines; // will hold the results of the detection
        cv::HoughLines(edges, lines, 1, CV_PI/180, 100, 0, 0); // runs the actual detection
        std::vector<cv::Point2f> pts; //will contain only 2 lines points
        cv::Mat only_lines(image.size().height,image.size().width,CV_8U);

        //removes all lines with a similar rho value --> erases close lines
        for(int i = 0; i < lines.size(); i++ ){
        float rho = lines[i][0], theta = lines[i][1];  
            for (int j = i+1; j < lines.size(); j++){
                if((lines[j][0] <= rho+30.0) && (lines[j][0] >= rho-30.0)){
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
            pt1.x = cvRound(x0 + 10000*(-b));
            pt1.y = cvRound(y0 + 10000*(a));
            pt2.x = cvRound(x0 - 10000*(-b));
            pt2.y = cvRound(y0 - 10000*(a));
            cv::line(only_lines, pt1, pt2, 128, 2, cv::LINE_AA);
	    }

        /* NEED TO WRITE CODE TO FIND CORNER POINTS FROM INTERSECTION OF 4 LINES --> USE HARRIS CORNER DETECTOR OR SOMETHING SIMILAR*/

        cv::Mat only_table_image = image.clone();

        //removes everything from the initial image apart from the pixels defined by the mask that segments the field
        for (int i = 0; i < mask.size().height; i++){
            for (int j = 0; j < mask.size().width; j++){
                if( field_contour.at<uchar>(i,j) == 0){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                }
            }
        }

        // colors all the pixels under the mask found with region grow algorithm with a fucsia color to try to help finding the balls better later
        for (int i = 0; i < visited_matrix_v.size().height; i++){
            for (int j = 0; j < visited_matrix_v.size().width; j++){
                if( visited_matrix_v.at<uchar>(i,j) == 255){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(150,255,255);
                }
            }
        }

        /* DRAWING CIRCLES AROUND BALLS, BALLS ARE NOT PERFECTLY CIRCLED BUT GIVES A REALLY GOOD APPROXIMATION */
        /* OUT OF ALL THE BALLS IN THE INITIAL FRAMES, ONLY 1 BALL IS NOT DETECTED AT ALL */
        /* IT DETECTED ALSO SOME OUTLIERS THAT NEEDS TO BE REMOVED IN SOME WAY */

        //find the circles around the balls --> parameters of hough_circles found after some manual testing to find a trade-off between all initial frames of the 10 clips
        vector<cv::Mat> channels_masked_table;
        cv::split(only_table_image,channels_masked_table);
        cv::cvtColor(only_table_image,only_table_image,cv::COLOR_HSV2BGR);
        ///
        cv::Mat exit1(image.size().height,image.size().width,CV_8UC3); // MMat that contains circles that will be used to find bounding boxes
        ///
        vector<cv::Vec3f> circles;
        HoughCircles(channels_masked_table[1], circles, cv::HOUGH_GRADIENT, 1.253, channels_masked_table[1].rows/27, 180, 17, 6, 16);


        //draws the circles around the balls
        for( size_t i = 0; i < circles.size(); i++ ){
            cv::Vec3i c = circles[i];
            cv::Point center = cv::Point(c[0], c[1]);
            int radius = c[2];
            cv::circle( exit1, center, radius, cv::Scalar(122, 255, 20), 1, cv::LINE_AA);
        }

        // NEED TO WRITE CODE TO ERASE ALL THE OUTLIERS CIRCLES BEFORE DRAWING THEM --> COULD BE USEFUL TO LOOK AT COLORS INFORMATIONS


        // find bounding boxes of all circles and draws them
        // NEED TO WRITE THE BOUNDING BOX VALUES IN A FILE FOR EVALUATIONS OF PERFORMANCES
        // DATA THAT NEEDS TO BE WRITTEN IS INSIDE boundRect VECTOR
        cv::Mat canny_output;
        cv::Canny( exit1, canny_output, 100, 100 );
        
        vector<vector<cv::Point> > contours56;
        findContours( canny_output, contours56, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );
        
        vector<vector<cv::Point> > contours_poly( contours56.size() );
        vector<cv::Rect> boundRect( contours56.size() );
        vector<cv::Point2f>centers( contours56.size() );
        vector<float>radius( contours56.size() );
        
        for( size_t i = 0; i < contours56.size(); i++ )
        {
            approxPolyDP( contours56[i], contours_poly[i], 3, true );
            boundRect[i] = boundingRect( contours_poly[i] );
            minEnclosingCircle( contours_poly[i], centers[i], radius[i] );
        }


        for( size_t i = 0; i< contours56.size(); i++ )
        {
            rectangle( exit1, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0,255,0), 1 );
        }

        // cv::namedWindow("mask display");
        // cv::imshow("mask display", mask);
        // cv::namedWindow("edges display");
        // cv::imshow("edges display", edges);
        // cv::namedWindow("field display");
        // cv::imshow("field display", field_contour);
        cv::namedWindow("cropped field display");
        cv::imshow("cropped field display", exit1);


    }
}


int main(int argc, char** argv)
{
    cv::Mat image = cv::imread("C:/Users/Alebo/Desktop/Dataset-20240529T161618Z-001/game1_clip1/frames/frame_first.png");
    cv::Mat image2;
    cv::Mat image3 = cv::imread("C:/Users/Alebo/Desktop/Dataset-20240529T161618Z-001/game1_clip2/frames/frame_first.png");
    cv::Mat image4;
    cv::cvtColor(image,image2,cv::COLOR_BGR2HSV);
    cv::cvtColor(image3,image4,cv::COLOR_BGR2HSV);
    cv::namedWindow("hsv window");
    cv::imshow("hsv window",image);
    cv::setMouseCallback("hsv window", my_HSV_callback2, &image2);
    cv::namedWindow("hsv window2");
    cv::imshow("hsv window2",image3);
    cv::setMouseCallback("hsv window2", my_HSV_callback2, &image4);
    cv::waitKey(0);

    return 0;
}


// ONLY 1 MISSING BALL!!!
// game 4 frame 2 green ball is not detected
// not all circles are perfectly centered on the balls
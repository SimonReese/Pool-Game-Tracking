#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <fstream>

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

std::string OUTPUT_DATASET = "../res/predictions";
std::string OUTPUT_CLIP = "/game1_clip1";

void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat image = *(cv::Mat*) userdata;

        cv::Mat less_blur_image = image.clone();

        cv::Mat no_blur_image = image.clone();

        cv::GaussianBlur(less_blur_image,less_blur_image,cv::Size(3,3),0,0); // used to find balls later on

        cv::GaussianBlur(image,image,cv::Size(7,7),0,0); // used to find field mask

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
        cv::HoughLines(edges, lines, 1.15, CV_PI/180, 125, 0, 0); // runs the actual detection
        std::vector<cv::Point2f> pts; //will contain only 2 lines points
        cv::Mat only_lines(image.size().height,image.size().width,CV_8U);

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

        /* NEED TO WRITE CODE TO FIND CORNER POINTS FROM INTERSECTION OF 4 LINES --> USE HARRIS CORNER DETECTOR OR SOMETHING SIMILAR*/
        /*Shi-Tomasi corner to find 4 points --> sort them as top_left, top_right, bottom_right, bottom_left*/

        vector<cv::Point2i> corners;
        vector<cv::Point2i> sorted_corners;

        cv::goodFeaturesToTrack(only_lines,corners, 4, 0.01, 10, cv::noArray(), 5);

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

        cv::Mat boundaries(image.size(),CV_8U);

        for (int i = 0; i < sorted_corners.size(); i++){
            cv::line(boundaries,sorted_corners[i%sorted_corners.size()],sorted_corners[(i+1)%sorted_corners.size()],255,1);
        }
        
        //////////

        vector<vector<cv::Point>> boundaries_contours;
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
        
        
        
        vector<cv::Point> boundaries_contours_poly( boundaries_contours.size() );
        
        for( size_t i = 0; i < boundaries_contours.size(); i++ )
        {
            approxPolyDP( boundaries_contours[i], boundaries_contours_poly, 1, true );
        }

        //////////
        
        // fine corner find and sort

        cv::Mat only_table_image = no_blur_image.clone();

        //removes everything from the initial image apart from the pixels defined by the mask that segments the field
        for (int i = 0; i < mask.size().height; i++){
            for (int j = 0; j < mask.size().width; j++){
                if( field_contour.at<uchar>(i,j) == 0){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                }
            }
        }

        
        /* DRAWING CIRCLES AROUND BALLS, BALLS ARE NOT PERFECTLY CIRCLED BUT GIVES A REALLY GOOD APPROXIMATION */
        /* OUT OF ALL THE BALLS IN THE INITIAL FRAMES, ONLY 1 BALL IS NOT DETECTED AT ALL */
        /* IT DETECTED ALSO SOME OUTLIERS THAT NEEDS TO BE REMOVED IN SOME WAY */

        //find the circles around the balls --> parameters of hough_circles found after some manual testing to find a trade-off between all initial frames of the 10 clips
        vector<cv::Mat> channels_masked_table;
        cv::split(only_table_image,channels_masked_table);
        ///
        cv::Mat exit1(image.size().height,image.size().width,CV_8UC3); // Mat that contains circles that will be used to find bounding boxes
        ///
        vector<cv::Vec3f> circles;
        HoughCircles(channels_masked_table[1], circles, cv::HOUGH_GRADIENT, 1.2, channels_masked_table[1].rows/27, 160, 14.5, 6, 13);

        //////////

        // takes all pixels colors in a kernel of size 11x11 centered on the center of the image
        vector<cv::Vec3b> vec2;
        for (int i = y-Y_VALUE/2; i <= y+Y_VALUE/2 && i < no_blur_image.size().height; i++)
        {
            for (int j = x-X_VALUE/2; j <= x+X_VALUE/2 && j < no_blur_image.size().width; j++)
            {
                if(i < 0 || j < 0){
                    continue;
                }else{
                    vec2.push_back(no_blur_image.at<cv::Vec3b>(i,j));
                }
            }
            
        }
        
        /*MASK1 FIELD CONTOUR*/ 
        //evaluates average value for h,s,v and range of value to consider to form the mask
        uint32_t h2 = 0;
        uint32_t s2 = 0;
        uint32_t v2 = 0;
        uchar k2 = 0;
        for (k2; k2 < vec2.size(); k2++)
        {
            h2 = h2 + (uint32_t)(vec[k2].val[0]);
            s2 = s2 + (uint32_t)(vec[k2].val[1]);
            v2 = v2 + (uint32_t)(vec[k2].val[2]); 
        }
        cv::Vec3b mean_color2(h2/k2,s2/k2,v2/k2);


        /////////

        //////
        uchar h_thr = 4;
        uchar s_thr = 80;  //parameters were obtained by various manual tries 
        uchar v_thr = 60;
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
                
                if (cv::pointPolygonTest(boundaries_contours_poly,center,true) < 14){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }

                bool remove = false;
                for (int j = 0; j < sorted_corners.size(); j++){
                   if (sqrt(pow(sorted_corners[j].x-center.x,2)+pow(sorted_corners[j].y-center.y,2)) < 32.0){
                        remove = true;
                    } 
                }
                
                if(remove){
                    circles.erase(circles.begin()+i);
                    i--;
                    continue;
                }
                

                cv::circle( exit1, center, radius, cv::Scalar(122, 255, 20), 1, cv::LINE_AA);
                cv::circle( only_table_image, center, radius, cv::Scalar(45, 255, 255), 1, cv::LINE_AA);

                cv::circle(field_contour, center, radius, 127, cv::FILLED, cv::LINE_AA);
            }   
        }


        // NEED TO WRITE CODE TO ERASE ALL THE OUTLIERS CIRCLES BEFORE DRAWING THEM --> COULD BE USEFUL TO LOOK AT COLORS INFORMATIONS


        // find bounding boxes of all circles and draws them
        // NEED TO WRITE THE BOUNDING BOX VALUES IN A FILE FOR EVALUATIONS OF PERFORMANCES
        // DATA THAT NEEDS TO BE WRITTEN IS INSIDE boundRect VECTOR
        
        cv::Mat bbox_edges(field_contour.size(),CV_8U);
        cv::Canny(field_contour,bbox_edges,100,400);
        vector<vector<cv::Point> > contours56;
        findContours( bbox_edges, contours56, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE );

        vector<cv::Rect> boundRect;
        cv::Rect rectHolder;
        
        for( size_t i = 0; i < contours56.size(); i++ ){

            if(rectHolder == cv::boundingRect(contours56[i])){ //check to remove multiple bounding boxes exaclty stacked one over the other
                contours56.erase(contours56.begin()+i);
                i--;
                continue;
            }
            rectHolder = boundingRect( contours56[i] );

            if(field_contour.at<uchar>(rectHolder.y+rectHolder.height/2,rectHolder.x+rectHolder.width/2) != 127){ //removes unwanted bounding box that do not frame any ball
                contours56.erase(contours56.begin()+i);
                i--;
                continue;
            }

            if(rectHolder.height*rectHolder.width > 1000 || rectHolder.height*rectHolder.width < 100){ //removes to small bounding boxes and too large bounding boxes 
                contours56.erase(contours56.begin()+i);
                i--;
                continue;
            }

            boundRect.push_back(rectHolder);
        }

        std::ofstream outfile("bounding_box_output.txt");

        for( size_t i = 0; i < boundRect.size(); i++ ){
            //saves to file all the bounding boxes --> top left x coord top left y coord width height class=1 because classification is not implemented yet
            outfile << to_string(boundRect[i].tl().x) << " " << to_string(boundRect[i].tl().y) << " " << to_string(boundRect[i].width) << " " << to_string(boundRect[i].height) << " " << 1 << std::endl;
            rectangle( exit1, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0,255,0), 1 );
        }

        outfile.close();

        // cv::namedWindow("mask display");
        // cv::imshow("mask display", mask);
        // cv::namedWindow("edges display");
        // cv::imshow("edges display", edges);
        // cv::namedWindow("field display");
        // cv::imshow("field display", field_contour);
        // cv::namedWindow("cropped field display");
        // cv::imshow("cropped field display", exit1);
        // cv::namedWindow("cropped field display");
        cv::cvtColor(only_table_image,only_table_image,cv::COLOR_HSV2BGR);
        cv::imshow("cropped field display", only_table_image);

        cv::cvtColor(image,image,cv::COLOR_HSV2BGR);
        cv::imshow("angoli",image);

        cv::imshow("boh",exit1);

        cv::imwrite("maschera.jpeg",field_contour); //salva mask del campo e palline --> background = 0 campo = 255 palline = 127


    }
}


int main(int argc, char* argv[])
{
    if (argc < 2){
        std::cout << "Please enter dataset path (usually in (root)/res/Dataset/)" << std::endl;
    }
    std::string datasetPath = argv[1];

    std::string clip_path = "/game1_clip1";

    cv::Mat image = cv::imread( datasetPath + clip_path + "/frames/frame_first.png");
    cv::Mat image2;
    cv::Mat image3 = cv::imread( datasetPath + clip_path + "/frames/frame_first.png");
    cv::Mat image4;
    cv::cvtColor(image,image2,cv::COLOR_BGR2HSV);
    cv::cvtColor(image3,image4,cv::COLOR_BGR2HSV);
    cv::namedWindow("hsv window");
    cv::imshow("hsv window",image);
    cv::setMouseCallback("hsv window", my_HSV_callback2, &image2);
    //cv::namedWindow("hsv window2");
    //cv::imshow("hsv window2",image3);
    //cv::setMouseCallback("hsv window2", my_HSV_callback2, &image4);
    cv::waitKey(0);

    return 0;
}


// ONLY 1 MISSING BALL!!!
// game 4 frame 2 green ball is not detected
// circles are more or less centered on the balls
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "Draw.h"

// TODO: FIX UNUSED HEADERS
#include <algorithm>
#include <opencv2/core/mat.hpp>

/**
 * TODO: TO BE REMOVED WHEN THINGS WILL WORK
 */
std::vector<cv::Point> Draw::detectTableCorners() const{
    // 0. Smoothing
    cv::Mat smoothed;
    cv::medianBlur(this->currentFrame, smoothed, 5);
    // 1. Filter table plane ---------------
    // Convert to hsv
    cv::Mat HSVframe;
    cv::cvtColor(smoothed, HSVframe, cv::COLOR_BGR2HSV);

    // Compute mean HSV value in the middle of image // Maybe is better to first find some lines, then compute dominat color and later filter the table more precisely?
    int x_center, y_center;
    x_center = this->currentFrame.cols / 2;
    y_center = this->currentFrame.rows / 2;
    int distance = 10;
    double H, S, V, area = std::pow(2*distance, 2);
    cv::Vec3b values;
    for(int row = y_center - distance; row < y_center + distance; row++){
        for (int col = x_center - distance; col < x_center + distance; col++){
            values = HSVframe.at<cv::Vec3b>(cv::Point(col, row));
            // std::cout << "Reading values " << values << std::endl;
            H += static_cast<double>(values[0]) / area;
            S += static_cast<double>(values[1]) / area;
            V += static_cast<double>(values[2]) / area;
        }
    }
    // Define bounds for HSV
    int H_bound, S_bound, V_bound;
    H_bound = 20;
    S_bound = 50;
    V_bound = 130;
    int H_lower_bound = H - H_bound, H_upper_bound = H + H_bound;
    int S_lower_bound = S - S_bound, S_upper_bound = S + S_bound;
    int V_lower_bound = V - V_bound, V_upper_bound = V + V_bound;

    cv::Mat tableMask;
    cv::inRange(HSVframe, cv::Scalar(H_lower_bound, S_lower_bound, V_lower_bound), cv::Scalar(H_upper_bound, S_upper_bound, V_upper_bound), tableMask);
    
    // Clean the table mask
    cv::Mat elem1 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(25, 25));
    cv::Mat elem2 = cv::getStructuringElement(cv::MorphShapes::MORPH_RECT, cv::Size(40, 40));
    cv::morphologyEx(tableMask, tableMask, cv::MORPH_ERODE, elem1);
    cv::morphologyEx(tableMask, tableMask, cv::MORPH_CLOSE, elem2);

    // Find contours
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(tableMask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Find convex hull of those contours
    // Unpack all points
    std::vector<cv::Point> points;
    for(std::vector<cv::Point> contour : contours){
        for(cv::Point point : contour){
            points.push_back(point);
        }
    }
    // Find convex hull of table
    std::vector<cv::Point> convexHull;
    cv::convexHull(points, convexHull);
    // Fill the convex hull
    cv::fillConvexPoly(tableMask, convexHull, cv::Scalar(255));
    
    
    
    // NOT NEEDED - Just for showing field
    // Make AND between table and mask
    //cv::Mat maskedTable;
    //cv::bitwise_and(this->currentFrame, this->currentFrame, maskedTable, tableMask);
    // --------------------------

    // 2. Find lines
    cv::Mat edges;
    cv::Canny(tableMask, edges, 100, 200);
    
    cv::Mat sides;
    cv::cvtColor(edges, sides, cv::COLOR_GRAY2BGR);
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, CV_PI / 180, 100);
    // std::cout << lines.size() << std::endl;

    // We need to filter lines
    if (true){
    // 1. Divide them by angle;
    std::vector<cv::Vec2f> hLines, vLines;
    float hMean = 0, vMean = 0;
    for (cv::Vec2f line : lines){
        float rho = line[0]; // Will be used to compute mean of rhos'
        float theta = line[1];
        // std::cout << "Line theta " << theta << std::endl;
        if (theta > CV_PI - (CV_PI/4 + CV_PI/6) || theta < (CV_PI / 4) + (CV_PI / 6)){
            vLines.push_back(line);
            vMean += rho;
        }
        else {
            hLines.push_back(line);
            hMean += rho;
        }
    }
    // std::cout << "Sizes " << vLines.size() << " " << hLines.size() << std::endl;
    vMean = vMean / vLines.size();
    hMean = hMean / hLines.size();

    bool lower = false;
    bool higher = false;
    lines.clear(); // Clear all lines
    // Now for each angle, we just need two lines
    for (std::vector<cv::Vec2f>::iterator it = hLines.begin(); it != hLines.end(); ){
        float rho = (*it)[0];
        if ( !lower && rho < hMean) {
            lines.push_back(*it);
            lower = true;
            it++;
        }
        else if (!higher){
            lines.push_back(*it);
            higher = true;
            it++;
        }
        else {
            it = hLines.erase(it);
        }
    }
    higher = false;
    lower = false;
    for (std::vector<cv::Vec2f>::iterator it = vLines.begin(); it != vLines.end(); ){
        float rho = (*it)[0];
        if ( !lower && rho < vMean) {
            lines.push_back(*it);
            lower = true;
            it++;
        }
        else if (!higher){
            lines.push_back(*it);
            higher = true;
            it++;
        }
        else {
            it = vLines.erase(it);
        }
    }
    }
    // At this point we should have only 4 lines

    // Draw lines and save points
    std::vector<std::vector<cv::Point> > linePoints;
    for (int i = 0; i < lines.size(); i++){
        float rho = lines[i][0], theta = lines[i][1];
        // std::cout << "Filetred theta" << theta << std::endl;
        cv::Point start, end;
        start.x = std::round(rho * std::cos(theta) - 1000*std::sin(theta));
        start.y = std::round(rho * std::sin(theta) + 1000*std::cos(theta));

        end.x = std::round(rho * std::cos(theta) + 1000*std::sin(theta));
        end.y = std::round(rho * std::sin(theta) - 1000*std::cos(theta));
        std::cout << "Drawing " << theta << std::endl;
        cv::line(sides, start, end, cv::Scalar(0, 0, 255), 2);
        linePoints.push_back(std::vector<cv::Point>{start, end});
    }

    
    
    // 3. Find corners at intersections of line points

    // Lambda function to compute intersections VERY UGLY TODO
    // Finds the intersection of two lines, or returns false.
    // The lines are defined by (o1, p1) and (o2, p2).
    auto intersection = [](cv::Point2f o1, cv::Point2f p1, cv::Point2f o2, cv::Point2f p2){
        cv::Point2f x = o2 - o1;
        cv::Point2f d1 = p1 - o1;
        cv::Point2f d2 = p2 - o2;

        float cross = d1.x*d2.y - d1.y*d2.x;
        if (abs(cross) < /*EPS*/1e-8)
            return cv::Point2f(-1, -1);

        double t1 = (x.x * d2.y - x.y * d2.x)/cross;
        return o1 + d1 * t1;
    };

    // Corners 
    std::vector<cv::Point> corners;
    // Now, for the first two lines, compute intersections with the prependicular two other lines
    for(int i = 0; i < 2; i++){
        cv::Point intersect1 = intersection(linePoints[i][0], linePoints[i][1], linePoints[2][0], linePoints[2][1]);
        cv::Point intersect2 = intersection(linePoints[i][0], linePoints[i][1], linePoints[3][0], linePoints[3][1]);
        corners.push_back(intersect1);
        corners.push_back(intersect2);
        std::cout << "Corners: " << intersect1 << " | " << intersect2 << std::endl;
    }

    // Sort corners
    auto pointSort = [](cv::Point a, cv::Point b){
        // Top goes first if not aligned
        if (a.y != b.y)
            return a.y < b.y;
        return a.x < b.x; // Else left goes first
    };
    std::sort(corners.begin(), corners.end(), pointSort);
    // Swap last two elements, but only if 3rd is to the left than 4th
    if (corners[2].x <= corners[3].x)
        std::iter_swap(corners.begin()+2, corners.begin()+3);

    // TODO: debug order
    for(cv::Point point : corners){
        std::cout << point << " ";
    }
    std::cout << std::endl;
    
    cv::imshow("Output", tableMask);
    return corners;
}



Draw::Draw(std::string fieldPath)
    : fieldPath{fieldPath}{
}

void Draw::setCurrentFrame(const cv::Mat &currentFrame){
    currentFrame.copyTo(this->currentFrame);
}

void Draw::getGameDraw(cv::Mat &outputDrawing) const{
    // Currently just return the current frame
    std::vector<cv::Point> corners = detectTableCorners();
    outputDrawing = correctPrespective(corners);
}

/**
 * TOOD: define size of image to choose coordinates of perspective transfomration
 */
cv::Mat Draw::correctPrespective(const std::vector<cv::Point>& corners) const{
    /**
     * TODO: optimize casting ? keep order
     */
    std::vector<cv::Point2f> srcCoord;
    for(cv::Point p : corners){
        srcCoord.push_back((cv::Point2f) p);
    }

    // We want to check if table is oriented horizontaly or vertically
    float horiz = cv::norm(corners[0] - corners[1]);
    float vert = cv::norm(corners[1] - corners[2]);

    // We build destination points accordingly
    std::vector<cv::Point2f> destCoord;
    cv::Mat result;
    cv::Size dsize;
    if (horiz / vert < 1.7){
        // We will build a vertical pool table 
        destCoord = {
            cv::Point(49, 49),
            cv::Point(149, 49),
            cv::Point(149, 249),
            cv::Point(49, 249)
        };
        result = cv::Mat(cv::Size(100, 200), CV_8UC3);
        dsize = cv::Size(200, 400);
    }
    else {
        // We build a horizontal pool table
        destCoord = {
            cv::Point(49, 49),
            cv::Point(249, 49),
            cv::Point(249, 149),
            cv::Point(49, 149)
        };
        result = cv::Mat(cv::Size(200, 100), CV_8UC3);
        dsize = cv::Size(400, 200);
    }
    
    // Compute transformation matrix
    cv::Mat transformation = cv::getPerspectiveTransform(srcCoord, destCoord);

    // Correct perspective
    cv::warpPerspective(this->currentFrame, result, transformation, dsize);

    return result;
}

/**
 * @author Simone Peraro.
 */
#ifndef DRAW
#define DRAW

#include <string>

#include <opencv2/core.hpp>

#include "Ball.h"


/**
 * Class to draw a scheme of current game situation.
 */
class Draw{

private:

    /**
     * Path to the field schematic draw (vertical alignment)
     */
    const std::string verticalTablePath = "../res/assets/pool-table-350x640.png";

    /**
     * Path to the field schematic draw (horiziontal alignment)
     */
    const std::string horizontalTablePath = "../res/assets/pool-table-640x350.png";

    /**
     * Size of the table draw schematic. Values must respect the size of the actual table schematic. 
     * Those are the values with respect to the vertical image, and are swapped for the horizontal image.
     * @see `computePrespective()`
     */
    const cv::Size tableDrawSize = cv::Size(350, 640);

    /**
     * Padding used to map corners of the frame to corners of the table draw schematic
     */
    const int padding = 20;

    /**
     * This parameter is used to test if the table is placed vertically or horizontally with respect to the camera angle.
     * After we detectected the corners of the table, we test if the distance d1 of (top-left), (top-right) corners 
     * is lower than the distance d2 of (top-right), (bottom-right) corners: d1 < tableRatio * d2.
     * @see `computePrespective()`
     */
    const double tableRatio = 1.7;

    /**
     * Perspective transformation matrix
     */
    cv::Mat perspectiveTrasformation;

    /**
     * Check if we already computed the perspective transformation matrix
     */
    bool computedPerspective = false;

    /**
     * Store current drawing (without balls)
     */
    cv::Mat drawingNoBalls;


    /**
     * Returns a new image composed by background image and the overlapping image
     * @param background background image
     * @param overlapping the image that will be put over the background
     * @param position coordinates where the overlapping image will be placed with respect to the background image
     */
    cv::Mat drawOver(const cv::Mat& background, const cv::Mat& overlapping, const cv::Point2i position) const;

public:

    /**
     * Default constructor for initializing class.
     */
    Draw();

    /**
     * Return an image representing current game situation.
     * 
     * @param balls a vector of balls objects with positions and class
     * @param displacements a vector of end and start point where we want to draw the trajectory line. Order of points must be the same in all
     */
    cv::Mat updateDrawing(std::vector<Ball> balls);

    /**
     * Return perspective corrected table image
     * @param corners a vector of point corners from top left in clockwise order.
     */
    void computePrespective(const std::vector<cv::Point>& corners);

};
#endif
/**
 * @author Simone Peraro.
 */
#ifndef DRAW
#define DRAW

#include <string>
#include <tuple>

#include <opencv2/core.hpp>

#include "Ball.h"


/**
 * Class to draw a scheme of current game situation.
 */
class Draw{

private:
    // Current frame which needs to be drawed;
    cv::Mat currentFrame;

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
     * Store white ball schematic image
     */
    cv::Mat whiteBallPNG;

    /**
     * Store black ball schematic image
     */
    cv::Mat blackBallPNG;

    /**
     * Store solid ball schematic image
     */
    cv::Mat solidBallPNG;

    /**
     * Store striped ball schematic image
     */
    cv::Mat stripedBallPNG;

    /**
     * Store unknown ball schematic image
     */
    cv::Mat unknownBallPNG;

    /**
     * Detect corners of table in the frame.
     * TODO: remove!
     * 
     *  @deprecated We want to use the specific class
     */ 
    std::vector<cv::Point> detectTableCorners() const;

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
    cv::Mat updateDrawing(std::vector<Ball> balls, std::vector<std::tuple<cv::Point2f, cv::Point2f> > displacements);

    /**
     * Return perspective corrected table image
     * @param corners a vector of point corners from top left in clockwise order.
     */
    void computePrespective(const std::vector<cv::Point>& corners);

    

};
#endif
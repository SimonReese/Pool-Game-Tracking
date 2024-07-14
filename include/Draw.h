/**
 * @author Simone Peraro.
 */
#ifndef DRAW
#define DRAW

#include <string>
#include <opencv2/core.hpp>


/**
 * Class to draw a scheme of current game situation.
 */
class Draw{

private:
    // Current frame which needs to be drawed;
    cv::Mat currentFrame;

    /**
     * Path to fiels schematic
     */
    std::string fieldPath;

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
    cv::Mat drawNoBalls;

    // Detect corners of table in the frame
    std::vector<cv::Point> detectTableCorners() const;

public:

    /**
     * Default constructor for initializing class.
     * 
     * @param fieldPath path to field file image which will be overdrawn.
     */
    Draw(std::string fieldPath);

    /**
     * Return an image representing current game situation.
     * @param outputDrawing output matrix that will be populated with drawing.
     */
    void getGameDraw(cv::Mat& outputDrawing);

    /**
     * Return perspective corrected table image
     * @param corners a vector of point corners from top left in clockwise order.
     */
    cv::Mat computePrespective(const std::vector<cv::Point>& corners);

    /**
     * Returns a new image composed by background image and the overlapping image
     * @param background background image
     * @param overlapping the image that will be put over the background
     * @param position coordinates where the overlapping image will be placed with respect to the background image
     */
    cv::Mat drawOver(const cv::Mat& background, const cv::Mat& overlapping, const cv::Point2i position) const;

};
#endif
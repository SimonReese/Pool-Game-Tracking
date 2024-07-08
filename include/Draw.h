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

    // Path to base field drew image
    std::string fieldPath;

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
     * Set current frame as reference for game status.
     * The frame will be copied.
     * @param currentFrame reference to the frame to consider, which will be copied.
     */
    void setCurrentFrame(const cv::Mat& currentFrame);

    /**
     * Return an image representing current game situation.
     * @param outputDrawing output matrix that will be populated with drawing.
     */
    void getGameDraw(cv::Mat& outputDrawing) const;

    /**
     * Return perspective corrected table image
     * @param corners a vector of point corners from top left in clockwise order.
     */
    cv::Mat correctPrespective(const std::vector<cv::Point>& corners) const;

    /**
     * Returns a new image composed by background image and the overlapping image
     * @param background background image
     * @param overlapping the image that will be put over the background
     * @param position coordinates where the overlapping image will be placed with respect to the background image
     */
    cv::Mat drawOver(const cv::Mat& background, const cv::Mat& overlapping, const cv::Point2i position) const;

};
#endif
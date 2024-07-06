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

};
#endif
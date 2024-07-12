/**
 * @author Simone Peraro
 * TODO: who will maintain this file?
 */

#ifndef TABLESEGMENTER_H
#define TABLESEGMENTER_H

#include <vector>

#include <opencv2/core.hpp>

class TableSegmenter{

    private:


    public:

    /**
     * Returns the playing field corners
     * 
     * Returns a vector containing the 4 points identifying the 4 corners of the 
     * playing field sorted as: top_left, top_right, bottom_right, bottom_left
     * 
     * @param approximate_field_lines
     */
    std::vector<cv::Point2i> findFieldCorners();

};

#endif
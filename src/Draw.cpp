#include "Draw.h"


cv::Mat Draw::detectTableCorners() const{
    
}

Draw::Draw(std::string fieldPath)
    : fieldPath{fieldPath}{
}

void Draw::setCurrentFrame(const cv::Mat &currentFrame){
    currentFrame.copyTo(this->currentFrame);
}

void Draw::getGameDraw(cv::Mat &outputDrawing) const{
    // Currently just return the current frame
    this->currentFrame.copyTo(outputDrawing);
}

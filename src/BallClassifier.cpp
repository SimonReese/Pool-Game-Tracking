#include <BallClassifier.h>


BallClassifier::BallClassifier(){}

float BallClassifier::calculateWhitePixelsRatio(const cv::Mat &inputBinaryImage) {
    // Count the number of white pixels in the binary image.
    int whitePixels = cv::countNonZero(inputBinaryImage == WHITE_COLOR);

    // Calculate the ratio of white pixels to the total number of pixels in the image.
    int totalPixels = inputBinaryImage.rows * inputBinaryImage.cols;
    return static_cast<float>(whitePixels) / totalPixels;
}

std::pair<Ball::BallType, float> BallClassifier::preliminaryBallClassifier(const cv::Mat &cutOutImage){

    // Convert cutOutImage from BGR to HLS  
    cv::Mat hlsImage;
    cv::cvtColor(cutOutImage, hlsImage, cv::COLOR_BGR2HLS);

    cv::Mat binaryImage; // binary image resulting from thresholding with inRange function
    cv::inRange(hlsImage, cv::Scalar(H_LOW, L_LOW, S_LOW), cv::Scalar(H_HIGH, L_HIGH, S_HIGH), binaryImage);

    // calculate the ratio of white pixels to the total number of pixels in the image
    float whitePixelsRatio = BallClassifier::calculateWhitePixelsRatio(binaryImage);

    // if the ratio of white pixels to the total number of pixels is greater than the classification threshold,
    // classify the ball as half white and return the white pixels ratio.
    if (whitePixelsRatio > CLASSIFICATION_TH) {

        // ball classified as half (colored stripes)
        return std::pair<Ball::BallType, float>(Ball::BallType::HALF, whitePixelsRatio);
    } else {
        // ball classified as full of one color
        return std::pair<Ball::BallType, float>(Ball::BallType::FULL, whitePixelsRatio);
    }
}

std::vector<Ball> BallClassifier::classify(const std::vector<Ball> ballsToClassify, const cv::Mat fullGameImage){

    if(ballsToClassify.empty()) throw std::invalid_argument("Error: empty list of balls to classify");
    if(fullGameImage.empty()) throw std::invalid_argument("Error: empty game image");

    // Set the list of balls to be classified
    this->ballsVector = ballsToClassify;
    this->fullGameImage = fullGameImage;

    // maxRatio is the ratio to identify the white ball
    float maxRatio = 0.;
    int maxRatioBallIndex = 0;

    // minRatio is the ratio to identify the black ball
    float minRatio = 1.;
    int minRatioBallIndex = 0;

    // given a set of balls in a game the idea is to find the ball with the highest ratio which corresponds to the white ball
    // and the ball with the lowest white pixels ratio is the black ball.
    for(int index = 0; index < this->ballsVector.size(); index++){

        // crop the ball from the full game image and classify it using the preliminary ball classifier
        cv::Mat cutOutImage = this->fullGameImage(this->ballsVector[index].getBoundingBox());
        std::pair<Ball::BallType, float> classificationResult = preliminaryBallClassifier(cutOutImage);
        
        // assigning to the ball the results from the classifier
        ballsVector[index].setBallType(classificationResult.first);
        ballsVector[index].setWhiteRatio(classificationResult.second);

        // updating maxRatio and minRatio to find the white and black balls.
        if(classificationResult.second > maxRatio){

            maxRatio = classificationResult.second;
            maxRatioBallIndex = index;

        }else if(classificationResult.second < minRatio){

            minRatio = classificationResult.second;
            minRatioBallIndex = index;

        }
    }

    // the ball with the highest white pixels ratio is the white ball
    this->ballsVector[maxRatioBallIndex].setBallType(Ball::BallType::WHITE);

    // the ball with the lowest white pixels ratio is the black ball
    this->ballsVector[minRatioBallIndex].setBallType(Ball::BallType::BLACK);

    return this->ballsVector;
}



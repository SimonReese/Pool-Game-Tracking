#include <BallClassifier.h>

BallClassifier::BallClassifier(){  
}

float BallClassifier::calculateWhitePixelsRatio(const cv::Mat &inputBinaryImage) {
    // Count the number of white pixels in the binary image.
    int whitePixels = cv::countNonZero(inputBinaryImage == 255);

    // Calculate the ratio of white pixels to the total number of pixels in the image.
    int totalPixels = inputBinaryImage.rows * inputBinaryImage.cols;
    return static_cast<float>(whitePixels) / totalPixels;
}

std::pair<Ball::BallType, float> BallClassifier::preliminaryBallClassifier(cv::Mat &cutOutImage){

    // Convert BGR to HLS  
    cv::Mat hlsImage;
    cv::cvtColor(cutOutImage, hlsImage, cv::COLOR_BGR2HLS);

    cv::Mat binaryImage; // binary image resulting from thresholding
    cv::inRange(hlsImage, cv::Scalar(H_LOW, L_LOW, S_LOW), cv::Scalar(H_HIGH, L_HIGH, S_HIGH), binaryImage);

    // calculate the ratio of white pixels to the total number of pixels in the image
    float whitePixelsRatio = calculateWhitePixelsRatio(binaryImage);

    // if the ratio of white pixels to the total number of pixels is greater than the classification threshold,
    // classify the ball as half white and update the white pixels ratio in the Ball object.
    if (whitePixelsRatio > CLASSIFICATION_TH) {
        
        return std::pair<Ball::BallType, float>(Ball::BallType::HALF, whitePixelsRatio);
    } else {

        return std::pair<Ball::BallType, float>(Ball::BallType::FULL, whitePixelsRatio);
    }
}

void BallClassifier::classify(std::vector<Ball> &ballsSet, const cv::Mat &fullGameImage){

    // maxRatio is the ratio to identify the white ball
    float maxRatio = 0.;
    int maxRatioBallIndex = 0;

    // minRatio is the ratio to identify the black ball
    float minRatio = 1.;
    int minRatioBallIndex = 0;

    // given a set of balls in a game the idea is to find the ball with the highest ratio which corresponds to the white ball
    // and the ball with the lowest white pixels ratio is the black ball.
    for(int index = 0; index < ballsSet.size(); index++){

        // crop the ball from the full game image and classify it using the preliminary ball classifier
        cv::Mat cutOutImage = fullGameImage(ballsSet[index].getBoundingBox());
        std::pair<Ball::BallType, float> classificationResult = preliminaryBallClassifier(cutOutImage);

        ballsSet[index].setBallType(classificationResult.first);

        if(classificationResult.second > maxRatio){
            maxRatio = classificationResult.second;
            maxRatioBallIndex = index;
        }else if(classificationResult.second < minRatio){
            minRatio = classificationResult.second;
            minRatioBallIndex = index;
        }
    }

    // the ball with the highest white pixels ratio is the white ball
    ballsSet[maxRatioBallIndex].setBallType(Ball::BallType::WHITE);

    // the ball with the lowest white pixels ratio is the black ball
    ballsSet[minRatioBallIndex].setBallType(Ball::BallType::BLACK);
}



void showHlsChannelsandBinary(const std::vector<cv::Mat> &channelsHlsImage, const cv::Mat &binaryImage, std::string windowName){

    int k = 13; // magnification factor

    cv::Mat L = channelsHlsImage[1].clone();
    cv::Mat S = channelsHlsImage[2].clone();

    cv::Mat binaryClone = binaryImage.clone();

    // cutout images are very small, we need to enlarge them
    cv::resize(S, S, cv::Size(S.rows*k, S.cols*k));
    cv::resize(L, L, cv::Size(L.rows*k, L.cols*k));
    cv::resize(binaryClone, binaryClone, cv::Size(binaryClone.rows*k, binaryClone.cols*k));

    // cv::imshow(windowName + "_S", S);
    // cv::imshow(windowName + "_L", L);
    cv::imshow(windowName + "_bin", binaryClone);

}

int evaluateBallsSet(const std::string datasetFolder, const std::string gameFolder, const std::string ballClassFolder){

    std::vector<cv::String> cutOutList = listFrames("../" + datasetFolder, gameFolder, ballClassFolder);

    Ball::BallType setClass;
    int wrongClassCounter = 0;

    (ballClassFolder == "full") ? setClass = Ball::BallType::FULL
    : (ballClassFolder == "half") ? setClass = Ball::BallType::HALF
    : throw std::invalid_argument("Invalid ball class folder: " + ballClassFolder);


    for (cv::String cutOutName : cutOutList){

        std::string imgPath = "../" + datasetFolder + "/" + gameFolder + "/" + ballClassFolder + "/" + cutOutName;
        cv::Mat cutout = cv::imread( imgPath.c_str() );
        std::pair<Ball::BallType, float> classificationResult = BallClassifier::preliminaryBallClassifier(cutout);

        (classificationResult.first != setClass) ? wrongClassCounter++ : 0;

    }

    return wrongClassCounter;

}

void evaluteGames(std::string datasetFolder){

    std::string datasetPath = "../" + datasetFolder;

    std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);

    int wrongClassifiedFull;
    int wrongClassifiedHalf;

    int wrongClassifiedSum = 0;

    for(cv::String game : gameFolders){

        std::string ballClassFolder = "full";
        wrongClassifiedFull = evaluateBallsSet(datasetFolder, game, ballClassFolder);   

        std::cout << "================" << std::endl;

        ballClassFolder = "half";
        wrongClassifiedHalf = evaluateBallsSet(datasetFolder, game, ballClassFolder); 

        std::cout << "Wrongly classified full balls in " << game << ": " << wrongClassifiedFull << std::endl;
        std::cout << "Wrongly classified half balls in " << game << ": " << wrongClassifiedHalf << "\n"<< std::endl;
        //std::cout << "================" << std::endl;
        // std::cout << "Total wrong classified balls in " << gameFolder << ": " << wrongClassifiedFull + wrongClassifiedHalf << std::endl;

        wrongClassifiedSum += wrongClassifiedFull + wrongClassifiedHalf;

        // cv::waitKey(0);
    }


    std::cout << "Total wrong classified balls in all games: " << wrongClassifiedSum << std::endl;
    
}



void saveTofile(const cv::Mat &inputImage, std::string imageName, std::string outputFolder){

    // Define the root folder
    std::string rootFolderPath = "../" + CUTOUT_DIR;

    // Check if the folder exists, if not, create its
    if (!cv::utils::fs::exists(rootFolderPath)) {
        if (cv::utils::fs::createDirectory(rootFolderPath)) {
            std::cout << "Directory created successfully: " << rootFolderPath << std::endl;
        } else {
            std::cout << "Failed to create directory: " << rootFolderPath << std::endl;
            // return -1;
        }
    }

    // Check if the folder exists, if not, create its
    if (!cv::utils::fs::exists(outputFolder)) {
        if (cv::utils::fs::createDirectory(outputFolder)) {
            std::cout << "Directory created successfully: " << outputFolder << std::endl;
        } else {
            std::cout << "Failed to create directory: " << outputFolder << std::endl;
            // return -1;
        }
    }

    // Define the output file path
    std::string outputPath = outputFolder + imageName + ".png";

    // Save the inputImage
    if (cv::imwrite(outputPath, inputImage)) {
        std::cout << "Image saved successfully to " << outputPath << std::endl;
    } else {
        std::cout << "Failed to save the image" << std::endl;
    }
}

void cutOutBalls(const cv::Mat &inputImage, const std::vector<Ball>& balls, const std::string &gameFolder){

    int i=0;
    for (Ball ball : balls) {

        cv::Mat boundingBoxCutOut = inputImage(ball.getBoundingBox());

        // maybe for balls better to have a tuple for center and a variable just for the radius
        cv::Mat circleMask = cv::Mat::zeros(boundingBoxCutOut.size(), CV_8UC1);

        cv::Point2i center(circleMask.cols /2, circleMask.rows /2);
        circle(circleMask, center , 9, cv::Scalar(255), -1);
        // Copy the original ROI to the ball cutout, but only where the mask is white (the ball is present)
        cv::Mat circleCutOut;

        // cv::bitwise_and(boundingBoxCutOut, boundingBoxCutOut, circleCutOut, circleMask);
        boundingBoxCutOut.copyTo(circleCutOut, circleMask); 

        // std::cout << "Ball center: " << ball.getBallCenterInBoundingBox() << boundingBoxCutOut.rows << std::endl;

        saveTofile(circleCutOut, "ball_cutout" + std::to_string(i), "../" + CUTOUT_DIR + "/" + gameFolder + "/" ); 
        // cv:imshow("Ball Detection" + std::to_string(i), circleMask);
        i++;
    }
    

}
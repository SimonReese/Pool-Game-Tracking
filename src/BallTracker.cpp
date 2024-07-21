/**
 * @author Federico Adami
 */

#include <opencv2/tracking.hpp>
#include <stdexcept>
#include "BallTracker.h"


BallTracker::BallTracker(const cv::Mat firstFrame, const std::vector<Ball> gameBalls){

        // the tracker cannot work if no balls are detected
        if (gameBalls.empty()) throw std::invalid_argument("Error: balls vector is empty");

        // need the first frame of the video to start the tracking of the balls
        if(firstFrame.empty()) throw std::invalid_argument("Error: first video frame is empty");


        //first frae of the video
        this->firstFrame = firstFrame;

        // vector of balls in the game initialization
        this->stillBalls = gameBalls;

        // comppute the balls that are expected to move at first i.e the white ball
        findWhiteBall();
        
        // a tracker is assigned to each ball expected to move 
        this->ballTrackers = BallTracker::createTrackers(firstFrame);
}


std::vector<cv::Ptr<cv::Tracker>> BallTracker::createTrackers(const cv::Mat &frame) const{

    // cannot create a tracker if the frame provided is empty
    if(frame.empty()) throw std::invalid_argument("Error: request initialization on empty frame");
    
    // vector that will contain all the trackers, one per ball
    std::vector<cv::Ptr<cv::Tracker>> trackers; 
    

    // initialization of each tracker using the bounding box of the ball, and push it into the vector of trackers
    for (int i = 0; i < this->movingBalls.size(); i++){

        // creation of the tracker
        cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create();
        // initialization of the tracker using the bounding box of the ball 
        tr->init(frame, this->movingBalls[i].getBoundingBox()); 
        
        // tracker pushed in the vector of trackers
        trackers.push_back(tr);
    }

    return trackers;
}


void BallTracker::updateBallsCenterAndBoundingBox(const std::vector<cv::Rect> &newBoundingBoxes){
    
    for (int i = 0; i < newBoundingBoxes.size(); i++){
        // new bounding box is assigned to the corresponding ball
        this->movingBalls[i].setBoundingBox(newBoundingBoxes[i]);
        //updates the coordinates of the center of the circle that represent the ball
        this->movingBalls[i].setBallCenter(cv::Point(newBoundingBoxes[i].x + newBoundingBoxes[i].width/2, newBoundingBoxes[i].y + newBoundingBoxes[i].height/2));    
    } 
}


bool BallTracker::compareWhiteRatio(Ball first, Ball second){ 
    // white ratio comparison between the two balls
    return (first.getWhiteRatio() > second.getWhiteRatio()); 
}


/**
 * used for the initialization of the @param movingBalls vector 
 * because from the game we know that the ball that moves first is the white ball.
 * The balls are ordered according to their white ratio and the first 3 with the 
 * highest white ratio are selected
*/
void BallTracker::findWhiteBall( int numTrackedBalls ){

    // avoid creating vector of moving balls that has more balls than the balls detected in the field
    if(numTrackedBalls > stillBalls.size()) numTrackedBalls = stillBalls.size();

    // sort the balls based on their white ratio in descending order
    std::sort(this->stillBalls.begin(),this->stillBalls.end(), compareWhiteRatio);

    // select the balls with highest white ratio and intert them inside movingBalls vector
    std::copy(this->stillBalls.begin(), this->stillBalls.begin() + numTrackedBalls, std::back_inserter(this->movingBalls));  

    // remove the moving balls from the stillBalls vector
    this->stillBalls.erase(this->stillBalls.begin(),this->stillBalls.begin() + numTrackedBalls);
}


float BallTracker::ballsDistance(Ball first, Ball second){
    // euclidean distance between the centers of the two balls
    return sqrt(pow(first.getBallCenter().x - second.getBallCenter().x, 2) + pow(first.getBallCenter().y - second.getBallCenter().y, 2));
}


/**
 * a new moving ball is detected if it is a still ball with distance from
 * a moving ball below the collision distance. If so the two balls are colliding
 * causing the still ball to move. 
*/
void BallTracker::updateMovingBalls(const cv::Mat& frame, float collosionDistance){

    // cannot create a tracker if the frame provided is empty
    if(frame.empty()) throw std::invalid_argument("Error: request initialization on empty frame");

    for(Ball moving : this->movingBalls){ 
        for(int i = 0; i < this->stillBalls.size(); i++){

            // check the distance between each moving ball and each still ball
            if(BallTracker::ballsDistance(moving, this->stillBalls[i]) < collosionDistance){

                
                this->movingBalls.push_back(this->stillBalls[i]); // a new moving ball added to the movingBalls vector

                this->stillBalls.erase(this->stillBalls.begin() + i); // remove still ball from the stillBalls vector
                

                // creation and initialization of the tracker using the bounding box of the new moving ball
                cv::Ptr<cv::Tracker> tr = cv::TrackerCSRT::create(); 
                tr->init(this->firstFrame, this->movingBalls.back().getBoundingBox()); 
                this->ballTrackers.push_back(tr);
            }

        }
    }

}


bool BallTracker::update(const cv::Mat &currentFrame, std::vector<Ball> &ballsToUpdate){
    
    // cannot perform tracking if the frame provided is empty
    if(currentFrame.empty()) throw std::invalid_argument("Error: requeste update on empty frame");
    std::vector<cv::Rect> BoundingBoxes;
    
    //update the tracking result
    for (int h = 0; h < this->ballTrackers.size(); h++){
        
        // update the bounding box of the ball using the tracker and the current frame
        cv::Rect BBox;
        bool ballFound = this->ballTrackers[h]->update(currentFrame, BBox);
        // if ball is not found then tracking is lost
        if(!ballFound){
            // tracking failed
            return false;
        }

        // add the updated bounding box to the vector of bounding boxes
        BoundingBoxes.push_back(BBox);
    }

    // update balls center and their bounding boxes
    BallTracker::updateBallsCenterAndBoundingBox(BoundingBoxes);

    // update the vector of moving balls
    BallTracker::updateMovingBalls(currentFrame);

    // combine all balls (moving and still) into one vector
    std::vector<Ball> allBalls;
    allBalls.insert( allBalls.end(), this->movingBalls.begin(), this->movingBalls.end() );
    allBalls.insert( allBalls.end(), this->stillBalls.begin(), this->stillBalls.end() );  

    // update the vector that keeps the balls state, with the new vector of all balls
    ballsToUpdate.clear();
    std::copy(allBalls.begin(), allBalls.end(), std::back_inserter(ballsToUpdate));

    // all the balls were tracked successfully
    return true;
}



 #include <opencv2/core/utility.hpp>
 #include <opencv2/tracking.hpp>
 #include <opencv2/videoio.hpp>
 #include <opencv2/highgui.hpp>
 #include <iostream>
 #include <cstring>
 #include <ctime>
 #include "C:\workspace\opencv\source\opencv_contrib-4.x\modules\tracking\samples\samples_utility.hpp"
 
 using namespace std;
 using namespace cv;
 
 int main( int argc, char** argv ){
  // set the default tracking algorithm
  std::string trackingAlg = "CSRT";
 
  // create the tracker
  legacy::MultiTracker trackers;

  // container of the tracked objects
  vector<Rect2d> objects;
 
  // set input video
  std::string video = "C:/Users/Alebo/Desktop/Dataset-20240529T161618Z-001/game1_clip3/game1_clip3.mp4";
  VideoCapture cap(video);
 
  Mat frame;

  //path to where the output videos will be written
  cv::VideoWriter output("output.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));
 
  cv::VideoWriter output2("output2.avi", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), 30, cv::Size(static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH)),static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT))));

  // get bounding box
  // it is done manually using the mouse --> will not be needed once the bounding boxes are found automatically
  cap >> frame;
  vector<Rect> ROIs;
  selectROIs("tracker",frame,ROIs,true);
 
  //quit when the tracked object(s) is not provided
  if(ROIs.size()<1)
  return 0;

  Mat trajectories(frame.size().height,frame.size().width,frame.type()); // Mat for drawing the trajectories of the tracked balls

  //find rough borders of the playing field to later implement a very poor way of detecting if a ball has left the playing field
  Mat find_borders_grey(frame.size().height,frame.size().width, CV_8U);

  cv::cvtColor(frame,find_borders_grey,COLOR_BGR2GRAY); 

  cv::GaussianBlur(find_borders_grey,find_borders_grey,cv::Size(3,3),0,0);

  cv::Mat edges;

  cv::Canny(find_borders_grey,edges,127,127);
  
  cv::Mat boundaries(frame.size().height,frame.size().width,frame.type());

  
  vector<cv::Vec2f> lines; // will hold the results of the detection
  HoughLines(edges, lines, 1, CV_PI/360, 150, 0, 0); // runs the actual detection
  vector<cv::Point2f> pts; //will contain only 2 lines points

  for(int i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0], theta = lines[i][1];  
    for (int j = i+1; j < lines.size(); j++){
      if((lines[j][0] <= rho+100.0) && (lines[j][0] >= rho-100.0)){
        lines.erase(lines.begin()+j);
        j--;
      }
    }
 }

    for( size_t i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0], theta = lines[i][1];  
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    line( boundaries, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
 }


 
  // initialize the tracker
  std::vector<Ptr<legacy::Tracker> > algorithms;
  vector<Point> current_center;
  for (size_t i = 0; i < ROIs.size(); i++){
    current_center.push_back(Point(( ROIs[i].x + ROIs[i].width/2), (ROIs[i].y + ROIs[i].height/2))); //center of the bounding box
    algorithms.push_back(createTrackerByName_legacy(trackingAlg)); //tracking algorithm used
    objects.push_back(ROIs[i]); // insert in objects vector the bounding box
  }
 
  trackers.add(algorithms,frame,objects);
  
 
  // do the tracking
  printf("Start the tracking process, press ESC to quit.\n");
  for ( ;; ){
  // get frame from the video
  cap >> frame;
  
  // stop the program if no more images
  if(frame.rows==0 || frame.cols==0)
  break;
 
  //update the tracking result
  trackers.update(frame);
 
  // draw the tracked object and implements a very poor way of detecting if a ball has left the playing field --> needs improvement for the balls that leave the field
  vector<int> indexes_to_remove;
  indexes_to_remove.clear();
  for(unsigned i=0;i<trackers.getObjects().size();i++){
    rectangle( frame, trackers.getObjects()[i], Scalar( 255, 0, 0 ), 2, 1 );
    Point new_center = Point(static_cast<int>( trackers.getObjects()[i].x + trackers.getObjects()[i].width/2), static_cast<int>(trackers.getObjects()[i].y + trackers.getObjects()[i].height/2));
    for (int rows_check = static_cast<int>(trackers.getObjects()[i].height/4); rows_check <= trackers.getObjects()[i].height/2; rows_check++){
      for (int cols_check = static_cast<int>(trackers.getObjects()[i].width/4); cols_check <= trackers.getObjects()[i].width/2; cols_check++){
        if(boundaries.at<Vec3b>(new_center.y + rows_check,new_center.x + cols_check)[2] == 255 || boundaries.at<Vec3b>(new_center.y + rows_check,new_center.x - cols_check)[2] == 255){
          indexes_to_remove.push_back(i);
        }else if(boundaries.at<Vec3b>(new_center.y - rows_check,new_center.x + cols_check)[2] == 255 || boundaries.at<Vec3b>(new_center.y - rows_check,new_center.x - cols_check)[2] == 255){
          indexes_to_remove.push_back(i);
        }
      }  
    }
    if(new_center.x == 0 && new_center.y == 0){ // checks if the ball has left the field --> in that case it stops drawing the trajectory for that ball

    }else{
      line(trajectories, current_center[i], new_center, Scalar(255, 0, 0), 1, LINE_8); //draws the balls trajectories
    }
    current_center[i]=new_center; //updates the centers of the different bounding boxes
  }

  // updates the objects to track by removing the balls that left the field from the instance of the tracker
  if(indexes_to_remove.size() > 0){
    vector<Rect2d> objects2 = trackers.getObjects();
    for (size_t i = (indexes_to_remove.size()-1); i >= 0; i--){
      objects2.erase(objects2.begin()+indexes_to_remove[i]);
      current_center.erase(current_center.begin()+indexes_to_remove[i]);
    }

    trackers.clear();
    algorithms.clear();

    for (size_t i = 0; i < objects2.size(); i++){
      algorithms.push_back(createTrackerByName_legacy(trackingAlg));
    }
    
    trackers.add(algorithms,frame,objects2);
  }

  //draws the boundaries on every new frame that is processed to make it visible --> not really needed, it's more for visual purpose
  for( size_t i = 0; i < lines.size(); i++ ){
    float rho = lines[i][0], theta = lines[i][1];  
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    line( frame, pt1, pt2, Scalar(0,0,255), 1, LINE_AA);
 }

  // show image with the tracked object
  imshow("tracker",frame);
  imshow("trajectories",trajectories);

  //write the processed frame in the video
  output.write(frame);
  output2.write(trajectories);
 
  //quit on ESC button
  if(waitKey(1)==27)break;
  }

    output.release();
    output2.release();
	  cap.release();

	// Destroy all windows
	cv::destroyAllWindows();

 }

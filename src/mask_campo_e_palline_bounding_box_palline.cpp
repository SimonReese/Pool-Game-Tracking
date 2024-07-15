#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>


#include "BallsDetection.h"
#include "BallClassifier.h"
#include "FieldGeometryAndMask.h"
#include "Ball.h"
#include "FilesystemUtils.h"

#define X_VALUE 11
#define Y_VALUE 11


using namespace std;

int value = 0;


std::string OUTPUT_DATASET = "../res/predictions";
std::string OUTPUT_CLIP = "/game1_clip1";


const std::string FRAMES_DIR = "frames";
const std::string BOX_DIR = "bounding_boxes";
const std::string MASK_DIR = "masks";


void my_HSV_callback2(int event, int x, int y, int flags, void* userdata, std::string bboxFileName, std::string maskFileName, std::string gameFolder){
    if(event == cv::EVENT_LBUTTONDOWN){
        cv::Mat image2 = *(cv::Mat*) userdata;
        cv::Mat no_mod_image = image2.clone();
        cv::Mat image; 

        cv::cvtColor(image2,image,cv::COLOR_BGR2HSV);

        cv::Mat less_blur_image = image.clone();

        cv::Mat no_blur_image = image.clone();

        cv::GaussianBlur(less_blur_image,less_blur_image,cv::Size(3,3),0,0); // used to find balls later on

        cv::GaussianBlur(image,image,cv::Size(7,7),0,0); // used to find field mask
        
        cv::Vec3b mean_color = fieldMeanColor(image,11);

        cv::Mat filled_field_contour = computeFieldMask(image,mean_color);
        
        cv::Mat approximate_field_lines = findFieldLines(filled_field_contour);

        vector<cv::Point2i> sorted_corners = findFieldCorners(approximate_field_lines);

        vector<cv::Point> boundaries_contours_poly = defineBoundingPolygon(sorted_corners,approximate_field_lines);

        cv::Mat only_table_image = no_blur_image.clone();

        //removes everything from the initial image apart from the pixels defined by the mask that segments the field
        for (int i = 0; i < image.size().height; i++){
            for (int j = 0; j < image.size().width; j++){
                if( filled_field_contour.at<uchar>(i,j) == 0){
                    only_table_image.at<cv::Vec3b>(i,j) = cv::Vec3b(0,0,0);
                }
            }
        }

        // cv::imshow("filled_field_contour", filled_field_contour);

        vector<Ball> balls = findBalls(only_table_image, filled_field_contour, boundaries_contours_poly, sorted_corners);

        drawBallsHSVChannels(balls, no_mod_image);

        cv::Mat field_and_balls_mask = drawBallsOnFieldMask(filled_field_contour,balls);

        vector<cv::Rect> boundRect = findBoundingRectangles(field_and_balls_mask);

        /*updates the bounding box value of each ball by assigning the bounding box that has center closest to the center of the circle that defines the ball*/
        int offset_thres = 1;
        for (int h = 0; h < boundRect.size(); h++){
            for (int b = 0; b < balls.size(); b++){
                if((boundRect[h].x+boundRect[h].width/2) >= balls[b].getBallPosition()[0]-offset_thres && (boundRect[h].x+boundRect[h].width/2) <= balls[b].getBallPosition()[0]+offset_thres){
                    if((boundRect[h].y+boundRect[h].height/2) >= balls[b].getBallPosition()[1]-offset_thres && (boundRect[h].y+boundRect[h].height/2) <= balls[b].getBallPosition()[1]+offset_thres){
                        balls[b].setBoundingBox(boundRect[h]);
                    }
                }
            }   
        }

        // -> here try the algorgthmt to classify balls

        // find bounding boxes of all circles and draws them
        
        drawBoundingBoxesHSVChannels(balls, no_mod_image);
        cutOutBalls(no_mod_image, balls, gameFolder);



        // writeBboxToFile(bboxFileName, balls);

        // cv::namedWindow("mask display");
        // cv::imshow("mask display", mask);
        // cv::namedWindow("edges display");
        // cv::imshow("edges display", edges);
        // cv::namedWindow("field display");
        // cv::imshow("field display", field_contour);
        // cv::namedWindow("cropped field display");
        // cv::imshow("cropped field display", exit1);
        // cv::namedWindow("cropped field display");
        // cv::cvtColor(only_table_image,only_table_image,cv::COLOR_HSV2BGR);
        // cv::imshow("cropped field display", only_table_image);

        // cv::cvtColor(image,image,cv::COLOR_HSV2BGR);
        

        // cv::imshow("boh",exit1);

        // cv::imwrite(maskFileName,field_and_balls_mask); //salva mask del campo e palline --> background = 0 campo = 255 palline = 127


    }
}

void my_HSV_callback2(int event, int x, int y, int flags, void* userdata){
    my_HSV_callback2(event, x, y, flags, userdata, "bounding_box_output.txt", "maschera.jpeg", "okok");
}

int main(int argc, char* argv[])
{
    if (argc < 3){
        std::cout << "Please enter dataset path (usually in (root)/res/Dataset/) and predictions path (usually in (root)/res/predictions/)" << std::endl;
    }
    std::string datasetPath = argv[1];
    // std::string predictionsPath = argv[2];
    std::vector<cv::String> gameFolders = listGameDirectories(datasetPath);


//the following lines are to extract singular bboxes from the dataset to cutout the balls
    //given dummy value just to use function my_HSV_callback2()
    std::string bboxFileName = "none"; // NOT USED
    std::string maskFileName = "none"; // NOT USED

    //using gamefolders computed before with listGameDirectories()

    //ANNOTATION: function my_HSV_callback2() has imwrite error in game2_clip2 and not running in game3_clip2
    /* for(cv::String game : gameFolders)
    for(int i = 6; i < gameFolders.size(); i++){

        std::string game = gameFolders[i];

        //game = "game4_clip2";

        // Open frame
        std::string fullFramePath = datasetPath + "/" + game + "/" + FRAMES_DIR + "/" + "frame_first.png";
        cv::Mat image = cv::imread(fullFramePath);

        my_HSV_callback2(cv::EVENT_LBUTTONDOWN, 0, 0, 0, &image, bboxFileName, maskFileName, game);
        
        //break; //used to test the function with a single frame
    } */

    std::string datasetFolder = "balls_cutout";
    
    evaluteGames(datasetFolder);

    cv::waitKey(0);

    return 0;
}

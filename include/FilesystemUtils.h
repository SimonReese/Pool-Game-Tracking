#ifndef FILESYSTEM_UTILS_H
#define FILESYSTEM_UTILS_H

#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
#include <Ball.h>
#include <filesystem>

/**
 * Testing function to greet user
 * @param username name of the user to greet
*/
void helloFunction(std::string username = "User");

std::vector<cv::String> listGameDirectories(std::string datasetPath);

std::vector<cv::String> listFrames(std::string datasetPath, std::string gamePath, std::string frameFolderName);

/**
 * Sort tuple for key descending order
 */
bool sortTupleKeysDescending(std::tuple<double, int, bool>& first, std::tuple<double, int, bool>& second);

#endif

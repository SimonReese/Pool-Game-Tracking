#include <FilesystemUtils.h>

std::vector<cv::String> listGameDirectories(std::string datasetPath){
    std::vector<cv::String> files;
    std::vector<cv::String> gameFolders;
    cv::utils::fs::glob_relative(datasetPath, "", files, false, true);
    for (cv::String file : files){
        // Check if it is a directory
        if (cv::utils::fs::isDirectory(datasetPath + "/" + file)){
            gameFolders.push_back(file);
        }
    }

    return gameFolders;
}

std::vector<cv::String> listFrames(std::string datasetPath, std::string gamePath, std::string frameFolderName){
    std::vector<cv::String> frameFullNames;
    std::string fullPath = datasetPath + "/" + gamePath + "/" + frameFolderName;
    cv::utils::fs::glob_relative(fullPath, "", frameFullNames);
    return frameFullNames;
}



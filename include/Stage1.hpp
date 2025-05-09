#ifndef STAGE1_HPP
#define STAGE1_HPP

#include <opencv2/opencv.hpp>
#include "utils.hpp"

class Stage1 {
public:
    Stage1();
    void run();
private:
    void onTrackbar(int, void*);
    cv::Point getContourCenter(const std::vector<cv::Point>& contour);
    
    // Variables for HSV controls
    int H_MIN, S_MIN, V_MIN;
    int H_MAX, S_MAX, V_MAX;
    const std::string WINDOW_NAME_CONTROLS = "HSV Controls - Stage 1";
};

#endif // STAGE1_HPP
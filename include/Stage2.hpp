#ifndef STAGE2_HPP
#define STAGE2_HPP

#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <vector>

class Stage2 {
public:
    Stage2();
    void run();

private:
    // Color range parameters for friend/foe detection
    struct ColorRange {
        std::string name;
        int h_min, s_min, v_min;
        int h_max, s_max, v_max;
        cv::Scalar bgr_color;
    };

    // Structure to hold target information
    struct TargetInfo {
        std::vector<cv::Point> contour;
        cv::Point center;
        cv::Rect box;
        double area;
        int colorId;
        std::string label;
    };

    cv::Point getContourCenter(const std::vector<cv::Point>& contour);
    bool compareTargetInfo(const TargetInfo& a, const TargetInfo& b);

    ColorRange foe_params;   // Enemy color parameters
    ColorRange friend_params;  // Friendly color parameters
};

#endif // STAGE2_HPP
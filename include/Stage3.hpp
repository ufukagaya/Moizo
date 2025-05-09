#ifndef STAGE3_HPP
#define STAGE3_HPP

#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "utils.hpp"
#include <string>
#include <vector>

class Stage3 {
public:
    Stage3();
    void run();

private:
    struct Stage3Engagement {
        int board_side;
        int required_color_id;
        int required_shape_id;
        std::string description_text;
    };

    struct TargetObjectInfo {
        cv::Rect box;
        cv::Point center;
        double area;
        int detected_color_id;
        int detected_shape_id;
        std::string combined_label;
        float confidence;
        std::vector<cv::Point> contour_points;
    };

    bool initializeYoloDetector();
    bool loadYoloShapeClasses(const std::string& filename);
    Stage3Engagement generateRandomEngagement();
    int determineDominantColorID(const cv::Mat& frame_hsv, const cv::Rect& roi);
    std::vector<cv::String> getYoloOutputLayerNames();

    cv::dnn::Net yolo_detection_net;
    std::vector<std::string> yolo_shape_classes;
    const float MIN_YOLO_CONFIDENCE = 0.35f;
    const float NMS_THRESHOLD = 0.4f;
    const int YOLO_INPUT_WIDTH = 416;
    const int YOLO_INPUT_HEIGHT = 416;

    std::string yolo_model_weights_path;
    std::string yolo_model_cfg_path;
    std::string yolo_class_names_path;
};

#endif // STAGE3_HPP
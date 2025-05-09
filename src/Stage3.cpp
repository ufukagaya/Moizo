#include "../include/Stage3.hpp"
#include <iostream>
#include <fstream>
#include <random>
#include <chrono>
#include <algorithm>

Stage3::Stage3() {
    yolo_model_weights_path = "yolov4-tiny.weights";
    yolo_model_cfg_path = "yolov4-tiny.cfg";
    yolo_class_names_path = "coco.names";
}

bool Stage3::initializeYoloDetector() {
    std::cout << "Initializing YOLO...\n";
    
    // Check if files exist
    std::ifstream cfg_file(yolo_model_cfg_path);
    std::ifstream weights_file(yolo_model_weights_path);
    
    if (!cfg_file.good() || !weights_file.good()) {
        std::cerr << "ERROR: YOLO model files not found!\n";
        return false;
    }
    
    cfg_file.close();
    weights_file.close();

    yolo_detection_net = cv::dnn::readNetFromDarknet(yolo_model_cfg_path, yolo_model_weights_path);
    if (yolo_detection_net.empty()) {
        std::cerr << "ERROR: Could not load YOLO model!\n";
        return false;
    }

    // Force CPU usage
    yolo_detection_net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
    yolo_detection_net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    return loadYoloShapeClasses(yolo_class_names_path);
}

bool Stage3::loadYoloShapeClasses(const std::string& filename) {
    std::ifstream ifs(filename);
    if (!ifs.is_open()) {
        std::cerr << "ERROR: Could not open class names file: " << filename << std::endl;
        return false;
    }

    std::string line;
    while (std::getline(ifs, line)) {
        yolo_shape_classes.push_back(line);
    }

    if (yolo_shape_classes.empty()) {
        std::cerr << "WARNING: No classes loaded!" << std::endl;
        return false;
    }

    std::cout << "Loaded shape classes:\n";
    for (const auto& className : yolo_shape_classes) {
        std::cout << "- " << className << "\n";
    }
    return true;
}

std::vector<cv::String> Stage3::getYoloOutputLayerNames() {
    std::vector<cv::String> names;
    std::vector<int> outLayers = yolo_detection_net.getUnconnectedOutLayers();
    std::vector<cv::String> layersNames = yolo_detection_net.getLayerNames();
    
    names.resize(outLayers.size());
    for (size_t i = 0; i < outLayers.size(); ++i) {
        names[i] = layersNames[outLayers[i] - 1];
    }
    return names;
}

Stage3::Stage3Engagement Stage3::generateRandomEngagement() {
    static std::mt19937 rng(std::chrono::steady_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> board_dist(0, 1);
    std::uniform_int_distribution<int> shape_dist(0, yolo_shape_classes.size() - 1);

    Stage3Engagement order;
    order.board_side = board_dist(rng);
    order.required_shape_id = shape_dist(rng);
    
    std::string shape_name = yolo_shape_classes[order.required_shape_id];
    order.description_text = (order.board_side == 0 ? "LEFT S." : "RIGHT S.") +
                           std::string(" ") + shape_name;
    return order;
}

void Stage3::run() {
    if (!initializeYoloDetector()) {
        std::cerr << "Stage 3 initialization failed!\n";
        return;
    }

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera!\n";
        return;
    }

    cv::namedWindow("Stage 3 - Live Feed");
    Stage3Engagement current_order = generateRandomEngagement();
    std::cout << "NEW ENGAGEMENT: " << current_order.description_text << std::endl;

    cv::Mat frame, blob;
    TargetObjectInfo locked_target;
    bool is_correctly_locked = false;

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        // Create blob for YOLO
        cv::dnn::blobFromImage(frame, blob, 1/255.0, 
            cv::Size(YOLO_INPUT_WIDTH, YOLO_INPUT_HEIGHT),
            cv::Scalar(0,0,0), true, false);
        
        yolo_detection_net.setInput(blob);
        std::vector<cv::Mat> outs;
        yolo_detection_net.forward(outs, getYoloOutputLayerNames());

        std::vector<int> class_ids;
        std::vector<float> confidences;
        std::vector<cv::Rect> boxes;

        for (const cv::Mat& out : outs) {
            for (int i = 0; i < out.rows; ++i) {
                cv::Mat scores = out.row(i).colRange(5, out.cols);
                cv::Point class_id_point;
                double confidence;
                cv::minMaxLoc(scores, nullptr, &confidence, nullptr, &class_id_point);

                if (confidence > MIN_YOLO_CONFIDENCE) {
                    int centerX = static_cast<int>(out.at<float>(i, 0) * frame.cols);
                    int centerY = static_cast<int>(out.at<float>(i, 1) * frame.rows);
                    int width = static_cast<int>(out.at<float>(i, 2) * frame.cols);
                    int height = static_cast<int>(out.at<float>(i, 3) * frame.rows);
                    int left = centerX - width/2;
                    int top = centerY - height/2;

                    class_ids.push_back(class_id_point.x);
                    confidences.push_back(static_cast<float>(confidence));
                    boxes.push_back(cv::Rect(left, top, width, height));
                }
            }
        }

        std::vector<int> indices;
        cv::dnn::NMSBoxes(boxes, confidences, MIN_YOLO_CONFIDENCE, NMS_THRESHOLD, indices);

        is_correctly_locked = false;
        locked_target = TargetObjectInfo();
        int frame_middle_x = frame.cols / 2;

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            
            // Check frame boundaries
            box.x = std::max(0, std::min(box.x, frame.cols - 1));
            box.y = std::max(0, std::min(box.y, frame.rows - 1));
            box.width = std::min(box.width, frame.cols - box.x);
            box.height = std::min(box.height, frame.rows - box.y);

            int detected_shape_id = class_ids[idx];
            cv::Point center(box.x + box.width/2, box.y + box.height/2);
            bool is_correct_board_side = (current_order.board_side == 0 && center.x < frame_middle_x) ||
                                       (current_order.board_side == 1 && center.x >= frame_middle_x);
            bool is_correct_shape = (detected_shape_id == current_order.required_shape_id);

            if (is_correct_board_side && is_correct_shape) {
                locked_target.box = box;
                locked_target.center = center;
                locked_target.detected_shape_id = detected_shape_id;
                locked_target.confidence = confidences[idx];
                locked_target.combined_label = yolo_shape_classes[detected_shape_id];
                is_correctly_locked = true;
                break;
            }
            
            // If no correct target found and this is the first target
            if (i == 0 && locked_target.confidence == 0.0f) {
                locked_target.box = box;
                locked_target.center = center;
                locked_target.detected_shape_id = detected_shape_id;
                locked_target.confidence = confidences[idx];
                locked_target.combined_label = yolo_shape_classes[detected_shape_id];
            }
        }

        // Center line
        cv::line(frame, cv::Point(frame_middle_x, 0), 
                cv::Point(frame_middle_x, frame.rows), 
                cv::Scalar(100,100,100), 1);

        // Engagement info
        cv::putText(frame, "Engagement: " + current_order.description_text,
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    cv::Scalar(255,255,0), 1);

        // Target box and status message
        if (locked_target.confidence > 0.0f) {
            cv::Scalar box_color = is_correctly_locked ? 
                                 cv::Scalar(0,255,0) : cv::Scalar(0,165,255);
            
            cv::rectangle(frame, locked_target.box, box_color, 2);
            std::string label = locked_target.combined_label + 
                              " (" + std::to_string(static_cast<int>(locked_target.confidence * 100)) + "%)";
            cv::putText(frame, label, 
                       cv::Point(locked_target.box.x, locked_target.box.y - 5),
                       cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color, 1);

            std::string status = is_correctly_locked ? "CORRECT TARGET LOCKED" : "WRONG TARGET LOCKED!";
            cv::putText(frame, status, cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2);
        } else {
            cv::putText(frame, "SEARCHING TARGET...", cv::Point(10, 60),
                       cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(200,200,200), 2);
        }

        cv::imshow("Stage 3 - Live Feed", frame);

        char key = (char)cv::waitKey(30);
        if (key == 27) break;  // ESC
        if (key == ' ') {      // SPACE
            if (is_correctly_locked) {
                std::cout << "FIRE! Correct target (" << locked_target.combined_label 
                         << ") has been eliminated.\n";
                current_order = generateRandomEngagement();
                std::cout << "NEW ENGAGEMENT: " << current_order.description_text << std::endl;
            } else if (locked_target.confidence > 0.0f) {
                std::cout << "WRONG TARGET FIRED AT! Locked: " << locked_target.combined_label
                         << ". Required: " << current_order.description_text << "\n";
                current_order = generateRandomEngagement();
                std::cout << "NEW ENGAGEMENT: " << current_order.description_text << std::endl;
            } else {
                std::cout << "CANNOT FIRE: No target locked.\n";
            }
        }
        if (key == 'n' || key == 'N') {
            current_order = generateRandomEngagement();
            std::cout << "MANUAL NEW ENGAGEMENT: " << current_order.description_text << std::endl;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}
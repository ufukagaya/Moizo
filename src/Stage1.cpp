#include "../include/Stage1.hpp"
#include <iostream>

Stage1::Stage1() {
    // Initial HSV values (e.g., for bright BLUE)
    H_MIN = 90; S_MIN = 100; V_MIN = 100;
    H_MAX = 130; S_MAX = 255; V_MAX = 255;
}

void Stage1::run() {
    std::cout << "Running Stage 1..." << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera!\n";
        return;
    }

    cv::namedWindow("Stage 1 - Original");
    cv::namedWindow("Stage 1 - Mask");
    cv::namedWindow(WINDOW_NAME_CONTROLS);

    // Create trackbars
    cv::createTrackbar("H_MIN", WINDOW_NAME_CONTROLS, &H_MIN, 180, nullptr);
    cv::createTrackbar("S_MIN", WINDOW_NAME_CONTROLS, &S_MIN, 255, nullptr);
    cv::createTrackbar("V_MIN", WINDOW_NAME_CONTROLS, &V_MIN, 255, nullptr);
    cv::createTrackbar("H_MAX", WINDOW_NAME_CONTROLS, &H_MAX, 180, nullptr);
    cv::createTrackbar("S_MAX", WINDOW_NAME_CONTROLS, &S_MAX, 255, nullptr);
    cv::createTrackbar("V_MAX", WINDOW_NAME_CONTROLS, &V_MAX, 255, nullptr);

    cv::Mat frame, hsvFrame, mask;
    bool targetLocked = false;
    cv::Point currentTargetCenter(-1, -1);

    while (true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::inRange(hsvFrame, 
                   cv::Scalar(H_MIN, S_MIN, V_MIN),
                   cv::Scalar(H_MAX, S_MAX, V_MAX), 
                   mask);

        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        cv::imshow("Stage 1 - Mask", mask);

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        targetLocked = false;
        currentTargetCenter = cv::Point(-1,-1);

        if (!contours.empty()) {
            std::sort(contours.begin(), contours.end(), 
                [](const std::vector<cv::Point>& c1, const std::vector<cv::Point>& c2){
                    return cv::contourArea(c1) > cv::contourArea(c2);
                });

            if (cv::contourArea(contours[0]) > 500) {
                cv::Rect boundingBox = cv::boundingRect(contours[0]);
                cv::rectangle(frame, boundingBox, cv::Scalar(0, 255, 0), 2);
                currentTargetCenter = getContourCenter(contours[0]);
                if(currentTargetCenter.x != -1) {
                    cv::circle(frame, currentTargetCenter, 5, cv::Scalar(0, 0, 255), -1);
                    targetLocked = true;
                }
            }
        }

        cv::putText(frame, targetLocked ? "TARGET LOCKED" : "SEARCHING TARGET...",
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    targetLocked ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);
        cv::imshow("Stage 1 - Original", frame);

        char key = (char)cv::waitKey(30);
        if (key == 27) break; // ESC
        if (key == ' ') { // SPACE
            if (targetLocked) 
                std::cout << "STAGE 1: FIRE! Target at (" << currentTargetCenter.x << ", " 
                         << currentTargetCenter.y << ") has been eliminated.\n";
            else 
                std::cout << "STAGE 1: Cannot fire, no target locked.\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

cv::Point Stage1::getContourCenter(const std::vector<cv::Point>& contour) {
    if (contour.empty()) return cv::Point(-1, -1);
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return cv::Point(-1, -1);
    
    int cx = static_cast<int>(m.m10 / m.m00);
    int cy = static_cast<int>(m.m01 / m.m00);
    return cv::Point(cx, cy);
}
#include "../include/Stage2.hpp"
#include <iostream>
#include <algorithm>

Stage2::Stage2() {
    // Initialize enemy and friendly color parameters
    foe_params = {"Foe (RED)", 0, 120, 70, 10, 255, 255, cv::Scalar(0,0,255)};
    friend_params = {"Friend (BLUE/GREEN)", 0, 100, 100, 0, 255, 255, cv::Scalar(0,255,0)};
}

void Stage2::run() {
    std::cout << "Running Stage 2..." << std::endl;
    
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Could not open camera!\n";
        return;
    }

    cv::namedWindow("Stage 2 - Original");
    cv::namedWindow("Stage 2 - Foe Mask");
    cv::namedWindow("Stage 2 - Friend Mask");

    cv::Mat frame, hsvFrame, maskFoe, maskFriend;
    bool foeLocked = false;
    cv::Point currentTargetCenter(-1,-1);
    bool currentTargetIsFoe = false;

    while(true) {
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));

        // Enemy mask (Red)
        cv::inRange(hsvFrame, 
                   cv::Scalar(foe_params.h_min, foe_params.s_min, foe_params.v_min),
                   cv::Scalar(foe_params.h_max, foe_params.s_max, foe_params.v_max), 
                   maskFoe);

        // Friend mask (Blue AND Green)
        cv::Mat maskBlue, maskGreen;
        // Mask for Blue
        cv::inRange(hsvFrame, cv::Scalar(90, 100, 100), cv::Scalar(130, 255, 255), maskBlue);
        // Mask for Green
        cv::inRange(hsvFrame, cv::Scalar(40, 100, 100), cv::Scalar(80, 255, 255), maskGreen);
        // Combine both masks
        maskFriend = maskBlue | maskGreen;

        cv::morphologyEx(maskFoe, maskFoe, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskFoe, maskFoe, cv::MORPH_CLOSE, kernel);
        cv::morphologyEx(maskFriend, maskFriend, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(maskFriend, maskFriend, cv::MORPH_CLOSE, kernel);

        cv::imshow("Stage 2 - Foe Mask", maskFoe);
        cv::imshow("Stage 2 - Friend Mask", maskFriend);

        std::vector<TargetInfo> allTargets;
        std::vector<std::vector<cv::Point>> contoursFoe, contoursFriend;

        // Enemy contours
        cv::findContours(maskFoe, contoursFoe, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for(const auto& c : contoursFoe) {
            if(cv::contourArea(c) > 500) {
                TargetInfo foe_target;
                foe_target.contour = c;
                foe_target.area = cv::contourArea(c);
                foe_target.center = getContourCenter(c);
                foe_target.box = cv::boundingRect(c);
                foe_target.colorId = 0; // 0 for Enemy
                foe_target.label = "ENEMY";
                allTargets.push_back(foe_target);
            }
        }

        // Friend contours
        cv::findContours(maskFriend, contoursFriend, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
        for(const auto& c : contoursFriend) {
            if(cv::contourArea(c) > 500) {
                TargetInfo friend_target;
                friend_target.contour = c;
                friend_target.area = cv::contourArea(c);
                friend_target.center = getContourCenter(c);
                friend_target.box = cv::boundingRect(c);
                friend_target.colorId = 1; // 1 for Friend
                friend_target.label = "FRIEND";
                allTargets.push_back(friend_target);
            }
        }

        // Sort targets by size
        std::sort(allTargets.begin(), allTargets.end(), 
            [](const TargetInfo& a, const TargetInfo& b) { return a.area > b.area; });

        foeLocked = false;
        currentTargetCenter = cv::Point(-1, -1);
        currentTargetIsFoe = false;
        TargetInfo primaryTarget;

        if(!allTargets.empty()) {
            // Look for enemy target first
            auto it_foe = std::find_if(allTargets.begin(), allTargets.end(), 
                [](const TargetInfo& t) { return t.label == "ENEMY"; });

            if(it_foe != allTargets.end()) {
                primaryTarget = *it_foe;
                currentTargetIsFoe = true;
            } else {
                primaryTarget = allTargets[0];
            }

            if(primaryTarget.area > 0) {
                currentTargetCenter = primaryTarget.center;
                cv::Scalar boxColor = currentTargetIsFoe ? foe_params.bgr_color : friend_params.bgr_color;
                cv::rectangle(frame, primaryTarget.box, boxColor, 2);
                cv::circle(frame, currentTargetCenter, 5, boxColor, -1);
                cv::putText(frame, primaryTarget.label, 
                          cv::Point(primaryTarget.box.x, primaryTarget.box.y - 5),
                          cv::FONT_HERSHEY_SIMPLEX, 0.5, boxColor, 2);
                if(currentTargetIsFoe) foeLocked = true;
            }
        }

        // Status message
        std::string status_text;
        cv::Scalar status_color;
        if(currentTargetCenter.x != -1) {
            status_text = currentTargetIsFoe ? "ENEMY LOCKED" : "FRIEND DETECTED";
            status_color = currentTargetIsFoe ? foe_params.bgr_color : friend_params.bgr_color;
        } else {
            status_text = "SEARCHING TARGET...";
            status_color = cv::Scalar(200,200,200);
        }
        cv::putText(frame, status_text, cv::Point(10,30), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);
        cv::imshow("Stage 2 - Original", frame);

        char key = (char)cv::waitKey(30);
        if(key == 27) break;  // ESC
        if(key == ' ') {      // SPACE
            if(foeLocked && currentTargetIsFoe) 
                std::cout << "STAGE 2: FIRE! Enemy has been eliminated.\n";
            else if (currentTargetCenter.x != -1 && !currentTargetIsFoe) 
                std::cout << "STAGE 2: DO NOT FIRE AT FRIENDLIES!\n";
            else 
                std::cout << "STAGE 2: Cannot fire, no enemy locked.\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
}

cv::Point Stage2::getContourCenter(const std::vector<cv::Point>& contour) {
    if (contour.empty()) return cv::Point(-1, -1);
    
    cv::Moments m = cv::moments(contour);
    if (m.m00 == 0) return cv::Point(-1, -1);
    
    int cx = static_cast<int>(m.m10 / m.m00);
    int cy = static_cast<int>(m.m01 / m.m00);
    return cv::Point(cx, cy);
}
#include <iostream>
#include "../include/Stage1.hpp"
#include "../include/Stage2.hpp"
#include "../include/Stage3.hpp"

int main() {
    int choice = 0;
    std::cout << "Air Defense System Simulation\n";
    std::cout << "Please select the stage you want to run:\n";
    std::cout << "1. Stage 1 (Single Target Elimination)\n";
    std::cout << "2. Stage 2 (Friend/Foe Discrimination)\n";
    std::cout << "3. Stage 3 (Elimination with Given Engagement)\n";
    std::cout << "Your selection (1-3): ";
    std::cin >> choice;

    try {
        switch (choice) {
            case 1: {
                Stage1 stage1;
                stage1.run();
                break;
            }
            case 2: {
                Stage2 stage2;
                stage2.run();
                break;
            }
            case 3: {
                Stage3 stage3;
                stage3.run();
                break;
            }
            default:
                std::cout << "Invalid selection.\n";
                break;
        }
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
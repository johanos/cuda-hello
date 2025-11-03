#include <GLFW/glfw3.h>
#include <iostream>
#include "image.h"

const int FPS = 30;

int main() {
    // Initialize GLFW
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }

    // Create a windowed mode window and its OpenGL context
    GLFWwindow* window = glfwCreateWindow(800, 600, "OpenGL Test", nullptr, nullptr);
    if (!window) {
        std::cerr << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }

    // Make the window's context current
    glfwMakeContextCurrent(window);

    // Main loop limit to 30 fps
    while (!glfwWindowShouldClose(window)) {
        // Set clear color (cornflower blue - a classic test color)
        float randomRed = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float randomGreen = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
        float randomBlue = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);   

        glClearColor(randomRed, randomGreen, randomBlue, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

        // Close window on escape key
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
            glfwSetWindowShouldClose(window, true);
        }
        // Limit to 30 FPS
        glfwWaitEventsTimeout(1.0 / FPS);
    }

    // Clean up
    glfwTerminate();
    return 0;
}

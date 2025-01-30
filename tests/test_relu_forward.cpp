// tests/test_linear_forward.cpp
#include <iostream>
#include "../include/activations.h"  // Adjust path as needed.

int main() {
    // Create a LinearLayer with input size 4 and output size 3.
    ReLu relu;

    // Define a test input vector of size 4.
    std::vector<float> input = {-1.0f, 2.0f, 3.0f, -4.0f};

    // Perform the forward pass.
    std::vector<float> output = relu.forward(input);

    // Print the output vector.
    std::cout << "Output of the forward pass:" << std::endl;
    for (float value : output) {
        std::cout << value << " ";
    }
    std::cout << std::endl;

    return 0;
}

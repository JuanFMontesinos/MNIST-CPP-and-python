// tests/test_flatten.cpp
#include <iostream>
#include "../include/modules.h"

int main()
{
    // Create a LinearLayer with input size 4 and output size 3.
    Flatten flatten;

    // Define a test input vector of size 4.
    std::vector<std::vector<float>> input = {{1.0f, 2.0f, 3.0f, 4.0f},
                                             {5.0f, 6.0f, 7.0f, 8.0f}};
    // Perform the forward pass.
    std::vector<float> output = flatten.forward(input);
    std::cout << "Input of the forward pass:" << std::endl;
    for (const std::vector<float> &row : input)
    {
        for (float value : row)
        {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "Output of the forward pass:" << std::endl;
    for (float value : output)
    {
        std::cout << value << " ";
    }
    std::cout << std::endl;
}

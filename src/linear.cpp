// src/linear.cpp
#include "modules.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>

LinearLayer::LinearLayer(int input_size, int output_size) : input_size(input_size), output_size(output_size)
{

    // Initialize weights and bias
    weights = std::vector<std::vector<float>>(output_size, std::vector<float>(input_size));
    bias = std::vector<float>(output_size);


    // Xavier initialization
    float xavier_limit = std::sqrt(2.0f / input_size);
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i][j] = ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1) * xavier_limit;
        }
    }

    for (int i = 0; i < output_size; ++i)
    {
        bias[i] = ((static_cast<float>(rand()) / RAND_MAX) * 2 - 1) * xavier_limit;
    }

    grad_bias = std::vector<float>(output_size, 0.0f);
    grad_weights = std::vector<std::vector<float>>(output_size, std::vector<float>(input_size, 0.0f));

}

std::vector<float> LinearLayer::forward(const std::vector<float> &input)
{

    assert(input.size() == input_size && "Input size must match the layer's input size");

    std::vector<float> output(output_size);
    this->input = input; // for backward

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            output[i] += weights[i][j] * input[j];
        }
        output[i] += bias[i];
    }

    return output;
}

std::vector<float> LinearLayer::backward(const std::vector<float> &grad_output)
{
    vector<float> grad_input(input_size, 0);

    for (int i = 0; i < output_size; ++i)
    {
        // Gradient w.r.t bias
        grad_bias[i] += grad_output[i];

        for (int j = 0; j < input_size; ++j)
        {
            // Gradient w.r.t weights
            grad_weights[i][j] += grad_output[i] * input[j];

            // Gradient w.r.t input
            grad_input[j] += grad_output[i] * weights[i][j];
        }
    }

    return grad_input;
}

void LinearLayer::update(float lr)
{
    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            weights[i][j] -= lr * grad_weights[i][j];
        }
        bias[i] -= lr * grad_bias[i];
    }

    // Reset gradients
    grad_weights = vector<vector<float>>(output_size, vector<float>(input_size, 0.0f));
    grad_bias = vector<float>(output_size, 0.0f);
}
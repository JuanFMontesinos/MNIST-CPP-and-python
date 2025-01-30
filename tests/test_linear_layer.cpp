//tests/test_linear_layer.cpp
#include <iostream>
#include <cmath>
#include <cassert>
#include <vector>

// Include your layer header:
#include "../include/modules.h"    // Contains LinearLayer
#include "../include/tensor.h"     // Hypothetical Tensor interface

// A small epsilon for floating-point comparisons:
static const float EPS = 1e-5f;

// Helper to check if two floats are "close enough"
bool closeEnough(float a, float b, float eps = EPS) {
    return std::fabs(a - b) < eps;
}

int main() {
    // ----------------------------------------------------------------
    // 1. Construct a small LinearLayer with known weights/bias
    // ----------------------------------------------------------------
    int input_size  = 3;
    int output_size = 2;
    int batch_size  = 2;

    // We create the layer. By default, the constructor randomizes
    // weights, but for a test, we'll forcibly overwrite them to
    // known values.
    LinearLayer layer(input_size, output_size, batch_size);

    // Overwrite weights on GPU/CPU with known values:
    // Suppose we have W = [[0.1,0.2,0.3],[0.4,0.5,0.6]]
    // and b = [0.01, 0.02]
    {
        // Move them back to CPU if needed, set data, then move to GPU.
        layer.weights.Cpu();
        layer.bias.Cpu();

        float* wData = static_cast<float*>(layer.weights.getAddress());
        float* bData = static_cast<float*>(layer.bias.getAddress());

        wData[0] = 0.1f; wData[1] = 0.2f; wData[2] = 0.3f;  // first row
        wData[3] = 0.4f; wData[4] = 0.5f; wData[5] = 0.6f;  // second row

        bData[0] = 0.01f;
        bData[1] = 0.02f;

        // Copy back to GPU if your layer expects that
        layer.weights.Cuda();
        layer.bias.Cuda();
    }

    // ----------------------------------------------------------------
    // 2. Forward pass
    // ----------------------------------------------------------------
    // We'll create an input = [[1,2,3],[4,5,6]]
    std::vector<float> inputData = { 1.f,2.f,3.f,  4.f,5.f,6.f };
    Tensor inputTensor({batch_size, input_size}, DType::FLOAT, DeviceType::CPU, inputData.data());
    inputTensor.Cuda();

    Tensor outputTensor = layer.forward(inputTensor);

    // Bring the result back to CPU to compare
    outputTensor.Cpu();
    float* outPtr = static_cast<float*>(outputTensor.getAddress());

    // Reference forward result:
    // [
    //   [1.41, 3.22],
    //   [3.21, 7.72]
    // ]
    float refForward[4] = {1.41f, 3.22f, 3.21f, 7.72f};

    for(int i = 0; i < 4; ++i) {
        assert(closeEnough(outPtr[i], refForward[i]) && "Forward pass mismatch!");
    }

    std::cout << "Forward pass OK.\n";

    // ----------------------------------------------------------------
    // 3. Backward pass
    // ----------------------------------------------------------------
    // Provide a known upstream gradient, shape = (2,2)
    std::vector<float> gradOutData = {0.1f, 0.2f,  0.3f, 0.4f};
    Tensor gradOutTensor({batch_size, output_size}, DType::FLOAT, DeviceType::CPU, gradOutData.data());
    gradOutTensor.Cuda();

    // Call backward
    Tensor downstreamGrad = layer.backward(gradOutTensor);

    // The backward call *should* fill layer.grad_weights, layer.grad_bias,
    // and return downstreamGrad. We must copy them all back to CPU
    // to compare with reference values.

    // 3.1. Check grad_weights
    layer.grad_weights.Cpu();
    float* gwPtr = static_cast<float*>(layer.grad_weights.getAddress());
    // Expected:
    // [
    //   [1.3, 1.7, 2.1],
    //   [1.8, 2.4, 3.0]
    // ]
    float refGW[6] = {1.3f, 1.7f, 2.1f,
                      1.8f, 2.4f, 3.0f};

    for(int i = 0; i < 6; ++i) {
        assert(closeEnough(gwPtr[i], refGW[i]) && "grad_weights mismatch!");
    }

    // 3.2. Check grad_bias
    layer.grad_bias.Cpu();
    float* gbPtr = static_cast<float*>(layer.grad_bias.getAddress());
    // Expected: [0.4, 0.6]
    float refGB[2] = {0.4f, 0.6f};
    for(int i = 0; i < 2; ++i) {
        assert(closeEnough(gbPtr[i], refGB[i]) && "grad_bias mismatch!");
    }

    // 3.3. Check downstream grad
    downstreamGrad.Cpu();
    float* dxPtr = static_cast<float*>(downstreamGrad.getAddress());
    // Expected:
    // [
    //   [0.09, 0.12, 0.15],
    //   [0.19, 0.26, 0.33]
    // ]
    float refDX[6] = {0.09f, 0.12f, 0.15f,
                      0.19f, 0.26f, 0.33f};
    for(int i = 0; i < 6; ++i) {
        assert(closeEnough(dxPtr[i], refDX[i]) && "Downstream grad mismatch!");
    }

    std::cout << "Backward pass OK.\n";

    // ----------------------------------------------------------------
    // 4. Parameter update
    // ----------------------------------------------------------------
    float lr = 0.1f;
    layer.update(lr);

    // Now check new weights and biases
    layer.weights.Cpu();
    layer.bias.Cpu();

    float* newW = static_cast<float*>(layer.weights.getAddress());
    float* newB = static_cast<float*>(layer.bias.getAddress());

    // Expected new weights:
    // [
    //   [-0.03, 0.03, 0.09],
    //   [ 0.22, 0.26, 0.30]
    // ]
    float refWUpdated[6] = {
       -0.03f,  0.03f,  0.09f,
        0.22f,  0.26f,  0.30f
    };

    // Expected new bias: [-0.03, -0.04]
    float refBUpdated[2] = {-0.03f, -0.04f};

    for(int i = 0; i < 6; ++i) {
        assert(closeEnough(newW[i], refWUpdated[i]) && "Updated weight mismatch!");
    }
    for(int i = 0; i < 2; ++i) {
        assert(closeEnough(newB[i], refBUpdated[i]) && "Updated bias mismatch!");
    }

    std::cout << "Update pass OK.\n";
    std::cout << "All LinearLayer tests passed successfully.\n";
    return 0;
}

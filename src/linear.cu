// src/linear.cu
#include <vector>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <stdexcept>
#include <cuda_runtime.h>
#include "modules.h"
#include "tensor.h"
#include "functionals.h"

LinearLayer::LinearLayer(int input_size, int output_size, int batch_size) : input_size(input_size),
                                                                            output_size(output_size),
                                                                            batch_size(batch_size)
{
    // Flatten weights and biases
    // This ensures contiguous memory.
    std::vector<float> weights_data(output_size * input_size);
    std::vector<float> bias_data(output_size);

    // Xavier initialization in cpu.
    float xavier_limit = std::sqrt(2.0f / input_size);

    for (int i = 0; i < output_size; ++i)
    {
        for (int j = 0; j < input_size; ++j)
        {
            float rand_val = static_cast<float>(rand()) / RAND_MAX; // [0,1)
            float scaled = (rand_val * 2.0f - 1.0f) * xavier_limit; // [-xavier_limit, xavier_limit]
            weights_data[i * input_size + j] = scaled;
        }
        float rand_val = static_cast<float>(rand()) / RAND_MAX;
        bias_data[i] = (rand_val * 2.0f - 1.0f) * xavier_limit;
    }

    // Creates weights and bias tensors
    // Arrays are copied when constructing the tensor so that
    // memory ownership is guaranteed by Tensor.
    this->weights = Tensor({output_size, input_size}, DType::FLOAT, DeviceType::CPU, weights_data.data());
    this->weights.Cuda();

    this->bias = Tensor({output_size}, DType::FLOAT, DeviceType::CPU, bias_data.data());
    this->bias.Cuda();

    // Grad bias: shape ( output_size)
    // Grad weights: shape ( output_size, input_size)
    std::vector<float> grad_bias_data(output_size, 0.0f);
    this->grad_bias = Tensor({output_size}, DType::FLOAT, DeviceType::CPU, grad_bias_data.data());
    this->grad_bias.Cuda();

    // Grad weights: shape ( output_size, input_size)
    std::vector<float> grad_weights_data(output_size * input_size, 0.0f);
    this->grad_weights = Tensor({output_size, input_size}, DType::FLOAT, DeviceType::CPU, grad_weights_data.data());
    this->grad_weights.Cuda();

    // Memory management for std::vectors is automatic and will be freed
    // after the constructor finishes.
}

// We declare the function so that the compiler knows it's defined elsewhere
__global__ void linear_forward(float *output, const float *weights, const float *bias, const float *input, int batch_size, int input_size, int output_size);

Tensor LinearLayer::forward(Tensor &input)
{
    // 1D computation, we create a 1D grid of threads
    dim3 tpb(128, 1); // 128 threads per block

    dim3 grid(
        (output_size + tpb.x - 1) / tpb.x,
        batch_size);
    Tensor output = Tensor::empty({batch_size, output_size}, DType::FLOAT, DeviceType::GPU);
    float *output_d = static_cast<float *>(output.getAddress());
    float *weights_d = static_cast<float *>(this->weights.getAddress());
    float *bias_d = static_cast<float *>(this->bias.getAddress());
    float *input_d = static_cast<float *>(input.getAddress());

    this->input = &input;
    linear_forward<<<grid, tpb>>>(output_d, weights_d, bias_d, input_d, batch_size, input_size, output_size);
    return output;
}
__global__ void linear_backward(
    float *downstream_grad,
    float *grad_weights,
    float *grad_bias,
    const float *upstream_grad,
    const float *weights,
    const float *input,
    int batch_size,
    int input_size,
    int output_size);

Tensor LinearLayer::backward(const Tensor &upstream_grad)
{
    dim3 tpb(128, 1);
    dim3 grid((output_size + tpb.x - 1) / tpb.x, batch_size);
    Tensor &input = *this->input;
    Tensor downstream_grad = Tensor::zeros({batch_size, input_size}, DType::FLOAT, DeviceType::GPU);
    float *downstream_grad_d = static_cast<float *>(downstream_grad.getAddress());
    float *grad_weights_d = static_cast<float *>(grad_weights.getAddress());
    float *grad_bias_d = static_cast<float *>(grad_bias.getAddress());
    float *upstream_grad_d = static_cast<float *>(upstream_grad.getAddress());
    float *weights_d = static_cast<float *>(weights.getAddress());
    float *input_d = static_cast<float *>(input.getAddress());

    return downstream_grad;
}

__global__ void linear_update_weights(float *weights, const float *grad_weights, float lr, int input_size, int output_size);
__global__ void linear_update_bias(float *bias, const float *grad_bias, float lr, int output_size);

void LinearLayer::update(float lr)
{

    int blockSize = 64;
    int total_elems = output_size * input_size;
    int gridSize = (total_elems + blockSize - 1) / blockSize;

    float *d_weights = static_cast<float *>(weights.getAddress());
    float *d_grad_weights = static_cast<float *>(grad_weights.getAddress());

    linear_update_weights<<<gridSize, blockSize>>>(d_weights, d_grad_weights, lr, input_size, output_size);

    int gridSize2 = (output_size + blockSize - 1) / blockSize;

    float *d_bias = static_cast<float *>(bias.getAddress());
    float *d_grad_bias = static_cast<float *>(grad_bias.getAddress());

    linear_update_bias<<<gridSize2, blockSize>>>(d_bias, d_grad_bias, lr, output_size);

    // 3. Reset gradients
    cudaMemset(d_grad_weights, 0, output_size * input_size * sizeof(float));
    cudaMemset(d_grad_bias, 0, output_size * sizeof(float));
}

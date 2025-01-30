//scr/kernels.cu
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>

__global__ void linear_forward(float *output, const float *weights, const float *bias, const float *input, int batch_size, int input_size, int output_size)
{
    int batch = blockIdx.y;                          // Each block processes a different batch
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Each thread computes one output neuron

    if (batch < batch_size && idx < output_size)
    {
        float sum = 0.0f;
        for (int i = 0; i < input_size; i++)
        {
            sum += input[batch * input_size + i] * weights[idx * input_size + i];
        }

        output[batch * output_size + idx] = sum + bias[idx];
    }
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
    int output_size)
{
    int batch = blockIdx.y;
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (batch < batch_size && out_idx < output_size)
    {
        // The gradient from layer above for (batch, out_idx)
        float up_val = upstream_grad[batch * output_size + out_idx];

        // Accumulate bias gradient
        atomicAdd(&grad_bias[out_idx], up_val);

        // Accumulate weight gradient and downstream gradient
        for (int in_idx = 0; in_idx < input_size; ++in_idx)
        {
            // grad_weights is shape [output_size, input_size],
            // so the element is at (out_idx, in_idx)
            atomicAdd(&grad_weights[out_idx * input_size + in_idx],
                      up_val * input[batch * input_size + in_idx]);

            // downstream_grad is shape [batch_size, input_size],
            // so the element is at (batch, in_idx)
            atomicAdd(&downstream_grad[batch * input_size + in_idx],
                      up_val * weights[out_idx * input_size + in_idx]);
        }
    }
}

__global__ void linear_update_weights(float *weights, const float *grad_weights, float lr, int input_size, int output_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elems = output_size * input_size;

    if (tid < total_elems)
    {
        weights[tid] -= lr * grad_weights[tid];
    }
}

__global__ void linear_update_bias(float *bias, const float *grad_bias, float lr, int output_size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < output_size)
    {
        bias[tid] -= lr * grad_bias[tid];
    }
}

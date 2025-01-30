//src/tensor.cpp
#include "tensor.h"
#include <unordered_map>
#include <vector>
#include <cassert>
#include <cstring>
#include <iostream>
#include <cuda_runtime.h>
#include <functionals.h>

using namespace std;

Tensor::Tensor() : shape({}), dtype(DType::FLOAT), device(DeviceType::CPU), A(nullptr)
{
}

Tensor::Tensor(const vector<int> &shape, DType dtype, DeviceType device, void *address) : shape(shape), dtype(dtype), device(device)
{
    // Copy the data from the address to the tensor to ensure memory ownership
    size_t tensorSize = this->TensorSize();
    if (device == DeviceType::GPU)
    {
        CUDA_CHECK(cudaMalloc(&A, tensorSize));
        CUDA_CHECK(cudaMemcpy(A, address, tensorSize, cudaMemcpyDeviceToDevice));
    }
    else
    {
        A = malloc(tensorSize);
        if (!A)
        {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
        memcpy(A, address, tensorSize);
    }
}
Tensor Tensor::copy() const
{
    // Just call the "main" constructor that copies data from a raw pointer.
    // The constructor will do GPU->GPU copy or CPU->CPU copy as needed.
    return Tensor(this->shape, this->dtype, this->device, this->A);
}
Tensor Tensor::empty(const vector<int> &shape, DType dtype, DeviceType device)
{
    size_t tensorSize = functionals::TensorSize(shape, dtype);
    if (device == DeviceType::GPU)
    {
        void *A_d;
        CUDA_CHECK(cudaMalloc(&A_d, tensorSize));
        return Tensor(shape, dtype, device, A_d);
    }
    else
    {
        void *A_h = malloc(tensorSize);
        if (!A_h)
        {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
        return Tensor(shape, dtype, device, A_h);
    }
}

Tensor Tensor::zeros(const vector<int> &shape, DType dtype, DeviceType device)
{
    size_t tensorSize = functionals::TensorSize(shape, dtype);
    if (device == DeviceType::GPU)
    {
        void *A_d;
        CUDA_CHECK(cudaMalloc(&A_d, tensorSize));
        CUDA_CHECK(cudaMemset(A_d, 0, tensorSize)); // Set GPU memory to zero
        return Tensor(shape, dtype, device, A_d);
    }
    else
    {
        void *A_h = calloc(1, tensorSize); // Allocate and initialize to zero
        if (!A_h)
        {
            throw std::runtime_error("Failed to allocate CPU memory");
        }
        return Tensor(shape, dtype, device, A_h);
    }
}

Tensor::~Tensor()
{
    if (device == DeviceType::GPU)
    {
        CUDA_CHECK(cudaFree(A));
    }
    else
    {
        free(A);
    }
}

const vector<int> &Tensor::getShape() const
{
    return shape;
}
DType Tensor::getDType() const
{
    return dtype;
}
DeviceType Tensor::getDevice() const
{
    return device;
}

bool Tensor::isCuda() const
{
    return device == DeviceType::GPU;
}
bool Tensor::isCpu() const
{
    return device == DeviceType::CPU;
}

size_t Tensor::TensorSize()
{
    return functionals::TensorSize(shape, dtype);
}
void Tensor::Cuda()
{
    if (device == DeviceType::GPU)
    {
        return;
    }
    else
    {
        void *A_h = A;
        void *A_d;
        size_t TensorSize = this->TensorSize();

        CUDA_CHECK(cudaMalloc(&A_d, TensorSize));
        // Copy data from CPU to GPU
        CUDA_CHECK(cudaMemcpy(A_d, A_h, TensorSize, cudaMemcpyHostToDevice));

        A = A_d;
        device = DeviceType::GPU;

        free(A_h);
    }
}

void Tensor::Cpu()
{
    if (device == DeviceType::CPU)
    {
        return;
    }
    else
    {
        size_t TensorSize = this->TensorSize();

        void *A_d = A;
        void *A_h = malloc(TensorSize);
        if (!A_h)
        {
            throw std::runtime_error("Failed to allocate CPU memory");
        }

        cudaDeviceSynchronize(); // Wait for GPU to finish
        CUDA_CHECK(cudaMemcpy(A_h, A_d, this->TensorSize(), cudaMemcpyDeviceToHost)); // Copy data from GPU to CPU
        CUDA_CHECK(cudaFree(A_d));                                                    // Free GPU memory

        A = A_h;
        device = DeviceType::CPU;
    }
}

void *Tensor::getAddress() const
{
    return A;
}

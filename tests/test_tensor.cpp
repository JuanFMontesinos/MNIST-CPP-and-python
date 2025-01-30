#include <vector>            // For std::vector
#include <cassert>           // For assert
#include <cstring>           // For std::memcmp
#include <iostream>          // For std::cout, std::cerr
#include <cuda_runtime.h>    // For CUDA runtime functions

#include "../include/tensor.h"

// Helper function to compare two buffers
bool compareBuffers(const void* a, const void* b, size_t size) {
    return std::memcmp(a, b, size) == 0;
}

// Test 1: Constructor for CPU Tensor
void testConstructorCPU() {
    std::vector<int> shape = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t(shape, DType::FLOAT, DeviceType::CPU, data);

    assert(t.getShape() == shape);
    assert(t.getDType() == DType::FLOAT);
    assert(t.getDevice() == DeviceType::CPU);

    // Verify the data was copied correctly
    assert(compareBuffers(t.getAddress(), data, t.TensorSize()));

    std::cout << "Test Constructor CPU: PASSED\n";
}

// Test 2: Constructor for GPU Tensor
void testConstructorGPU() {
    std::vector<int> shape = {2, 3};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    void* deviceData;
    CUDA_CHECK(cudaMalloc(&deviceData, sizeof(data)));
    CUDA_CHECK(cudaMemcpy(deviceData, data, sizeof(data), cudaMemcpyHostToDevice));

    Tensor t(shape, DType::FLOAT, DeviceType::GPU, deviceData);

    assert(t.getShape() == shape);
    assert(t.getDType() == DType::FLOAT);
    assert(t.getDevice() == DeviceType::GPU);

    // Verify the data was copied correctly
    float* hostCopy = new float[shape[0] * shape[1]];
    CUDA_CHECK(cudaMemcpy(hostCopy, t.getAddress(), t.TensorSize(), cudaMemcpyDeviceToHost));
    assert(compareBuffers(hostCopy, data, t.TensorSize()));

    delete[] hostCopy;
    std::cout << "Test Constructor GPU: PASSED\n";
}

// Test 3: Transition from CPU to GPU
void testCpuToGpuTransition() {
    std::vector<int> shape = {2, 2};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t(shape, DType::FLOAT, DeviceType::CPU, data);

    // Move tensor to GPU
    t.Cuda();
    assert(t.getDevice() == DeviceType::GPU);

    // Verify the data on the GPU
    float* hostCopy = new float[shape[0] * shape[1]];
    CUDA_CHECK(cudaMemcpy(hostCopy, t.getAddress(), t.TensorSize(), cudaMemcpyDeviceToHost));
    assert(compareBuffers(hostCopy, data, t.TensorSize()));

    delete[] hostCopy;
    std::cout << "Test CPU to GPU Transition: PASSED\n";
}

// Test 4: Transition from GPU to CPU
void testGpuToCpuTransition() {
    std::vector<int> shape = {2, 2};
    float data[] = {5.0f, 6.0f, 7.0f, 8.0f};
    void* deviceData;
    CUDA_CHECK(cudaMalloc(&deviceData, sizeof(data)));
    CUDA_CHECK(cudaMemcpy(deviceData, data, sizeof(data), cudaMemcpyHostToDevice));

    Tensor t(shape, DType::FLOAT, DeviceType::GPU, deviceData);

    // Move tensor to CPU
    t.Cpu();
    assert(t.getDevice() == DeviceType::CPU);

    // Verify the data on the CPU
    assert(compareBuffers(t.getAddress(), data, t.TensorSize()));

    std::cout << "Test GPU to CPU Transition: PASSED\n";
}

// Test 5: Destructor Frees Memory
void testDestructor() {
    std::vector<int> shape = {2, 2};
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};

    {
        Tensor t(shape, DType::FLOAT, DeviceType::CPU, data);
    } // Destructor should free CPU memory without errors

    {
        void* deviceData;
        CUDA_CHECK(cudaMalloc(&deviceData, sizeof(data)));
        Tensor t(shape, DType::FLOAT, DeviceType::GPU, deviceData);
    } // Destructor should free GPU memory without errors

    std::cout << "Test Destructor: PASSED\n";
}

// Run All Tests
int main() {
    try {
        testConstructorCPU();
        testConstructorGPU();
        testCpuToGpuTransition();
        testGpuToCpuTransition();
        testDestructor();
        std::cout << "All Tests PASSED!\n";
    } catch (const std::exception& ex) {
        std::cerr << "Test FAILED: " << ex.what() << std::endl;
        return 1;
    }
    return 0;
}

#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <enums.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                         \
    do                                                                           \
    {                                                                            \
        cudaError_t err = call;                                                  \
        if (err != cudaSuccess)                                                  \
        {                                                                        \
            std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl; \
            exit(err);                                                           \
        }                                                                        \
    } while (0)

// Tensor class
class Tensor
{
private:
    DType dtype;            // Data type of the tensor
    std::vector<int> shape; // Shape of the tensor (e.g., [2, 3, 4])
    DeviceType device;      // Device where the tensor resides (CPU or GPU)
    void *A;                // Memory address (can represent GPU or CPU memory)

public:
    // Constructor
    Tensor();
    Tensor(const std::vector<int> &shape, DType dtype, DeviceType device, void *address);
    static Tensor empty(const std::vector<int> &shape, DType dtype, DeviceType device);
    static Tensor zeros(const std::vector<int> &shape, DType dtype, DeviceType device);
    // 1) Delete copy constructor & copy assignment
    Tensor(const Tensor &) = delete;
    Tensor &operator=(const Tensor &) = delete;

    // move constructor
    Tensor(Tensor &&other) noexcept
    {
        this->A = other.A;
        this->shape = std::move(other.shape);
        this->dtype = other.dtype;
        this->device = other.device;
        other.A = nullptr;
    }

    // move assignment operator
    Tensor &operator=(Tensor &&other) noexcept
    {
        if (this != &other)
        {
            // First free our own memory
            if (A)
            {
                if (device == DeviceType::GPU)
                    cudaFree(A);
                else
                    free(A);
            }
            // Transfer ownership from 'other'
            A = other.A;
            shape = std::move(other.shape);
            dtype = other.dtype;
            device = other.device;
            other.A = nullptr;
        }
        return *this;
    }
    // Destructor
    ~Tensor();

    // Accessors
    const std::vector<int> &getShape() const;
    Tensor copy() const;
    size_t TensorSize();
    DType getDType() const;
    DeviceType getDevice() const;
    void *getAddress() const;
    void Cuda();
    void Cpu();
    bool isCuda() const;
    bool isCpu() const;
};

#endif // TENSOR_H

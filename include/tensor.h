#ifndef TENSOR_H
#define TENSOR_H

#include <vector>
#include <string>
#include <enums.h>

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

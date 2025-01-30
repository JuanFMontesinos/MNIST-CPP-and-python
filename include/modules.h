// src/modules.h
#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "tensor.h"

using namespace std;
class Module
{
public:
    virtual Tensor forward(Tensor &input) = 0;
    virtual Tensor backward(const Tensor &grad_output) = 0;
    virtual void update(float lr) = 0;
};

class LinearLayer : public Module
{
public:
    LinearLayer(int input_size, int output_size, int batch_size);
    Tensor forward(Tensor &input) override;
    Tensor backward(const Tensor &grad_output) override;
    void update(float lr) override;
    // Note that now we account for a batched processing. Each element of the batch will have its own gradient.
    // But they all use the same weights and bias.
    Tensor weights; // output_size X input_size
    Tensor bias; // output_size
    Tensor grad_weights; // Batch X output_size X input_size
    Tensor grad_bias; // Batch X output_size

private:
    int input_size;
    int output_size;
    int batch_size;

    Tensor *input;
};

#endif
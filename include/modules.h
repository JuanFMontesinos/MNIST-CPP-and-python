// src/modules.h
#ifndef MODULE_H
#define MODULE_H

#include <vector>
using namespace std;
class Module
{
public:
    virtual vector<float> forward(const vector<float> &input) = 0;
    virtual vector<float> backward(const vector<float> &grad_output) = 0;
    virtual void update(float lr) = 0;
};

class LinearLayer : public Module
{
public:
    LinearLayer(int input_size, int output_size);
    vector<float> forward(const vector<float> &input) override;
    vector<float> backward(const vector<float> &grad_output) override;
    void update(float lr) override;

    // Expose weights and biases for comparison
    vector<vector<float>> weights;
    vector<float> bias;
    vector<vector<float>> grad_weights;
    vector<float> grad_bias;

private:
    int input_size;
    int output_size;

    // Store input for backward pass
    vector<float> input;
};

#endif
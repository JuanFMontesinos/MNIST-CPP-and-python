#ifndef ACTIVATIONS
#define ACTIVATIONS

#include <vector>
#include "modules.h"

class ReLu : public Module
{
public:
    std::vector<float> forward(const std::vector<float> &input) override;
    std::vector<float> backward(const std::vector<float> &grad_output) override;
    void update(float lr) override;
private:
    std::vector<bool> zeroed; // Store for backpropagation
};

#endif
#include "activations.h"
#include <cstddef>
#include <vector>
#include <cmath>

using namespace std;

vector<float> ReLu::forward(const vector<float> &input)
{
    vector<float> output = input;
    zeroed = vector<bool>(input.size(), false);
    for (size_t i = 0; i < input.size(); i++)
    {
        if (input[i] < 0)
        {
            output[i] = 0.0f;
            zeroed[i] = true;
        }
    }
    return output;
}
vector<float> ReLu::backward(const vector<float> &grad_output)
{
    vector<float> grad_input(grad_output.size());
    for (size_t i = 0; i < grad_output.size(); i++)
    {
        grad_input[i] = zeroed[i] ? 0.0f : grad_output[i];
    }
    return grad_input;
}
void ReLu::update(float lr)
{
}
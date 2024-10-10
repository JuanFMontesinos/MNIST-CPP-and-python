
#include <vector>
#include <cmath>
#include "cross_entropy.h"
#include "functionals.h"
using namespace std;

// https://sefiks.com/2017/12/17/a-gentle-introduction-to-cross-entropy-loss-function/

float SoftmaxndCrossEntropy::forward(const vector<float> &input, int class_label)
{
    probabilities = functionals::softmax(input);
    this->class_label = class_label; // save for backpropagation
    double loss = 0;

    // E = – ∑ ci . log(pi) + (1 – ci ). log(1 – pi)
    for (int i = 0; i < num_classes; i++)
    {
        
        loss -= class_label == i ? log(probabilities[i]) : log(1 - probabilities[i]);
    }
    return (float)loss;
};
vector<float> SoftmaxndCrossEntropy::backward()
{
    // ∂E/∂zi = pi – ci for each class i
    vector<float> grad_input(num_classes);
    for (int i = 0; i < num_classes; ++i)
    {
        grad_input[i] = probabilities[i] - (i == class_label ? 1 : 0);
    }
    return grad_input;
}

#ifndef LOSS_H

#include <vector>

using namespace std;

class SoftmaxndCrossEntropy
{
public:
    SoftmaxndCrossEntropy(int num_classes) : num_classes(num_classes) {}
    const int num_classes;
    float forward(const vector<float> &input, int class_label);
    vector<float> backward();

private:
    vector<float> probabilities; // for backpropagation
    int class_label;             // for backpropagation
};

#endif
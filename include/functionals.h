#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H

#include <vector>
#include <cmath>
#include <enums.h>

using namespace std;

namespace functionals
{
    vector<float> softmax(const vector<float> &input);
    vector<float> flatten2d(const vector<vector<float>> &input);
    size_t TensorSize(const vector<int> &shape, DType dtype);
}

#endif
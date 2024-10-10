#ifndef FUNCTIONALS_H
#define FUNCTIONALS_H

#include <vector>
#include <cmath>

using namespace std;

namespace functionals
{
    vector<float> softmax(const vector<float> &input);
    vector<float> flatten2d(const vector<vector<float>> &input);
}

#endif
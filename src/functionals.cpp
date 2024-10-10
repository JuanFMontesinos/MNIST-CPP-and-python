#include <vector>
#include "functionals.h"

using namespace std;
vector<float> functionals::softmax(const vector<float> &input)
{
    vector<float> output(input.size());
    double exp_sum = 0;
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] = exp(input[i]);
        exp_sum += output[i];
    }
    for (size_t i = 0; i < input.size(); i++)
    {
        output[i] /= exp_sum;
    }
    return output;
}
vector<float> functionals::flatten2d(const vector<vector<float>> &input)
{
    size_t n_rows = input.size();
    size_t n_cols = input[0].size();
    vector<float> output(n_rows * n_cols);
    for (size_t i = 0; i < n_rows; i++)
    {
        for (size_t j = 0; j < n_cols; j++)
        {
            output[i * n_cols + j] = input[i][j];
        }
    }
    return output;
}
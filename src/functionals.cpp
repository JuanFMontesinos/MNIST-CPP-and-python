#include <vector>
#include "functionals.h"
#include <unordered_map>

using namespace std;

static const std::unordered_map<DType, size_t> DTypeSizeMap = {
    {DType::INT, sizeof(int)},
    {DType::FLOAT, sizeof(float)},
    {DType::DOUBLE, sizeof(double)},
    {DType::CHAR, sizeof(char)}};
    
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

size_t functionals::TensorSize(const vector<int> &shape, DType dtype)
{
    size_t tensor_size = 1;
    for (int i = 0; i < shape.size(); i++)
    {
        tensor_size *= shape[i];
    }
    auto dtype_size = DTypeSizeMap.find(dtype)->second;

    return tensor_size * dtype_size;
}
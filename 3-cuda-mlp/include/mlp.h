#ifndef __MLP_HPP__
#define __MLP_HPP__

#include <vector>

void printCudaVersion();

class MLP_Network
{
public:
    MLP_Network(int input_dim, int output_dim, std::vector<int> hidden_dim, std::vector<std::vector<float>> weight, std::vector<std::vector<float>> bias);
    ~MLP_Network();
    void forward(float *input, float *output);
    // void backward();
    // void update();
    // void train();
    // void test();
    // void save();
    // void load();

private:
    float **weights_;
    float **biases_;
    std::vector<int> hidden_dim_;
    int input_dim_;
    int output_dim_;
};

#endif // __MLP_HPP__
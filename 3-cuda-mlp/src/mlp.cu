#include <stdio.h>

#include <iostream>
#include <fstream>

#include "mlp.h"
#include "cuda_runtime.h"

void printCudaVersion()
{
    int runtime_ver, driver_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    cudaDriverGetVersion(&driver_ver);
    printf("<<<=====================|||=====================>>>\n");
    printf("CUDA Compiled version: %d, %d, %d\n", __CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, __CUDACC_VER_BUILD__);
    printf("CUDA Runtime version: %d\n", runtime_ver);
    printf("CUDA Driver version: %d\n", driver_ver);
    printf("<<<=====================|||=====================>>>\n\n");
}

__global__ void matrixMulKernel(float *matrixA, float *matrixB, float *matrixC, int rowsA, int colsA, int colsB)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rowsA && col < colsB)
    {
        float sum = 0.0f;
        for (int k = 0; k < colsA; k++)
        {
            sum += matrixA[row * colsA + k] * matrixB[k * colsB + col];
        }
        matrixC[row * colsB + col] = sum;

        // Print intermediate result
        // printf("Intermediate result at rowsA %d, colsB %d  [%d][%d]: %.5f\n", rowsA, colsB, row, col, sum);
    }
}

__global__ void matrixAddKernel(float *matrixA, float *matrixB, float *matrixC, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        matrixC[index] = matrixA[index] + matrixB[index];
    }
}

__global__ void softmaxKernel(float *input, float *output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;

        // Compute the exponential of each element
        float expVal = expf(input[index]);

        // Compute the sum of exponentials for the row
        float sumExp = 0.0f;
        for (int i = 0; i < cols; ++i)
        {
            sumExp += expf(input[row * cols + i]);
        }

        // Compute the softmax value for the element
        output[index] = expVal / sumExp;
    }
}

// elu activation function
__global__ void eluKernel(float *input, float *output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols)
    {
        int index = row * cols + col;
        output[index] = input[index] > 0 ? input[index] : expf(input[index]) - 1;
    }
}

float *matrixMul(float *matrixA, float *matrixB, int rowsA, int colsA, int rowsB, int colsB)
{
    float *matrixC;

    printf("MUL: ROW A %d COLS A %d ROW B %d COLS B %d\n", rowsA, colsA, rowsB, colsB);

    cudaMalloc((void **)&matrixC, rowsA * colsB * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((colsB + blockSize.x - 1) / blockSize.x, (rowsA + blockSize.y - 1) / blockSize.y);

    matrixMulKernel<<<gridSize, blockSize>>>(matrixA, matrixB, matrixC, rowsA, colsA, colsB);

    return matrixC;
}

float *matrixAdd(float *matrixA, float *matrixB, int rows, int cols)
{
    float *matrixC;

    printf("ADD: ROW %d COLS %d \n", rows, cols);

    cudaMalloc((void **)&matrixC, rows * cols * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    matrixAddKernel<<<gridSize, blockSize>>>(matrixA, matrixB, matrixC, rows, cols);

    return matrixC;
}

float *softmax(float *input, int rows, int cols)
{
    float *matrixC;

    printf(" ROW %d  COLS %d \n", rows, cols);

    cudaMalloc((void **)&matrixC, rows * cols * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    softmaxKernel<<<gridSize, blockSize>>>(input, matrixC, rows, cols);

    return matrixC;
}

float *elu(float *input, int rows, int cols)
{
    float *matrixC;

    printf("ELU: ROW %d COLS %d \n", rows, cols);

    cudaMalloc((void **)&matrixC, rows * cols * sizeof(float));

    dim3 blockSize(16, 16);
    dim3 gridSize((cols + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);

    eluKernel<<<gridSize, blockSize>>>(input, matrixC, rows, cols);

    return matrixC;
}

MLP_Network::MLP_Network(int input_dim, int output_dim, std::vector<int> hidden_dim)
    : input_dim_(input_dim), output_dim_(output_dim), hidden_dim_(hidden_dim)
{
    // weights_ = new float *[hidden_dim.size() + 1];
    // biases_ = new float *[hidden_dim.size() + 1];
    // for (int i = 0; i < hidden_dim.size() + 1; i++)
    // {
    //     weights_[i] = new float[hidden_dim[i] * (i == 0 ? input_dim : hidden_dim[i - 1])];
    //     biases_[i] = new float[hidden_dim[i]];
    // }

}

MLP_Network::~MLP_Network(){
    for (int i = 0; i < hidden_dim_.size() + 1; i++)
    {
        cudaFree(weights_[i]);
        cudaFree(biases_[i]);
    }
}

void MLP_Network::forward(float *input, float *output)
{
    //memcpy input to device
    float *d_input;
    int input_size = input_dim_ * sizeof(float);
    cudaMalloc(&d_input, input_size);
    cudaMemcpy(d_input, input, input_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();


    float *d_temp;

    // forward
    d_temp = matrixMul(weights_[0], d_input, hidden_dim_[0], input_dim_, input_dim_, 1);
    d_temp = matrixAdd(d_temp, biases_[0], hidden_dim_[0], 1);
    d_temp = elu(d_temp, 1, hidden_dim_[0]);

    for (int i = 1; i < hidden_dim_.size(); i++)
    {
        d_temp = matrixMul(weights_[i], d_temp, hidden_dim_[i], hidden_dim_[i - 1], hidden_dim_[i - 1], 1);
        d_temp = matrixAdd(d_temp, biases_[i], hidden_dim_[i], 1);
        d_temp = elu(d_temp, 1, hidden_dim_[i]);
    }

    d_temp = matrixMul(weights_[hidden_dim_.size()], d_temp, output_dim_, hidden_dim_[hidden_dim_.size() - 1], hidden_dim_[hidden_dim_.size() - 1], 1);
    d_temp = matrixAdd(d_temp, biases_[hidden_dim_.size()], output_dim_, 1);
    
    // memcpy output to host
    int output_size = output_dim_ * sizeof(float);
    cudaMemcpy(output, d_temp, output_size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    //free memory
    cudaFree(d_input);
    // cudaFree(d_output);
    cudaFree(d_temp);
}

void MLP_Network::load(std::string weight_path)
{
    printf("Loading weights and biases from %s\n", weight_path.c_str());
    // if (this->weights_ != NULL)
    // {
    //     for (int i = 0; i < hidden_dim_.size() + 1; i++)
    //     {
    //         cudaFree(weights_[i]);
    //         cudaFree(biases_[i]);
    //     }
    // }
    if(this->hidden_dim_.size() == 0){
        printf("No hidden layer\n");
    }
    else{
        printf("Hidden layer size: %ld\n", this->hidden_dim_.size());
    }

    //load weights and biases
    std::vector<std::vector<float>> weight;
    std::vector<std::vector<float>> bias;

    weight.resize(this->hidden_dim_.size() + 1);
    bias.resize(this->hidden_dim_.size() + 1);

    for (int i = 0; i < this->hidden_dim_.size() + 1; i++)
    {
        if (i == 0)
        {
            weight[i].resize(this->input_dim_ * this->hidden_dim_[i]);
            bias[i].resize(this->hidden_dim_[i]);
        }
        else if (i == this->hidden_dim_.size())
        {
            weight[i].resize(this->hidden_dim_[i - 1] * this->output_dim_);
            bias[i].resize(this->output_dim_);
        }
        else
        {
            weight[i].resize(this->hidden_dim_[i - 1] * this->hidden_dim_[i]);
            bias[i].resize(this->hidden_dim_[i]);
        }
    }

    // from weights dir import weight, bias, file_name format as model.0.weight, model.0.bias model.1.weight, model.1.bias ... file fromat as txt
    for (int i = 0; i < this->hidden_dim_.size() + 1; i++)
    {
        std::string weight_file = weight_path;
        weight_file += std::to_string(i * 2) ;
        weight_file += ".weight";

        std::string bias_file = weight_path;
        bias_file += std::to_string(i * 2);
        bias_file += ".bias";

        std::ifstream weight_in(weight_file);
        std::ifstream bias_in(bias_file);
        if (!weight_in.is_open() || !bias_in.is_open())
        {
            std::cout << "file open failed" << std::endl;
            return ;
        }
        for (int j = 0; j < weight[i].size(); j++)
        {
            weight_in >> weight[i][j];
        }
        for (int j = 0; j < bias[i].size(); j++)
        {
            bias_in >> bias[i][j];
        }
    }

    // // print weight and bias
    for (int i = 0; i < this->hidden_dim_.size() + 1; i++)
    {
        for (int j = 0; j < weight[i].size(); j++)
        {
            std::cout << weight[i][j] << " ";
        }
        std::cout << std::endl;
        for (int j = 0; j < bias[i].size(); j++)
        {
            std::cout << bias[i][j] << " ";
        }
        std::cout << std::endl;
    }

    weights_ = new float *[this->hidden_dim_.size() + 1];
    biases_ = new float *[this->hidden_dim_.size() + 1];
    for (int i = 0; i < this->hidden_dim_.size() + 1; i++)
    {
        weights_[i] = new float[this->hidden_dim_[i] * (i == 0 ? this->input_dim_ : this->hidden_dim_[i - 1])];
        biases_[i] = new float[this->hidden_dim_[i]];
    }

    // copy weights_ and bias to device
    for (int i = 0; i < this->hidden_dim_.size(); i++)
    {
        cudaMalloc(&weights_[i], this->hidden_dim_[i] * (i == 0 ? this->input_dim_ : this->hidden_dim_[i - 1]) * sizeof(float));
        cudaMemcpy(weights_[i], weight[i].data(), this->hidden_dim_[i] * (i == 0 ? this->input_dim_ : this->hidden_dim_[i - 1]) * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&biases_[i], this->hidden_dim_[i] * sizeof(float));
        cudaMemcpy(biases_[i], bias[i].data(), this->hidden_dim_[i] * sizeof(float), cudaMemcpyHostToDevice);
    }

    // output layer
    cudaMalloc(&weights_[this->hidden_dim_.size()], this->hidden_dim_[this->hidden_dim_.size() - 1] * this->output_dim_ * sizeof(float));
    cudaMemcpy(weights_[this->hidden_dim_.size()], weight[this->hidden_dim_.size()].data(), this->hidden_dim_[this->hidden_dim_.size() - 1] * this->output_dim_ * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc(&biases_[this->hidden_dim_.size()], this->output_dim_ * sizeof(float));
    cudaMemcpy(biases_[this->hidden_dim_.size()], bias[this->hidden_dim_.size()].data(), this->output_dim_ * sizeof(float), cudaMemcpyHostToDevice);
}
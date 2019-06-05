#include <cuda.h>
#include <cublas_v2.h>
#include <omp.h>

#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <ctime>

using namespace std;

#include "error_helper.hpp"

#define CUDA_CHECK_ERROR

#define CudaSafeCall(err) __CudaSafeCall(err, __FILE__, __LINE__)
#define CudaCheckError() __CudaCheckError(__FILE__, __LINE__)

__host__ void __CudaSafeCall(cudaError err, const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

__host__ void __CudaCheckError(const char *file, const int line) {
#ifdef CUDA_CHECK_ERROR
    cudaError err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line,
                cudaGetErrorString(err));
        exit(-1);
    }
#endif
}

const int layers[] = {784, 128, 128, 10};
const int num_layers = 4;
const int num_classes = 10;
const int num_epochs = 20;
const float learning_rate = 1e-1;
const int batch_size = 100;

float *weights;
float *bias;
int num_weights, num_bias;

float *zs;
float *as;
float *p;
float *y;

float *dJ_dp;
float *dp_da;
float *dJ_da;
float *dJ_dz;
float *dJ_dW;
float *dJ_db;

const int num_train_data = 60000;
const int num_test_data = 10000;
vector<int *> train_data;
vector<int *> x_test;
vector<int> y_test;

// for cublas
cublasHandle_t cu_handle;
const float alpha = 1.0f;
const float beta = 0.0f;

void ReadDataset(const char *training_file, const char *testing_file)
{
    FILE *train_file = fopen(training_file, "r");
    FILE *test_file = fopen(testing_file, "r");

    train_data = vector<int *>(num_train_data);
    for (int i = 0; i < num_train_data; i++) {
        train_data[i] = new int[layers[0] + 1];
        fscanf(train_file, "%d", &train_data[i][0]);
        for (int j = 1; j < layers[0] + 1; j++)
            fscanf(train_file, ",%d", &train_data[i][j]);
    }

    x_test = vector<int *>(num_test_data);
    y_test = vector<int>(num_test_data);
    for (int i = 0; i < num_test_data; i++) {
         x_test[i] = new int[layers[0]];
        fscanf(test_file, "%d", &y_test[i]);
        for (int j = 0; j < layers[0]; j++)
            fscanf(test_file, ",%d", &x_test[i][j]);
    }

    fclose(train_file);
    fclose(test_file);
}

void InitializeWeights()
{
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 1.0);

    num_weights = 0;
    num_bias = 0;
    for (int n = 1; n < num_layers; n++) {
        num_weights += layers[n - 1] * layers[n];
        num_bias += layers[n];
    }

    float *ws  = new float[num_weights];
    for (int i = 0; i < num_weights; i++)
        ws[i] = distribution(generator);

    float *bs = new float[num_bias];
    for (int i = 0; i < num_bias; i++)
        bs[i] = distribution(generator);

    CudaSafeCall(cudaMalloc(&weights, num_weights * sizeof(float)));
    CudaSafeCall(cudaMalloc(&bias, num_bias * sizeof(float)));

    CudaSafeCall(cudaMemcpy(weights, ws, num_weights * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(bias, bs, num_bias * sizeof(float), cudaMemcpyHostToDevice));
}

void Init()
{
    int num_activations = 0;
    for (int i = 0; i < num_layers; i++)
        num_activations += batch_size * layers[i];

    int num_weights = 0;
    int num_bias = 0;
    for (int n = 1; n < num_layers; n++) {
        num_weights += layers[n - 1] * layers[n];
        num_bias += layers[n];
    }

    CudaSafeCall(cudaMalloc(&zs, num_activations * sizeof(float)));
    CudaSafeCall(cudaMalloc(&as, num_activations * sizeof(float)));
    CudaSafeCall(cudaMalloc(&p, batch_size * num_classes * sizeof(float)));
    CudaSafeCall(cudaMalloc(&y, batch_size * num_classes * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dJ_dp, batch_size * num_classes * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dp_da, batch_size * num_classes * num_classes * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dJ_da, num_activations * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dJ_dz, num_activations * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dJ_dW, num_weights * sizeof(float)));
    CudaSafeCall(cudaMalloc(&dJ_db, num_bias * sizeof(float)));
}

__device__ float ActivationFunction(const float z)
{
    return 1.0 / (1.0 + expf(-z));
}

__device__ float ActivationDerivative(const float a)
{
    return a * (1.0 - a);
}

__global__ void Softmax(float *p, const float *a, const int batch_size, const int num_classes)
{
    const int thread_id = blockIdx.x * blockDim.x + threadIdx.x;

    if (thread_id < batch_size) {
        int offset = thread_id * num_classes;

        float exp_sum = 0.0;
        for (int i = 0; i < num_classes; i++) {
            float e = expf(*(a + offset + i));
            *(p + offset + i) = e;
            exp_sum += e;
        }
        for (int i = 0; i < num_classes; i++)
            *(p + offset + i) /= exp_sum;
    }
}

__global__ void BiasActivation(float *z, float *a, const float *b)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    *(z + offset) += *(b + threadIdx.x);
    *(a + offset) = ActivationFunction(*(z + offset));
}

void Forward()
{
    int weights_offset = 0;
    int bias_offset = 0;
    int as_offset = 0;
    int zs_offset = batch_size * layers[0];

    for (int n = 1; n < num_layers; n++) {
        error_check(cublasSgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, layers[n], batch_size, layers[n - 1],
                    &alpha, weights + weights_offset, layers[n], as + as_offset, layers[n - 1],
                    &beta, zs + zs_offset, layers[n]));

        as_offset += batch_size * layers[n - 1];

        BiasActivation <<< batch_size, layers[n] >>> (zs + zs_offset, as + as_offset, bias + bias_offset);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        weights_offset += layers[n - 1] * layers[n];
        bias_offset += layers[n];
        zs_offset += batch_size * layers[n];
    }

    Softmax <<< (batch_size - 1) / 32 + 1, 32 >>> (p, as + as_offset, batch_size, num_classes);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
}

float Loss(float *h_p, const float *y_batch)
{
    CudaSafeCall(cudaMemcpy(h_p, p, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost));

    float sum = 0.0;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            int idx = i * num_classes + j;
            if (h_p[idx] > 0)
                sum += -y_batch[idx] * log(h_p[idx]);
        }
    }

    return sum / (batch_size * num_classes);
}

int Accuracy(float *h_p, const int offset)
{
    CudaSafeCall(cudaMemcpy(h_p, p, batch_size * num_classes * sizeof(float), cudaMemcpyDeviceToHost));

    int acc_cnt = 0;
    for (int i = 0; i < batch_size; i++) {
        int max_label = -1;
        float max_prob = 0.0;
        for (int j = 0; j < num_classes; j++) {
            int idx = i * num_classes + j;
            if (h_p[idx] > max_prob) {
                max_label = j;
                max_prob = h_p[idx];
            }
        }

        if (max_label == y_test[offset + i])
            acc_cnt++;
    }

    return acc_cnt;
}

__global__ void dJdaChainRules(float *dJdp, float *dpda, const float *y, const float *p) 
{
    // block: batch, thread: class
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_classes = blockDim.x;

    dJdp[offset] = -y[offset] / p[offset];

    for (int i = 0; i < num_classes; i++) {
        if (i == threadIdx.x)
            dpda[offset * num_classes + i] = p[offset] - p[offset] * p[offset];
        else
            dpda[offset * num_classes + i] = -p[offset] * p[blockIdx.x * num_classes + i];
    }
}

__global__ void dJdzDerivative(float *dJdz, const float *dJda, const float *a)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    *(dJdz + offset) = *(dJda + offset) * ActivationDerivative(*(a + offset));
}

__global__ void DivideBatchSize(float *dJdW, const int batch_size)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    *(dJdW + offset) /= batch_size;
}

__global__ void BiasDerivative(float *dJdb, const float *dJdz, const int num_neurons, const int batch_size)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;

    if (offset < num_neurons) {
        float sum = 0.0;
        for (int i = 0; i < batch_size; i++)
            sum += *(dJdz + i * num_neurons + offset);
        *(dJdb + offset) = sum / batch_size;
    }
}

__global__ void WeightBiasDerivative(float *dJdW, float *dJdb, const float *dJdz, const int batch_size)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_neurons = blockDim.x;

    *(dJdW + offset) /= batch_size;

    if (blockIdx.x == 0) {
        float sum = 0.0;
        for (int i = 0; i < batch_size; i++)
            sum += *(dJdz + i * num_neurons + threadIdx.x);
        *(dJdb + threadIdx.x) = sum / batch_size;
    }
}

void Backpropagation()
{
    dJdaChainRules <<< batch_size, num_classes >>> (dJ_dp, dp_da, y, p);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());

    int dJda_offset = 0;
    for (int i = 0; i < num_layers - 1; i++)
        dJda_offset += batch_size * layers[i];

    for (int i = 0; i < batch_size; i++) {
        int offset = i * num_classes;
        error_check(cublasSgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_N, 1, num_classes, num_classes,
                    &alpha, dJ_dp + offset, 1, dp_da + offset * num_classes, num_classes,
                    &beta, dJ_da + dJda_offset + offset, 1));
    }

    int dJdz_offset = dJda_offset;
    int dJdW_offset = 0;
    int dJdb_offset = 0;
    for (int i = 1; i < num_layers - 1; i++) {
        dJdW_offset += layers[i - 1] * layers[i];
        dJdb_offset += layers[i];
    }

    for (int n = num_layers - 1; n >= 1; n--) {
        dJdzDerivative <<< batch_size, layers[n] >>> (dJ_dz + dJdz_offset, dJ_da + dJda_offset, as + dJda_offset);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        dJda_offset -= batch_size * layers[n - 1];

        error_check(cublasSgemm(cu_handle, CUBLAS_OP_N, CUBLAS_OP_T, layers[n], layers[n - 1], batch_size,
                    &alpha, dJ_dz + dJdz_offset, layers[n], as + dJda_offset, layers[n - 1],
                    &beta, dJ_dW + dJdW_offset, layers[n]));
/*
        DivideBatchSize <<< layers[n - 1], layers[n] >>> (dJ_dW + dJdW_offset, batch_size);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());

        BiasDerivative <<< (layers[n] - 1) / 32 + 1, 32 >>> (dJ_db + dJdb_offset, dJ_dz + dJdz_offset, layers[n], batch_size);
        CudaCheckError();
        CudaSafeCall(cudaDeviceSynchronize());
*/
        WeightBiasDerivative <<< layers[n - 1], layers[n] >>> (dJ_dW + dJdW_offset, dJ_db + dJdb_offset, dJ_dz + dJdz_offset, batch_size);

        error_check(cublasSgemm(cu_handle, CUBLAS_OP_T, CUBLAS_OP_N, layers[n - 1], batch_size, layers[n],
                    &alpha, weights + dJdW_offset, layers[n], dJ_dz + dJdz_offset, layers[n - 1],
                    &beta, dJ_da + dJda_offset, layers[n - 1]));

        dJdz_offset -= batch_size * layers[n -1];
        if (n != 1) {
            dJdW_offset -= layers[n - 2] * layers[n - 1];
            dJdb_offset -= layers[n - 1];
        }
    }
}

__global__ void UpdateWeights(float *w, const float *dJdW, const float learning_rate, const int num_weights)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < num_weights)
        *(w + offset) -= learning_rate * (*(dJdW + offset));
}

__global__ void UpdateBias(float *b, const float *dJdb, const float learning_rate, const int num_bias)
{
    const int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset < num_bias)
        *(b + offset) -= learning_rate * (*(dJdb + offset));
}

__global__ void UpdateGradientDevice(float *w, float *b, const float *dJdW, float *dJdb, const float learning_rate)
{
}

void UpdateGradient()
{
    UpdateWeights <<< (num_weights - 1) / 32 + 1, 32 >>> (weights, dJ_dW, learning_rate, num_weights);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
    UpdateBias <<< (num_bias - 1) / 32 + 1, 32 >>> (bias, dJ_db, learning_rate, num_bias);
    CudaCheckError();
    CudaSafeCall(cudaDeviceSynchronize());
}

void PrepareBatchData(float *x_batch, float *y_batch, const int batch)
{
    const int offset = batch * batch_size;

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++)
            y_batch[i * num_classes + j] = 0;

        y_batch[i * num_classes + train_data[offset + i][0]] = 1;
        for (int j = 0; j < layers[0]; j++)
            x_batch[i * layers[0] + j] = train_data[offset + i][j + 1];
    }

    CudaSafeCall(cudaMemcpy(as, x_batch, batch_size * layers[0] * sizeof(float), cudaMemcpyHostToDevice));
    CudaSafeCall(cudaMemcpy(y, y_batch, batch_size * num_classes * sizeof(float), cudaMemcpyHostToDevice));
}

void Train()
{
    int num_batches = num_train_data / batch_size;
    float *x_batch = new float[batch_size * layers[0]];
    float *y_batch = new float[batch_size * num_classes];
    float *h_p = new float[batch_size * num_classes];

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        random_shuffle(train_data.begin(), train_data.end());

        float epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            PrepareBatchData(x_batch, y_batch, b);
            Forward();
            epoch_loss += Loss(h_p, y_batch);
            Backpropagation();
            UpdateGradient();
        }

        epoch_loss /= num_batches;
        printf("Epoch %3d, loss=%.6f\n", epoch, epoch_loss);
    }
}

void Test()
{
    int num_batches = num_test_data / batch_size;
    float *x_batch = new float[batch_size * layers[0]];
    float *h_p = new float[batch_size * num_classes];

    int acc = 0;
    for (int b = 0; b < num_batches; b++) {
        int offset = b * batch_size;
        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < layers[0]; j++)
                x_batch[i * layers[0] + j] = x_test[offset + i][j];

        CudaSafeCall(cudaMemcpy(as, x_batch, batch_size * layers[0] * sizeof(float), cudaMemcpyHostToDevice));

        Forward();
        acc += Accuracy(h_p, offset);
    }

    printf("Test Accuracy: %.6f\n", (float)acc / num_test_data);
}

int main(int argc, char *argv[])
{
    assert(argc == 3 && "Require 2 arguments.");
    srand(time(0));
    cublasCreate(&cu_handle);

    ReadDataset(argv[1], argv[2]);

    InitializeWeights();
    Init();

    Train();

    Test();

    cublasDestroy(cu_handle);

    return 0;
}

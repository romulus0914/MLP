#include <iostream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <cmath>
#include <ctime>

using namespace std;

vector<int> neurons;
int num_layers;
vector<float **> weights;
vector<float *> bias;

vector<float **> zs;
vector<float **> as;
float **p;

float **dJ_dp;
vector<float **> dp_da;
vector<float **> dJ_da;
vector<float **> dJ_dz;
vector<float **> dJ_dW;
vector<float *> dJ_db;

int num_classes;
int num_epochs;
float learning_rate;
int batch_size;

const int num_train_data = 60000;
const int num_test_data = 10000;
vector<vector<int>> train_data;
vector<vector<int>> x_test;
vector<int> y_test;

void ReadDataset(const char *training_file, const char *testing_file)
{
    FILE *train_file = fopen(training_file, "r");
    FILE *test_file = fopen(testing_file, "r");

    train_data = vector<vector<int>>(num_train_data);
    for (int i = 0; i < num_train_data; i++) {
        train_data[i] = vector<int>(neurons[0] + 1);

        fscanf(train_file, "%d", &train_data[i][0]);
        for (int j = 1; j < neurons[0] + 1; j++)
            fscanf(train_file, ",%d", &train_data[i][j]);
    }

    x_test = vector<vector<int>>(num_test_data);
    y_test = vector<int>(num_test_data);
    for (int i = 0; i < num_test_data; i++) {
        x_test[i] = vector<int>(neurons[0]);

        fscanf(test_file, "%d", &y_test[i]);
        for (int j = 0; j < neurons[0]; j++)
            fscanf(test_file, ",%d", &x_test[i][j]);
    }

    fclose(train_file);
    fclose(test_file);
}

void InitializeWeights()
{
    default_random_engine generator;
    normal_distribution<float> distribution(0.0, 1.0);

    weights = vector<float **>(num_layers);
    bias = vector<float *>(num_layers);
    for (int i = 1; i < num_layers; i++) {
        float **w = new float*[neurons[i - 1]];
        for (int j = 0; j < neurons[i - 1]; j++) {
            w[j] = new float[neurons[i]];
            for (int k = 0; k < neurons[i]; k++)
                w[j][k] = distribution(generator);
        }
        weights[i] = w;

        float *b = new float[neurons[i]];
        for (int j = 0; j < neurons[i]; j++)
            b[j] = distribution(generator);
        bias[i] = b;
    }
}

void Init()
{
    zs = vector<float **>(num_layers);
    for (int i = 0; i < num_layers; i++) {
        float **z = new float*[batch_size];
        for (int j = 0; j < batch_size; j++)
            z[j] = new float[neurons[i]];
        zs[i] = z;
    }

    as = vector<float **>(num_layers);
    for (int i = 0; i < num_layers; i++) {
        float **a = new float*[batch_size];
        for (int j = 0; j < batch_size; j++)
            a[j] = new float[neurons[i]];
        as[i] = a;
    }

    p = new float*[batch_size];
    for (int i = 0; i < batch_size; i++)
        p[i] = new float[num_classes];

    dJ_dp = new float*[batch_size];
    for (int i = 0; i < batch_size; i++)
        dJ_dp[i] = new float[num_classes];

    dp_da = vector<float **>(batch_size);
    for (int i = 0; i < batch_size; i++) {
        float **_dp_da = new float*[num_classes];
        for (int j = 0; j < num_classes; j++)
            _dp_da[j] = new float[num_classes];
        dp_da[i] = _dp_da;
    }

    dJ_da = vector<float **>(num_layers);
    for (int i = 0; i < num_layers; i++) {
        float **_dJ_da = new float*[batch_size];
        for (int j = 0; j < batch_size; j++)
            _dJ_da[j] = new float[neurons[i]];
        dJ_da[i] = _dJ_da;
    }

    dJ_dz = vector<float **>(num_layers);
    for (int i = 0; i < num_layers; i++) {
        float **_dJ_dz = new float*[batch_size];
        for (int j = 0; j < batch_size; j++)
            _dJ_dz[j] = new float[neurons[i]];
        dJ_dz[i] = _dJ_dz;
    }

    dJ_dW = vector<float **>(num_layers);
    dJ_db = vector<float *>(num_layers);
    for (int i = 1; i < num_layers; i++) {
        float **_dJ_dW = new float*[neurons[i - 1]];
        for (int j = 0; j < neurons[i - 1]; j++)
            _dJ_dW[j] = new float[neurons[i]];
        dJ_dW[i] = _dJ_dW;

        float *_dJ_db = new float[neurons[i]];
        dJ_db[i] = _dJ_db;
    }
}

float ActivationFunction(const float z)
{
    return 1.0 / (1.0 + exp(-z));
}

float ActivationDerivative(const float a)
{
    return a * (1.0 - a);
}

void Softmax(float ** const a)
{
    for (int i = 0; i < batch_size; i++) {
        float exp_sum = 0.0;
        for (int j = 0; j < num_classes; j++) {
            p[i][j] = exp(a[i][j]);
            exp_sum += p[i][j];
        }
        for (int j = 0; j < num_classes; j++)
            p[i][j] /= exp_sum;
    }
}

void Forward()
{
    for (int n = 1; n < num_layers; n++) {
        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < neurons[n]; j++) {
                float sum = 0.0;
                for (int k = 0; k < neurons[n - 1]; k++)
                    sum += as[n - 1][i][k] * weights[n][k][j];
                zs[n][i][j] = sum + bias[n][j];
                as[n][i][j] = ActivationFunction(zs[n][i][j]);
            }
        }
    }

    Softmax(as[num_layers - 1]);
}

float Loss(float ** const y)
{
    float sum = 0.0;
    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            if (p[i][j] > 0)
                sum += -log(p[i][j]) * y[i][j];
        }
    }

    return sum / (batch_size * num_classes);
}

int Accuracy(const int start_idx)
{
    int acc_cnt = 0;
    for (int i = 0; i < batch_size; i++) {
        int max_label = -1;
        float max_prob = 0.0;
        for (int j = 0; j < num_classes; j++) {
            if (p[i][j] > max_prob) {
                max_label = j;
                max_prob = p[i][j];
            }
        }

        if (max_label == y_test[start_idx + i])
            acc_cnt++;
    }

    return acc_cnt;
}

void Backpropagation(float ** const y)
{
    for (int i = 0; i < batch_size; i++)
        for (int j = 0; j < num_classes; j++)
            dJ_dp[i][j] = -y[i][j] / p[i][j];

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            for (int k = 0; k < num_classes; k++) {
                if (j == k)
                    dp_da[i][j][k] = p[i][j] - p[i][j] * p[i][j];
                else
                    dp_da[i][j][k] = -p[i][j] * p[i][k];
            }
        }
    }

    for (int i = 0; i < batch_size; i++) {
        for (int j = 0; j < num_classes; j++) {
            float sum = 0.0;
            for (int k = 0; k < num_classes; k++)
                sum += dp_da[i][j][k] * dJ_dp[i][k];
            dJ_da[num_layers - 1][i][j] = sum;
        }
    }

    for (int n = num_layers - 1; n >= 1; n--) {
        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < neurons[n]; j++)
                dJ_dz[n][i][j] = dJ_da[n][i][j] * ActivationDerivative(as[n][i][j]);

        for (int i = 0; i < neurons[n - 1]; i++) {
            for (int j = 0; j < neurons[n]; j++) {
                float sum = 0.0;
                for (int k = 0; k < batch_size; k++)
                    sum += as[n - 1][k][i] * dJ_dz[n][k][j];
                dJ_dW[n][i][j] = sum / batch_size;
            }
        }

        for (int i = 0; i < neurons[n]; i++) {
            float sum = 0.0;
            for (int j = 0; j < batch_size; j++)
                sum += dJ_dz[n][j][i];
            dJ_db[n][i] = sum / batch_size;
        }

        for (int i = 0; i < batch_size; i++) {
            for (int j = 0; j < neurons[n - 1]; j++) {
                float sum = 0.0;
                for (int k = 0; k < neurons[n]; k++)
                    sum += dJ_dz[n][i][k] * weights[n][j][k];
                dJ_da[n - 1][i][j] = sum;
            }
        }
    }
}

void UpdateGradient()
{
    for (int n = 1; n < num_layers; n++) {
        for (int i = 0; i < neurons[n - 1]; i++)
            for (int j = 0; j < neurons[n]; j++)
                weights[n][i][j] -= learning_rate * dJ_dW[n][i][j];
        for (int i = 0; i < neurons[n]; i++)
            bias[n][i] -= learning_rate * dJ_db[n][i];
    }
}

void Train()
{
    int num_batches = num_train_data / batch_size;

    for (int epoch = 1; epoch <= num_epochs; epoch++) {
        random_shuffle(train_data.begin(), train_data.end());

        float epoch_loss = 0.0;
        for (int b = 0; b < num_batches; b++) {
            float **y_batch = new float*[batch_size];
            for (int i = 0; i < batch_size; i++)
                y_batch[i] = new float[num_classes];

            int start_idx = b * batch_size;
            for (int i = 0; i < batch_size; i++) {
                y_batch[i][train_data[start_idx + i][0]] = 1;
                for (int j = 0; j < neurons[0]; j++)
                    as[0][i][j] = train_data[start_idx + i][j + 1];
            }

            Forward();
            epoch_loss += Loss(y_batch);
            Backpropagation(y_batch);
            UpdateGradient();
        }

        epoch_loss /= num_batches;
        printf("Epoch %3d, loss=%.6f\n", epoch, epoch_loss);
    }
}

void Test()
{
    int num_batches = num_test_data / batch_size;

    int acc = 0;
    for (int b = 0; b < num_batches; b++) {
        int start_idx = b * batch_size;
        for (int i = 0; i < batch_size; i++)
            for (int j = 0; j < neurons[0]; j++)
                as[0][i][j] = x_test[start_idx + i][j];

        Forward();
        acc += Accuracy(start_idx);
    }

    printf("Test Accuracy: %.6f\n", (float)acc / num_test_data);
}

int main(int argc, char *argv[])
{
    assert(argc == 3 && "Require 2 arguments.");

    int layers[] = {784, 128, 128, 10};
    num_layers = 4;
    neurons = vector<int>(layers, layers + num_layers);

    num_classes = layers[num_layers - 1];
    num_epochs = 20;
    learning_rate = 1e-1;
    batch_size = 100;

    srand(time(0));

    ReadDataset(argv[1], argv[2]);

    InitializeWeights();
    Init();

    Train();

    Test();

    return 0;
}

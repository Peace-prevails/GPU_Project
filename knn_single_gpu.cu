#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include<time.h>
#include <float.h>

#define NUM_FEATURES 4
#define MAX_DATA_SIZE 30001
#define NUM_CLASSES 3
#define TRAIN_TEST_SPLIT 0.8
double train_features[MAX_DATA_SIZE * NUM_FEATURES];
int train_labels[MAX_DATA_SIZE];
double test_features[MAX_DATA_SIZE * NUM_FEATURES];
int test_labels[MAX_DATA_SIZE];
// int labelToInt(const char* label) {
//     if (strcmp(label, "Setosa") == 0) return 0;
//     if (strcmp(label, "Virginica") == 0) return 1;
//     if (strcmp(label, "Versicolor") == 0) return 2;
//     printf("Unrecognized label: %s\n", label);
//     return -1;
// }
int labelToInt(const char* label) {
    if (strcmp(label, "low") == 0) return 0;
    if (strcmp(label, "medium") == 0) return 1;
    if (strcmp(label, "high") == 0) return 2;
    printf("Unrecognized label: %s\n", label);
    return -1;
}

int loadData(const char* filename, double* features, int* labels) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        perror("Error opening file");
        return -1;
    }

    char line[1024];
    int count = 0;
    fgets(line, sizeof(line), file); // 跳过标题行

    while (fgets(line, sizeof(line), file) && count < MAX_DATA_SIZE) {
        char* token = strtok(line, ",");
        for (int i = 0; i < NUM_FEATURES; i++) {
            if (token == NULL) {
                fprintf(stderr, "Error in data format\n");
                fclose(file);
                return -1;
            }
            features[count * NUM_FEATURES + i] = atof(token);
            token = strtok(NULL, ",");
        }

        if (token == NULL) {
            fprintf(stderr, "Error in data format\n");
            fclose(file);
            return -1;
        }
        token[strcspn(token, "\n")] = 0; // 移除换行符
        labels[count] = labelToInt(token);
        count++;
    }

    fclose(file);
    return count;
}

__global__ void computeDistances(double *train_features, double *test_features, double *distances, int train_size, int test_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < test_size) {
        for (int i = 0; i < train_size; i++) {
            double dist = 0.0;
            for (int j = 0; j < NUM_FEATURES; j++) {
                double diff = train_features[i * NUM_FEATURES + j] - test_features[idx * NUM_FEATURES + j];
                dist += diff * diff;
            }
            
            distances[idx * train_size + i] = sqrt(dist);
        }
    }
}

int majorityVote(double *distances, int *train_labels, int train_size, int idx, int k) {
    int votes[NUM_CLASSES] = {0};
    double temp_distances[train_size];

    // 复制对应测试样本的距离
    for (int i = 0; i < train_size; i++) {
        temp_distances[i] = distances[idx * train_size + i];
    }

    for (int i = 0; i < k; i++) {
        double min_dist = DBL_MAX;
        int min_index = -1;
        for (int j = 0; j < train_size; j++) {
            if (temp_distances[j] < min_dist) {
                min_dist = temp_distances[j];
                min_index = j;
            }
        }

        if (min_index != -1) {
            votes[train_labels[min_index]]++;
            temp_distances[min_index] = DBL_MAX; // 将已选择的最小距离设置为最大值，以便在下一轮中忽略它
        }
    }

    int max_votes = 0, predicted_label = -1;
    for (int i = 0; i < NUM_CLASSES; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            predicted_label = i;
        }
    }
    return predicted_label;
}


int main() {
    // Load data into flat arrays
    int total_size = loadData("synthetic_knn_dataset.csv", train_features, train_labels);
    if (total_size == -1) return 1;
    
    // printf("First 5 samples of train_features:\n");
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < NUM_FEATURES; ++j) {
    //         printf("%f ", train_features[i * NUM_FEATURES + j]);
    //     }
    //     printf("\n");
    // }

    // // 打印前5个样本的标签
    // printf("First 5 labels of train_labels:\n");
    // for (int i = 0; i < 5; ++i) {
    //     printf("%d\n", train_labels[i]);
    // }

    int train_size = (int)(total_size * TRAIN_TEST_SPLIT);
    int test_size = total_size - train_size;
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            test_features[i * NUM_FEATURES + j] = train_features[(train_size + i) * NUM_FEATURES + j];
        }
        test_labels[i] = train_labels[train_size + i];
    }
    double time_taken2;
    double time_taken1,time_taken3;
    clock_t start, end;
    start = clock();
    // Allocate memory on GPU
    
    double *d_train_features, *d_test_features, *d_distances;
    cudaMalloc(&d_train_features, train_size * NUM_FEATURES * sizeof(double));
    cudaMalloc(&d_test_features, test_size * NUM_FEATURES * sizeof(double));
    cudaMalloc(&d_distances, train_size * test_size * sizeof(double));

    // Copy training and testing data to GPU
    cudaMemcpy(d_train_features, train_features, train_size * NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_features, test_features, test_size * NUM_FEATURES * sizeof(double), cudaMemcpyHostToDevice);

    // Setup kernel execution parameters and launch kernel
    
    dim3 blockSize(128);
    dim3 gridSize((test_size + blockSize.x - 1) / blockSize.x);
    computeDistances<<<gridSize, blockSize>>>(d_train_features, d_test_features, d_distances, train_size, test_size);
    end = clock();
    cudaDeviceSynchronize();
    time_taken1 = ((double)(end - start))/ CLOCKS_PER_SEC;
    // Allocate memory on host for the distances and copy from device to host
    double *h_distances = (double*)malloc(train_size * test_size * sizeof(double));
    start = clock();
    cudaMemcpy(h_distances, d_distances, train_size * test_size * sizeof(double), cudaMemcpyDeviceToHost);
    end = clock();
    time_taken2 = ((double)(end - start))/ CLOCKS_PER_SEC;
    
    // Here is the logic for the k nearest neighbors and majority voting
    double best_accuracy = 0.0;
    int best_k = 1;
    start=clock();
    // 尝试不同的k值
    for (int k = 1; k <= 5; k++) {
        int correct_predictions = 0;

        // 对于测试集中的每个点
        for (int i = 0; i < test_size; i++) {
            int predicted_label = majorityVote(h_distances, train_labels, train_size, i, k);
            if (predicted_label == test_labels[i]) {
                correct_predictions++;
            }
        }

        double accuracy = (double)correct_predictions / test_size;
        // printf("k = %d, Accuracy: %f\n", k, accuracy);

        // 更新最佳k值
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_k = k;
        }
    }
   end=clock();
   time_taken3 = ((double)(end - start))/ CLOCKS_PER_SEC;
   printf("Time taken for copying data to device,launch kernels: %lf\n", time_taken1);
   printf("Time taken for copying data back to the host: %lf\n", time_taken2);
   printf("Total overhead: %lf\n", time_taken1+time_taken2);
   printf("Total time: %lf\n", time_taken1+time_taken2+time_taken3);
   printf("Best k value: %d with accuracy: %f\n", best_k, best_accuracy);


    // Free GPU memory
    cudaFree(d_train_features);
    cudaFree(d_test_features);
    cudaFree(d_distances);

    // Free host memory
    // free(h_distances);

    return 0;
}

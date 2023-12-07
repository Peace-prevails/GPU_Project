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
int majorityVote(int* labels, double* distances, int size, int k, int test_idx) {
    int votes[NUM_CLASSES] = {0};
    double temp_distances[size];

    // 复制对应测试样本的距离
    for (int i = 0; i < size; i++) {
        temp_distances[i] = distances[test_idx * size + i];
    }

    for (int i = 0; i < k; i++) {
        double min_dist = DBL_MAX;
        int min_index = -1;
        for (int j = 0; j < size; j++) {
            if (temp_distances[j] < min_dist) {
                min_dist = temp_distances[j];
                min_index = j;
            }
        }

        if (min_index != -1) {
            votes[labels[min_index]]++;
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


double euclideanDistance(const double* a, const double* b) {
    double sum = 0.0;
    for (int i = 0; i < NUM_FEATURES; i++) {
        sum += (a[i] - b[i]) * (a[i] - b[i]);
    }
    return sqrt(sum);
}

int main() {
    double *all_features = malloc(MAX_DATA_SIZE * NUM_FEATURES * sizeof(double));
    int *all_labels = malloc(MAX_DATA_SIZE * sizeof(int));
    double *train_features = malloc(MAX_DATA_SIZE * NUM_FEATURES * sizeof(double));
    int *train_labels = malloc(MAX_DATA_SIZE * sizeof(int));
    double *test_features = malloc(MAX_DATA_SIZE * NUM_FEATURES * sizeof(double));
    int *test_labels = malloc(MAX_DATA_SIZE * sizeof(int));
    double *distances = malloc(MAX_DATA_SIZE * MAX_DATA_SIZE * sizeof(double));

    if (!all_features || !all_labels || !train_features || !train_labels || !test_features || !test_labels || !distances) {
        printf("Memory allocation failed\n");
        return 1;
    }
    // 加载数据
    int total_size = loadData("synthetic_knn_dataset.csv", all_features, all_labels);
    if (total_size == -1) return 1;
    double time_taken1,time_taken2;
    clock_t start, end;
    // 分割数据为训练集和测试集
    int train_size = (int)(total_size * TRAIN_TEST_SPLIT);
    int test_size = total_size - train_size;
    for (int i = 0; i < train_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            train_features[i * NUM_FEATURES + j] = all_features[i * NUM_FEATURES + j];
        }
        train_labels[i] = all_labels[i];
    }
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < NUM_FEATURES; j++) {
            test_features[i * NUM_FEATURES + j] = all_features[(train_size + i) * NUM_FEATURES + j];
        }
        test_labels[i] = all_labels[train_size + i];
    }
    start=clock();
     for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < train_size; j++) {
            distances[i * train_size + j] = euclideanDistance(&test_features[i * NUM_FEATURES], &train_features[j * NUM_FEATURES]);
        }
    }
    end=clock();
    time_taken1 = ((double)(end - start))/ CLOCKS_PER_SEC;
    start=clock();
    // 测试不同的 k 值
    int best_k = 1;
    double best_accuracy = 0;
    for (int k = 1; k <= 5; k++) {
        int correct = 0;
        for (int i = 0; i < test_size; i++) {
            int predicted_label = majorityVote(train_labels, distances, train_size, k, i);
            if (predicted_label == test_labels[i]) {
                correct++;
            }
        }
        double accuracy = (double)correct / test_size;
        if (accuracy > best_accuracy) {
            best_accuracy = accuracy;
            best_k = k;
        }
    }

    end=clock();
    time_taken2 = ((double)(end - start))/ CLOCKS_PER_SEC;
    printf("Time taken for calculating the distance: %lf\n", time_taken1);
    printf("Time taken for selecting k: %lf\n", time_taken2);
    printf("Total time taken: %lf\n", time_taken1+time_taken2);
    printf("Best k value: %d with accuracy: %f\n", best_k, best_accuracy);
    free(all_features);
    free(all_labels);
    free(train_features);
    free(train_labels);
    free(test_features);
    free(test_labels);
    free(distances);
    return 0;
}
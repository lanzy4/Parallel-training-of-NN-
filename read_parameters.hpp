#pragma once

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include "cblas.h"
#include <mpi.h>

#include "network.hpp"


struct DatasetParameters {
    int test_amount;
    int train_amount;
    int img_size;
    int channels_n;
    int classes_number;  
};


int reverseInt (int i);


void read_mnist_data(float*** &result, std::string filename);
void read_mnist_labels(int* &result, std::string filename);


void read_cifar10(float*** &x_train, int* &y_train, float*** &x_test, int* &y_test);
void read_cifar100(float*** &x_train, int* &y_train, float*** &x_test, int* &y_test);


void read_dataset(float*** &x_train, int* &y_train, float*** &x_test,
                     int* &y_test, DatasetParameters &params, std::string dataset_name);
void free_dataset(float*** &x_train, int* &y_train, float*** &x_test,
                     int* &y_test, DatasetParameters &params);


void model_from_file(Network &model, std::string filename);



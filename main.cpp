#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include "cblas.h"
#include <mpi.h>

//test test test

#include "network.hpp"
#include "read_parameters.hpp"




void print_weights(Network &model) {
    const auto& wc1 = model.conv_weights[0];
    const auto& wc2 = model.conv_weights[1];
    const auto& w1 = model.fc_weights[0];
    const auto& w2 = model.fc_weights[1];

    std::cout << std::endl << "WEIGHTS:" << std::endl;

    std::cout << "wc1: " << std::endl;
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << wc1[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "wc2: " << std::endl;
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << wc2[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "w1: " << std::endl;
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << w1[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "w2: " << std::endl;
    for(int i = 0; i < 5; i++) {
        for(int j = 0; j < 10; j++) {
            std::cout << w2[i][j] << " ";
        }
        std::cout << std::endl;
    }

    std::cout << std::endl;
}



float test_model(Network &model, float*** x_test, int* y_test,
                 int img_size_x, int img_size_y, int channels_n, int test_size, int classes_amount) {

    int *y_pred = new int[test_size];
    float *res = new float[classes_amount];
    int right_predictions = 0;
    for(int i = 0; i < test_size; i++) {
        model.predict(x_test[i], img_size_x, img_size_y, channels_n, res);

        float max = res[0];
        int max_ind = 0;
        for(int j = 0; j < classes_amount; j++) {
            if (res[j] > max) {
                max = res[j];
                max_ind = j;
            }
        }

        y_pred[i] = max_ind;

        if (y_pred[i] == y_test[i]) {
            right_predictions++;
        }
    }
    delete[] res;
    delete[] y_pred;
    
    float accuracy = (float)right_predictions / test_size;
    return accuracy;
}




int main(int argc, char** argv) {

    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    DatasetParameters dataset_params;
    float ***x_train;
    float ***x_test;
    int *y_train;
    int* y_test;
    read_dataset(x_train, y_train, x_test, y_test, dataset_params, "CIFAR10");
    

    int iter_num = 6;
    float learning_rate = 0.0001;
    int minibatch_size = 100;

    int communication_freq = 1;
    int gossip_flag = 0;
    int sparsification_flag = 0;
    double sparsification_threshold = 0.0000001;

    if(argc > 1) {
        iter_num = std::atoi(argv[1]);
    }
    if(argc > 2) {
        learning_rate = std::atof(argv[2]);
    }
    if(argc > 3) {
        minibatch_size = std::atoi(argv[3]);
    }
    if(argc > 4) {
        communication_freq = std::atoi(argv[4]);
    }
    if(argc > 5) {
        gossip_flag = std::atoi(argv[5]);
    }
    if(argc > 6) {
        sparsification_flag = std::atoi(argv[6]);
    }
    if(argc > 7) {
        sparsification_threshold = std::atof(argv[7]);
    }

    Network model;
    model.set_communication_options(communication_freq, gossip_flag,
                                     sparsification_flag, sparsification_threshold);


    model_from_file(model, "Models/test.txt");


    //print_weights(model);

    double start_time, finish_time;
    double duration = 0;

    for(int i = 0; i < iter_num; i++) {
        start_time = MPI_Wtime();
        model.fit(x_train, y_train, dataset_params.img_size, dataset_params.img_size,
                     dataset_params.channels_n, dataset_params.train_amount, minibatch_size, 1, learning_rate);
        finish_time = MPI_Wtime();
        duration += finish_time - start_time;
        float accuracy = test_model(model, x_test, y_test, dataset_params.img_size,
                                     dataset_params.img_size, dataset_params.channels_n,
                                      dataset_params.test_amount, dataset_params.classes_number);

        if (rank == 0) {
            std::cout << "Iteration " << (i+1) <<" Accuracy: " << accuracy << std::endl;
        }
    }


    double local_synchronization_time = model.synchronization_time;
    double synchronization_time;
    MPI_Reduce(&local_synchronization_time, &synchronization_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_communication_time = model.communication_time;
    double communication_time;
    MPI_Reduce(&local_communication_time, &communication_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::cout << "Training is over" << std::endl;
        std::cout << "Time: " << duration  << " s" << std::endl;
        std::cout << "Synchronization time: " << synchronization_time << std::endl;
        std::cout << "Communication time: " << communication_time << std::endl;

        //print_weights(model);

        float accuracy = test_model(model, x_test, y_test, dataset_params.img_size,
                                     dataset_params.img_size, dataset_params.channels_n,
                                      dataset_params.test_amount, dataset_params.classes_number);
        std::cout << "Accuracy: " << accuracy << std::endl;
    }
    

    free_dataset(x_train, y_train, x_test, y_test, dataset_params);

    MPI_Finalize();

    return 0;
}




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



void save_weights(Network &model, int epoch) {

    std::ofstream file("weights.bin", std::ios::out);

    file.write((char*)&epoch, sizeof(int));

    for(int i = 0; i < model.conv_layers_number; i++) {
        const auto& w = model.conv_weights[i];
        const auto& b = model.conv_bias[i];
        int d1 = model.conv_params[i].maps_out;
        int d2 = model.conv_params[i].kernel_size * model.conv_params[i].kernel_size * model.conv_params[i].maps_in;
        for(int j = 0; j < d1; j++) {
            file.write((char*)w[j], sizeof(float) * d2);
        }
        file.write((char*)b, sizeof(float) * d1);
    }

    for(int i = 0; i < model.fc_layers_number; i++) {
        const auto& w = model.fc_weights[i];
        const auto& b = model.fc_bias[i];
        int d1 = model.fc_params[i].n_in;
        int d2 = model.fc_params[i].n_out;
        for(int j = 0; j < d1; j++) {
            file.write((char*)w[j], sizeof(float) * d2);
        }
        file.write((char*)b, sizeof(float) * d2);
    }

    file.close();
}

void load_weights(Network &model, int &epoch) {
    std::ifstream file("weights.bin", std::ios::in);

    file.read((char*)&epoch, sizeof(int));

    if(epoch == 0) {
        return;
    }

    for(int i = 0; i < model.conv_layers_number; i++) {
        auto& w = model.conv_weights[i];
        auto& b = model.conv_bias[i];
        int d1 = model.conv_params[i].maps_out;
        int d2 = model.conv_params[i].kernel_size * model.conv_params[i].kernel_size * model.conv_params[i].maps_in;
        for(int j = 0; j < d1; j++) {
            file.read((char*)w[j], sizeof(float) * d2);
        }
        file.read((char*)b, sizeof(float) * d1);
    }

    for(int i = 0; i < model.fc_layers_number; i++) {
        auto& w = model.fc_weights[i];
        auto& b = model.fc_bias[i];
        int d1 = model.fc_params[i].n_in;
        int d2 = model.fc_params[i].n_out;
        for(int j = 0; j < d1; j++) {
            file.read((char*)w[j], sizeof(float) * d2);
        }
        file.read((char*)b, sizeof(float) * d2);
    }

    file.close();
}



float test_model(Network &model, std::string dataset_filename, DatasetParameters params) {

    int rank, np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    int y_pred;
    float *res = new float[params.classes_number];

    int y_test;
    float **x_test = new float*[params.channels_n];
    for(int i = 0; i < params.channels_n; i++) {
        x_test[i] = new float[params.img_size * params.img_size];
    }

    std::ifstream data_file(dataset_filename);

    int processed_samples = params.test_amount / np + (rank < params.test_amount % np);
    int s_start = params.test_amount / np * rank + (rank < params.test_amount % np ? rank : params.test_amount % np);

    int bytes_for_image = params.img_size * params.img_size * params.channels_n + 1;
    data_file.seekg(s_start * bytes_for_image, std::ios::beg);
    
    int right_predictions = 0;
    for(int i = 0; i < processed_samples; i++) {

        unsigned char temp=0;
        data_file.read((char*)&temp,sizeof(temp));
        y_test = temp;
        for(int j = 0; j < params.channels_n; j++) {
            for(int k = 0; k < params.img_size * params.img_size; k++) {
                data_file.read((char*)&temp,sizeof(temp));
                float value = (float)temp / params.max_pixel_value;
                value -= params.mean[j];
                value /= params.std[j];
                x_test[j][k] = value;
            }
        }

        model.predict(x_test, params.img_size, params.img_size, params.channels_n, res);

        float max = res[0];
        int max_ind = 0;
        for(int j = 0; j < params.classes_number; j++) {
            if (res[j] > max) {
                max = res[j];
                max_ind = j;
            }
        }

        y_pred = max_ind;

        if (y_pred == y_test) {
            right_predictions++;
        }
    }
    delete[] res;

    for(int i = 0; i < params.channels_n; i++) {
        delete[] x_test[i];
    } 
    delete[] x_test;

    data_file.close();

    int right_predictions_sum = 0;
    MPI_Allreduce(&right_predictions, &right_predictions_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    float accuracy = (float)right_predictions_sum / params.test_amount;
    return accuracy;
}




int main(int argc, char** argv) {

    int rank, np;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    
    std::string dataset_name = "CIFAR10";
    std::string train_dataset_path = dataset_name + "/train.bin";
    std::string test_dataset_path = dataset_name + "/val.bin";
    DatasetParameters dataset_params;
    set_dataset_params(dataset_params, dataset_name);
    

    int iter_num = 6;
    float learning_rate = 0.0001;
    int minibatch_size = 100;

    int communication_freq = 1;
    int gossip_flag = 0;
    int sparsification_flag = 0;
    double sparsification_threshold = 0.0000001;
    int group_communication_flag = 0;

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
    if(argc > 8) {
        group_communication_flag = std::atof(argv[8]);
    }

    Network model;
    model.set_communication_options(communication_freq, gossip_flag,
                                     sparsification_flag, sparsification_threshold, group_communication_flag);


    model_from_file(model, "Models/test.txt");

    int epochs_done = 0;
    load_weights(model, epochs_done);

    //print_weights(model);

    double start_time, finish_time;
    double duration = 0;

    for(int i = 0; i < iter_num; i++) {
        start_time = MPI_Wtime();
        model.fit(train_dataset_path, dataset_params, minibatch_size, 1, learning_rate, epochs_done);
        
        double acc_check_start, acc_check_finish;
        acc_check_start = MPI_Wtime();
        float accuracy = test_model(model, test_dataset_path, dataset_params);
        acc_check_finish = MPI_Wtime();

        finish_time = MPI_Wtime();
        duration += finish_time - start_time;
        duration -= acc_check_finish - acc_check_start;

        if (rank == 0) {
            std::cout << "Iteration " << (i+epochs_done+1) <<" Accuracy: " << accuracy << std::endl;
        }
    }

    if(rank == 0) {
        save_weights(model, epochs_done + iter_num);
    }


    double local_synchronization_time = model.synchronization_time;
    double synchronization_time;
    MPI_Reduce(&local_synchronization_time, &synchronization_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_communication_time = model.communication_time;
    double communication_time;
    MPI_Reduce(&local_communication_time, &communication_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    double local_dataload_time = model.dataload_time;
    double dataload_time;
    MPI_Reduce(&local_dataload_time, &dataload_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0) {
        std::cout << "Training is over" << std::endl;
        std::cout << "Time: " << duration  << " s" << std::endl;
        std::cout << "Synchronization time: " << synchronization_time << std::endl;
        std::cout << "Communication time: " << communication_time << std::endl;
        std::cout << "Data load time: " << dataload_time << std::endl;
        //print_weights(model);
    }

    float accuracy = test_model(model, test_dataset_path, dataset_params);
    if(rank == 0) {
        std::cout << "Accuracy: " << accuracy << std::endl;
    }
    

    MPI_Finalize();

    return 0;
}




#pragma once

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>

#include <cstring>
#include <map>
#include <vector>
#include <unistd.h>
#include <cerrno>
#include <netdb.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "cblas.h"
#include <mpi.h>

struct DatasetParameters {
    int test_amount;
    int train_amount;
    int img_size;
    int channels_n;
    int classes_number;  
    float max_pixel_value = 255.0;
    float mean[3];
    float std[3];
};

class Network{
public:
    Network();


    class ConvParameters {
        public:
        ConvParameters() : img_size_x(0), img_size_y(0), maps_in(0), maps_out(0),
                             kernel_size(0), stride(0), padding(0), maxpool_flag(0),
                              locations_n_x(0), locations_n_y(0), locations_n(0) {};

        ConvParameters(int img_x, int img_y, int m_in, int m_out, int ker, int s, int p, int maxpool_f);
        void set(int img_x, int img_y, int m_in, int m_out, int ker, int s, int p, int maxpool_f);
        
        int img_size_x;
        int img_size_y;
        int maps_in;
        int maps_out;
        int kernel_size;
        int stride;
        int padding;
        int maxpool_flag;
        int locations_n_x;
        int locations_n_y;
        int locations_n;
    };


    class FcParameters {
        public:
        FcParameters() : n_in(0), n_out(0){};
        FcParameters(int n_input, int n_output) : n_in(n_input), n_out(n_output){};
        void set(int n_input, int n_output);

        int n_in;
        int n_out;
    };


    void add_conv_layer(int img_x, int img_y, int m_in, int m_out, int ker, int s, int p, int maxpool_f = 0);
    void add_fc_layer(int n_input, int n_output);

    void set_communication_options(int frq = 1, int gossip_f = 0, int spars_f = 0, double spars_thr = 0.1, int group_f = 0);

    void predict( float const* const* x, int img_size_x, int img_size_y, int channels_n, float* res);

    ~Network();

    void fit(std::string dataset_filename, DatasetParameters dataset_params,
             int minibatch_size = 0, int iter_n = 100, float learning_rate = 0.0001, int epochs_done = 0);



// private:
    const int max_conv_layers = 50;
    const int max_fc_layers = 50;
    int conv_layers_number;
    int fc_layers_number;
    float ***conv_weights;
    float **conv_bias;
    float ***fc_weights;
    float **fc_bias;
    float ***conv_weights_grad;
    float **conv_bias_grad;
    float ***fc_weights_grad;
    float **fc_bias_grad;
    ConvParameters *conv_params;
    FcParameters *fc_params;
    float ***conv_inputs;
    float **fc_inputs;
    int ***maxpool_indices;


    float l_rate;
    int number_of_channels;
    int class_n;
    int image_size_x;
    int image_size_y;
    int image_size;

    std::random_device rd{};
    std::mt19937 gen{100};

    double communication_time;
    double synchronization_time;
    double dataload_time;
    int communication_frequency;
    int gossip_flag;
    int sparsification_flag;
    double spars_threshold;
    int group_communication_flag;

    int gossip_step;

    std::string hostname;
    std::map<std::string, std::vector<int>> address_map;
    int leader;
    MPI_Comm group_communicator;
    int group_size;

    DatasetParameters dataset_params;



    void initial_weights_synch();

    void grad_synch(int global_iteration = 0);
    void grad_zero();


    void im2col(float **x, float **res, ConvParameters &params);
    void col2im(float **x, float **res, ConvParameters &params, int size);

    void matrix_mult(float const* const* a, float const* const* b, float** c,
                     int a_size1, int a_size2, int b_size1, int b_size2,
                      int a_trans_flag = 0, int b_trans_flag = 0);
    float vectors_mult(const float* a, const float* b, int size);
    void vector_matrix_mult( const float* a,  float const* const* b, float* c,
                             int a_size, int b_size1, int b_size2);
    void matrix_vector_mult(float const* const* a, const float* b, float* c,
                             int a_size1, int a_size2, int b_size);
    void matrix_trans(float const* const* m, float** res, int m_size1, int m_size2);

    void softmax(float* x, float* res, int x_size);

    void relu(const float* x, float* res, int x_size);
    void relu_backward(float* x, float* d_x, int n);

    void maxpool(float const* const* x, float** res, int** indices, int fmap_n, int img_size);
    void maxpool_backward(float **x, float **res, int **indices, ConvParameters &params, int size);

    void shuffle(int* y, int n);

    void fit_step(float*** X, int* y, int size);

    void sgd();

};



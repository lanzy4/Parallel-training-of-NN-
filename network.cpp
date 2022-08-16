#include "network.hpp"

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include "cblas.h"
#include <mpi.h>


Network::Network() {

    srand(100);

    communication_time = 0;
    synchronization_time = 0;
    communication_frequency = 1;
    gossip_flag = 0;
    sparsification_flag = 0;
    spars_threshold = 0.1;

    gossip_step = 1;

    conv_layers_number = 0;
    fc_layers_number = 0;

    conv_weights = new float**[max_conv_layers];
    conv_bias = new float*[max_conv_layers];
    fc_weights = new float**[max_fc_layers];
    fc_bias = new float*[max_fc_layers];
    conv_weights_grad = new float**[max_conv_layers];
    conv_bias_grad = new float*[max_conv_layers];
    fc_weights_grad = new float**[max_fc_layers];
    fc_bias_grad = new float*[max_fc_layers];
    conv_params = new ConvParameters[max_conv_layers];
    fc_params = new FcParameters[max_fc_layers];
    conv_inputs = new float**[max_conv_layers];
    fc_inputs = new float*[max_fc_layers];

    maxpool_indices = new int**[max_conv_layers];

}

Network::ConvParameters::ConvParameters(int img_x, int img_y, int m_in, int m_out,
                                         int ker, int s, int p, int maxpool_f) : img_size_x(img_x),
                                          img_size_y(img_y), maps_in(m_in), maps_out(m_out),
                                           kernel_size(ker), stride(s), padding(p), maxpool_flag(maxpool_f){
    locations_n_x = ((img_x + 2*p - ker)/s + 1);
    locations_n_y = ((img_y + 2*p - ker)/s + 1);
    locations_n = locations_n_x * locations_n_y;
};

void Network::ConvParameters::set(int img_x, int img_y, int m_in, int m_out,
                                     int ker, int s, int p, int maxpool_f) {
    img_size_x = img_x;
    img_size_y = img_y;
    maps_in = m_in;
    maps_out = m_out;
    kernel_size = ker;
    stride = s;
    padding = p;
    maxpool_flag = maxpool_f;
    locations_n_x = ((img_x + 2*p - ker)/s + 1);
    locations_n_y = ((img_y + 2*p - ker)/s + 1);
    locations_n = locations_n_x * locations_n_y;
}


void Network::FcParameters::set(int n_input, int n_output) {
    n_in = n_input;
    n_out = n_output;
}



void Network::add_conv_layer(int img_x, int img_y, int m_in, int m_out,
                             int ker, int s, int p, int maxpool_f) {


    conv_params[conv_layers_number].set(img_x, img_y, m_in, m_out, ker, s, p, maxpool_f);

    conv_weights[conv_layers_number] = new float*[m_out];
    conv_weights_grad[conv_layers_number] = new float*[m_out];
    for(int i = 0; i < m_out; i++) {
        conv_weights[conv_layers_number][i] = new float[ker*ker*m_in];
        conv_weights_grad[conv_layers_number][i] = new float[ker*ker*m_in];
        for(int j = 0; j < ker*ker*m_in; j++) {
            int cur_feature_n = ker*ker*m_in;
            float sigma = sqrt(2.0/cur_feature_n);
            std::normal_distribution<> d{0, sigma};
            conv_weights[conv_layers_number][i][j] = d(gen);
        }
    }

    conv_bias[conv_layers_number] = new float[m_out];
    conv_bias_grad[conv_layers_number] = new float[m_out];
    for(int i = 0; i < m_out; i++) {
        conv_bias[conv_layers_number][i] = 0;
    }

    int locations_n = ((img_x + 2*p - ker)/s + 1) * ((img_y + 2*p - ker)/s + 1);
    conv_inputs[conv_layers_number] = new float*[ker*ker*m_in];
    for(int i = 0; i < ker*ker*m_in; i++) {
        conv_inputs[conv_layers_number][i] = new float[locations_n];
    }

    if(maxpool_f) {
        int new_img_size = (img_x/2)*(img_y/2);
        maxpool_indices[conv_layers_number] = new int*[m_out];
        for(int i = 0; i < m_out; i++) {
            maxpool_indices[conv_layers_number][i] = new int[new_img_size];
            for(int j = 0; j < new_img_size; j++) {
                maxpool_indices[conv_layers_number][i][j] = 0;
            }
        }
    }

    conv_layers_number++;
}

void Network::add_fc_layer(int n_input, int n_output) {
    

    fc_params[fc_layers_number].set(n_input, n_output);

    fc_weights[fc_layers_number] = new float*[n_input];
    fc_weights_grad[fc_layers_number] = new float*[n_input];
    for(int i = 0; i < n_input; i++) {
        fc_weights[fc_layers_number][i] = new float[n_output];
        fc_weights_grad[fc_layers_number][i] = new float[n_output];
        for(int j = 0; j < n_output; j++) {
            int cur_feature_n = n_input;
            float sigma = sqrt(2.0/cur_feature_n);
            std::normal_distribution<> d{0, sigma};
            fc_weights[fc_layers_number][i][j] = d(gen);
        }
    }

    fc_bias[fc_layers_number] = new float[n_output];
    fc_bias_grad[fc_layers_number] = new float[n_output];
    for(int i = 0; i < n_output; i++) {
        fc_bias[fc_layers_number][i] = 0;
    }

    fc_inputs[fc_layers_number] = new float[n_input];

    class_n = n_output;

    fc_layers_number++;
}

void Network::set_communication_options(int frq, int gossip_f, int spars_f, double spars_thr) {
    communication_time = 0;
    synchronization_time = 0;
    communication_frequency = frq;
    gossip_flag = gossip_f;
    sparsification_flag = spars_f;
    spars_threshold = spars_thr;
}

void Network::predict( float const* const* x, int img_size_x, int img_size_y, int channels_n, float* res) {

    int input_image_dim1 = channels_n;
    int input_image_dim2 = img_size_x * img_size_y;
    float **input_image = new float*[input_image_dim1];
    for(int i = 0; i < input_image_dim1; i++) {
        input_image[i] = new float[input_image_dim2];
        for(int j = 0; j < input_image_dim2; j++) {
            input_image[i][j] = x[i][j];
        }
    }

    for(int i = 0; i < conv_layers_number; i++) {
        im2col(input_image, conv_inputs[i], conv_params[i]);

        for(int j = 0; j < input_image_dim1; j++) {
            delete[] input_image[j];
        }
        delete[] input_image;

        float **c_out = new float*[conv_params[i].maps_out];
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            c_out[j] = new float[conv_params[i].locations_n];
        }

        int dim_conv_inputs1 = conv_params[i].kernel_size * 
                                conv_params[i].kernel_size * conv_params[i].maps_in;
        matrix_mult(conv_weights[i], conv_inputs[i], c_out, conv_params[i].maps_out,
                     dim_conv_inputs1, dim_conv_inputs1, conv_params[i].locations_n);

        for(int j = 0; j < conv_params[i].maps_out; j++) {
            for(int k = 0; k < conv_params[i].locations_n; k++) {
                c_out[j][k] += conv_bias[i][j];
            }
        }

        for(int j = 0; j < conv_params[i].maps_out; j++) {
            relu(c_out[j], c_out[j], conv_params[i].locations_n);
        }

        if(conv_params[i].maxpool_flag) {
            input_image_dim1 = conv_params[i].maps_out;
            input_image_dim2 = (conv_params[i].locations_n_x/2)*(conv_params[i].locations_n_y/2);

            input_image = new float*[input_image_dim1];
            for(int j = 0; j < input_image_dim1; j++) {
                input_image[j] = new float[input_image_dim2];
                for(int k = 0; k < input_image_dim2; k++) {
                    input_image[j][k] = 0;
                }
            }
            maxpool(c_out, input_image, maxpool_indices[i], conv_params[i].maps_out,
                     conv_params[i].locations_n_x);
            
            for(int j = 0; j < conv_params[i].maps_out; j++) {
                delete[] c_out[j];
            }
            delete[] c_out;

        } else {
            input_image_dim1 = conv_params[i].maps_out;
            input_image_dim2 = conv_params[i].locations_n;
            input_image = c_out;
        }

    }


    for(int i = 0; i < input_image_dim1; i++) {
        for(int j = 0; j < input_image_dim2; j++) {
            fc_inputs[0][i*input_image_dim2 + j] = input_image[i][j];
        }   
    }
    for(int i = 0; i < input_image_dim1; i++) {
        delete[] input_image[i];
    }
    delete[] input_image;

    for(int i = 0; i < fc_layers_number; i++) {
        if(i != fc_layers_number - 1) {
            vector_matrix_mult(fc_inputs[i], fc_weights[i], fc_inputs[i + 1],
                                 fc_params[i].n_in, fc_params[i].n_in, fc_params[i].n_out);
            for(int j = 0; j < fc_params[i].n_out; j++) {
                fc_inputs[i + 1][j] += fc_bias[i][j];
            }
            relu(fc_inputs[i+1], fc_inputs[i+1], fc_params[i+1].n_in);
        } else {
            float *last_outputs = new float[fc_params[i].n_out];
            vector_matrix_mult(fc_inputs[i], fc_weights[i], last_outputs,
                                 fc_params[i].n_in, fc_params[i].n_in, fc_params[i].n_out);
            for(int j = 0; j < fc_params[i].n_out; j++) {
                last_outputs[j] += fc_bias[i][j];
            }
            softmax(last_outputs, res, fc_params[i].n_out);
            delete[] last_outputs;
        }
    }

}

Network::~Network() {
    for(int i = 0; i < conv_layers_number; i++) {
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            delete[] conv_weights[i][j];
            delete[] conv_weights_grad[i][j];
        }
        delete[] conv_weights[i];
        delete[] conv_weights_grad[i];

        delete[] conv_bias[i];
        delete[] conv_bias_grad[i];

        int l = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < l; j++) {
            delete[] conv_inputs[i][j];
        }
        delete[] conv_inputs[i];

        if(conv_params[i].maxpool_flag) {
            for(int j = 0; j < conv_params[i].maps_out; j++) {
                delete[] maxpool_indices[i][j];
            }
            delete[] maxpool_indices[i];
        }
    }
    delete[] conv_weights;
    delete[] conv_bias;
    delete[] conv_weights_grad;
    delete[] conv_bias_grad;
    delete[] conv_inputs;
    delete[] conv_params;
    delete[] maxpool_indices;

    for(int i = 0; i < fc_layers_number; i++) {
        for(int j = 0; j < fc_params[i].n_in; j++) {
            delete[] fc_weights[i][j];
            delete[] fc_weights_grad[i][j];
        }
        delete[] fc_weights[i];
        delete[] fc_bias[i];
        delete[] fc_weights_grad[i];
        delete[] fc_bias_grad[i];
        delete[] fc_inputs[i];
    }
    delete[] fc_weights;
    delete[] fc_bias;
    delete[] fc_weights_grad;
    delete[] fc_bias_grad;
    delete[] fc_inputs;
    delete[] fc_params;

}


void Network::fit(float ***X, int *y, int img_size_x, int img_size_y, int channels_n,
                     int sample_size, int minibatch_size, int iter_n, float learning_rate) {
    
    int rank;
    int np;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    l_rate = learning_rate;
    number_of_channels = channels_n;
    image_size_x = img_size_x;
    image_size_y = img_size_y;
    image_size = image_size_x * image_size_y;
    int img_size = image_size;

    initial_weights_synch();

    if (minibatch_size == 0) {
        minibatch_size = sample_size;
    }

    int minibatch_n  = sample_size / minibatch_size;
    int last_minibatch = sample_size % minibatch_size;
    if(last_minibatch != 0) {
        minibatch_n++;
    }
    for(int i = 0; i < iter_n; i++) {
        shuffle(X, y, sample_size);

        int n = minibatch_n / np + (rank < minibatch_n % np);
        int j_start = minibatch_n / np * rank + (rank < minibatch_n % np ? rank : minibatch_n % np);
        int j_finish = j_start + n;
        
        for(int j = j_start; j < j_finish; j++) {
            std::cout << "\r                                                          \r";
            std::cout << " iteration: " << i << ", minibatch: " << j << std::flush;
            int start, finish;
            start = j * minibatch_size;
            finish = (j + 1) * minibatch_size;
            if (finish > sample_size) {
                finish = sample_size;
            }
            int cur_minibatch_size = minibatch_size;
            if (j == minibatch_n - 1 && last_minibatch) {
                cur_minibatch_size = last_minibatch;
            }

            float ***X_mini = new float**[cur_minibatch_size];
            int *y_mini = new int[cur_minibatch_size];
            for (int k = 0; k < cur_minibatch_size; k++) {
                X_mini[k] = X[start + k];
                y_mini[k] = y[start + k];
            }


            fit_step(X_mini, y_mini, cur_minibatch_size);


            delete[] X_mini;
            delete[] y_mini;

            if(j != j_finish - 1 || rank >= minibatch_n % np) {
                if( (j - j_start) % communication_frequency == 0) {
                    double start_time, finish_time;
                    start_time = MPI_Wtime();
                    MPI_Barrier(MPI_COMM_WORLD);
                    finish_time = MPI_Wtime();
                    synchronization_time += finish_time - start_time;

                    start_time = MPI_Wtime();
                    grad_synch();
                    finish_time = MPI_Wtime();
                    communication_time += finish_time - start_time;
                    synchronization_time += finish_time - start_time;
                }
                sgd();
            }

        }
        
        if(rank >= minibatch_n % np) {
            grad_zero();
        }
        double start_time, finish_time;
        start_time = MPI_Wtime();
        MPI_Barrier(MPI_COMM_WORLD);
        finish_time = MPI_Wtime();
        synchronization_time += finish_time - start_time;

        start_time = MPI_Wtime();
        grad_synch();
        finish_time = MPI_Wtime();
        communication_time += finish_time - start_time;
        synchronization_time += finish_time - start_time;
        sgd();
        
    }

}



void Network::initial_weights_synch() {

    for(int i = 0; i < conv_layers_number; i++) {
        int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            MPI_Bcast(conv_weights[i][j], weights_number, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        int biases_number = conv_params[i].maps_out;
        MPI_Bcast(conv_bias[i], biases_number, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    for(int i = 0; i < fc_layers_number; i++) {
        int weights_number = fc_params[i].n_out;
        for(int j = 0; j < fc_params[i].n_in; j++) {
            MPI_Bcast(fc_weights[i][j], weights_number, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }

        int biases_number = fc_params[i].n_out;
        MPI_Bcast(fc_bias[i], biases_number, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

}


void Network::grad_synch() {

    int np;
    int rank;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if(gossip_flag == 0) {


        for(int i = 0; i < conv_layers_number; i++) {
            int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                    conv_params[i].maps_in;
            for(int j = 0; j < conv_params[i].maps_out; j++) {
                MPI_Allreduce(MPI_IN_PLACE, conv_weights_grad[i][j], weights_number, 
                                MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for(int k = 0; k < weights_number; k++) {
                    conv_weights_grad[i][j][k] /= np;
                }
            }

            int biases_number = conv_params[i].maps_out;
            MPI_Allreduce(MPI_IN_PLACE, conv_bias_grad[i], biases_number, 
                            MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for(int j = 0; j < biases_number; j++) {
                conv_bias_grad[i][j] /= np;
            }
        }

        for(int i = 0; i < fc_layers_number; i++) {
            int weights_number = fc_params[i].n_out;
            for(int j = 0; j < fc_params[i].n_in; j++) {
                MPI_Allreduce(MPI_IN_PLACE, fc_weights_grad[i][j], weights_number,
                                 MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
                for(int k = 0; k < weights_number; k++) {
                    fc_weights_grad[i][j][k] /= np;
                }
            }

            int biases_number = fc_params[i].n_out;
            MPI_Allreduce(MPI_IN_PLACE, fc_bias_grad[i], biases_number, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
            for(int j = 0; j < biases_number; j++) {
                fc_bias_grad[i][j] /= np;
            }
        }



    } else if (sparsification_flag == 0) {

        int neighbour_l, neighbour_r;
        neighbour_l = rank - gossip_step;
        if(neighbour_l < 0) {
            neighbour_l += np;
        }
        neighbour_r = rank + gossip_step;
        if(neighbour_r >= np) {
            neighbour_r -= np;
        }
        
        
        for(int i = 0; i < conv_layers_number; i++) {
            int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                    conv_params[i].maps_in;
            float *grad_left = new float[weights_number];
            float *grad_right = new float[weights_number];

            for(int j = 0; j < conv_params[i].maps_out; j++) {
                MPI_Sendrecv(conv_weights_grad[i][j], weights_number, MPI_FLOAT, neighbour_r, 0, 
                        grad_left, weights_number, MPI_FLOAT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
                MPI_Sendrecv(conv_weights_grad[i][j], weights_number, MPI_FLOAT, neighbour_l, 0, 
                        grad_right, weights_number, MPI_FLOAT, neighbour_r, 0, MPI_COMM_WORLD, NULL);
                for(int k = 0; k < weights_number; k++) {
                    conv_weights_grad[i][j][k] = grad_left[k]/3 + conv_weights_grad[i][j][k]/3 + 
                                                    grad_right[k]/3;
                }
            }
            delete[] grad_left;
            delete[] grad_right;

            int biases_number = conv_params[i].maps_out;
            float *grad_bias_left = new float[biases_number];
            float *grad_bias_right = new float[biases_number];
            MPI_Sendrecv(conv_bias_grad[i], biases_number, MPI_FLOAT, neighbour_r, 0, 
                        grad_bias_left, biases_number, MPI_FLOAT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(conv_bias_grad[i], biases_number, MPI_FLOAT, neighbour_l, 0, 
                        grad_bias_right, biases_number, MPI_FLOAT, neighbour_r, 0, MPI_COMM_WORLD, NULL);
            for(int k = 0; k < biases_number; k++) {
                conv_bias_grad[i][k] = grad_bias_left[k]/3 + conv_bias_grad[i][k]/3 + grad_bias_right[k]/3;
            }
            delete[] grad_bias_left;
            delete[] grad_bias_right;
        }


        for(int i = 0; i < fc_layers_number; i++) {
            int weights_number = fc_params[i].n_out;
            float *grad_left = new float[weights_number];
            float *grad_right = new float[weights_number];

            for(int j = 0; j < fc_params[i].n_in; j++) {
                MPI_Sendrecv(fc_weights_grad[i][j], weights_number, MPI_FLOAT, neighbour_r, 0, 
                        grad_left, weights_number, MPI_FLOAT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
                MPI_Sendrecv(fc_weights_grad[i][j], weights_number, MPI_FLOAT, neighbour_l, 0, 
                        grad_right, weights_number, MPI_FLOAT, neighbour_r, 0, MPI_COMM_WORLD, NULL);
                for(int k = 0; k < weights_number; k++) {
                    fc_weights_grad[i][j][k] = grad_left[k]/3 + fc_weights_grad[i][j][k]/3 + grad_right[k]/3;
                }
            }
            delete[] grad_left;
            delete[] grad_right;

            int biases_number = fc_params[i].n_out;
            float *grad_bias_left = new float[biases_number];
            float *grad_bias_right = new float[biases_number];
            MPI_Sendrecv(fc_bias_grad[i], biases_number, MPI_FLOAT, neighbour_r, 0, 
                        grad_bias_left, biases_number, MPI_FLOAT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(fc_bias_grad[i], biases_number, MPI_FLOAT, neighbour_l, 0, 
                        grad_bias_right, biases_number, MPI_FLOAT, neighbour_r, 0, MPI_COMM_WORLD, NULL);
            for(int k = 0; k < biases_number; k++) {
                fc_bias_grad[i][k] = grad_bias_left[k]/3 + fc_bias_grad[i][k]/3 + grad_bias_right[k]/3;
            }
            delete[] grad_bias_left;
            delete[] grad_bias_right;
        }

        gossip_step++;
        if(gossip_step > np/2) {
            gossip_step = 1;
        }
        
    } else {

        int neighbour_l, neighbour_r;
        neighbour_l = rank - gossip_step;
        if(neighbour_l < 0) {
            neighbour_l += np;
        }
        neighbour_r = rank + gossip_step;
        if(neighbour_r >= np) {
            neighbour_r -= np;
        }


        int total_n = 0;
        int kek_n = 0;
        for(int i = 0; i < conv_layers_number; i++) {
            int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                    conv_params[i].maps_in;
            for(int j = 0; j < conv_params[i].maps_out; j++) {
                for(int k = 0; k < weights_number; k++) {
                    if(conv_weights_grad[i][j][k] < spars_threshold && 
                            conv_weights_grad[i][j][k] > -spars_threshold) {

                        kek_n++;
                    }
                    total_n++;
                }
            }

            int biases_number = conv_params[i].maps_out;

            for(int k = 0; k < biases_number; k++) {
                if(conv_bias_grad[i][k] < spars_threshold && conv_bias_grad[i][k] > -spars_threshold) {
                    kek_n++;
                }
                total_n++;
            }
        }

        for(int i = 0; i < fc_layers_number; i++) {
            int weights_number = fc_params[i].n_out;
            for(int j = 0; j < fc_params[i].n_in; j++) {
                for(int k = 0; k < weights_number; k++) {
                    if(fc_weights_grad[i][j][k] < spars_threshold && 
                            fc_weights_grad[i][j][k] > -spars_threshold) {

                        kek_n++;
                    }
                    total_n++;
                }
            }

            int biases_number = fc_params[i].n_out;

            for(int k = 0; k < biases_number; k++) {
                if(fc_bias_grad[i][k] < spars_threshold && fc_bias_grad[i][k] > -spars_threshold) {
                    kek_n++;
                }
                total_n++;
            }
        }

        if(rank == 0) {
            std::cout << " " << kek_n << " / " << total_n << std::endl;
        }


        for(int i = 0; i < conv_layers_number; i++) {
            int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                    conv_params[i].maps_in;
            float *spars_grads = new float[weights_number * conv_params[i].maps_out];
            int *indices = new int[weights_number * conv_params[i].maps_out];

            int sparse_grads_number = 0;
            for(int j = 0; j < conv_params[i].maps_out; j++) {
                for(int k = 0; k < weights_number; k++) {
                    if(conv_weights_grad[i][j][k] > spars_threshold || 
                            conv_weights_grad[i][j][k] < -spars_threshold) {
                        spars_grads[sparse_grads_number] = conv_weights_grad[i][j][k];
                        indices[sparse_grads_number] = j * weights_number + k;
                        sparse_grads_number++;
                    }
                }
            }
            

            int sparse_grads_number_left, sparse_grads_number_right;
            MPI_Sendrecv(&sparse_grads_number, 1, MPI_INT, neighbour_r, 0, 
                        &sparse_grads_number_left, 1, MPI_INT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(&sparse_grads_number, 1, MPI_INT, neighbour_l, 0, 
                        &sparse_grads_number_right, 1, MPI_INT, neighbour_r, 0, MPI_COMM_WORLD, NULL);


            float *spars_grads_left = new float[sparse_grads_number_left];
            int *indices_left = new int[sparse_grads_number_left];
            float *spars_grads_right = new float[sparse_grads_number_right];
            int *indices_right = new int[sparse_grads_number_right];

            MPI_Sendrecv(spars_grads, sparse_grads_number, MPI_FLOAT, neighbour_r, 0, 
                         spars_grads_left, sparse_grads_number_left, MPI_FLOAT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(spars_grads, sparse_grads_number, MPI_FLOAT, neighbour_l, 0, 
                         spars_grads_right, sparse_grads_number_right, MPI_FLOAT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);

            MPI_Sendrecv(indices, sparse_grads_number, MPI_INT, neighbour_r, 0, 
                         indices_left, sparse_grads_number_left, MPI_INT, neighbour_l, 0, MPI_COMM_WORLD,
                         NULL);
            MPI_Sendrecv(indices, sparse_grads_number, MPI_INT, neighbour_l, 0, 
                         indices_right, sparse_grads_number_right, MPI_INT, neighbour_r, 0, MPI_COMM_WORLD,
                         NULL);


            for(int j = 0; j < conv_params[i].maps_out; j++) {
                for(int k = 0; k < weights_number; k++) {
                    conv_weights_grad[i][j][k] /= 3;
                }
            }

            for(int k = 0; k < sparse_grads_number_left; k++) {
                int index_j = indices_left[k] / weights_number;
                int index_k = indices_left[k] % weights_number;
                conv_weights_grad[i][index_j][index_k] += spars_grads_left[k]/3;
            }
            for(int k = 0; k < sparse_grads_number_right; k++) {
                int index_j = indices_right[k] / weights_number;
                int index_k = indices_right[k] % weights_number;
                conv_weights_grad[i][index_j][index_k] += spars_grads_right[k]/3;
            }

            delete[] spars_grads_left;
            delete[] indices_left;
            delete[] spars_grads_right;
            delete[] indices_right;


            delete[] spars_grads;
            delete[] indices;


            int biases_number = conv_params[i].maps_out;
            float *spars_bias_grads = new float[biases_number];
            int *indices_b = new int[biases_number];

            int sparse_bias_grads_number = 0;
            for(int k = 0; k < biases_number; k++) {
                if(conv_bias_grad[i][k] > spars_threshold || conv_bias_grad[i][k] < -spars_threshold) {
                    spars_bias_grads[sparse_bias_grads_number] = conv_bias_grad[i][k];
                    indices_b[sparse_bias_grads_number] = k;
                    sparse_bias_grads_number++;
                }
            }


            int sparse_bias_grads_number_left, sparse_bias_grads_number_right;
            MPI_Sendrecv(&sparse_bias_grads_number, 1, MPI_INT, neighbour_r, 0, 
                        &sparse_bias_grads_number_left, 1, MPI_INT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(&sparse_bias_grads_number, 1, MPI_INT, neighbour_l, 0, 
                        &sparse_bias_grads_number_right, 1, MPI_INT, neighbour_r, 0, MPI_COMM_WORLD, NULL);


            float *spars_bias_grads_left = new float[sparse_bias_grads_number_left];
            int *indices_b_left = new int[sparse_bias_grads_number_left];
            float *spars_bias_grads_right = new float[sparse_bias_grads_number_right];
            int *indices_b_right = new int[sparse_bias_grads_number_right];


            MPI_Sendrecv(spars_bias_grads, sparse_bias_grads_number, MPI_FLOAT, neighbour_r, 0, 
                         spars_bias_grads_left, sparse_bias_grads_number_left, MPI_FLOAT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(spars_bias_grads, sparse_bias_grads_number, MPI_FLOAT, neighbour_l, 0, 
                         spars_bias_grads_right, sparse_bias_grads_number_right, MPI_FLOAT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);

            MPI_Sendrecv(indices_b, sparse_bias_grads_number, MPI_INT, neighbour_r, 0, 
                         indices_b_left, sparse_bias_grads_number_left, MPI_INT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(indices_b, sparse_bias_grads_number, MPI_INT, neighbour_l, 0, 
                         indices_b_right, sparse_bias_grads_number_right, MPI_INT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);



            for(int k = 0; k < biases_number; k++) {
                conv_bias_grad[i][k] /= 3;
            }
            for(int k = 0; k < sparse_bias_grads_number_left; k++) {
                int index = indices_b_left[k];
                conv_bias_grad[i][index] += spars_bias_grads_left[k]/3;
            }
            for(int k = 0; k < sparse_bias_grads_number_right; k++) {
                int index = indices_b_right[k];
                conv_bias_grad[i][index] += spars_bias_grads_right[k]/3;
            }


            delete[] spars_bias_grads_left;
            delete[] indices_b_left;
            delete[] spars_bias_grads_right;
            delete[] indices_b_right;

            delete[] spars_bias_grads;
            delete[] indices_b;

        }


        for(int i = 0; i < fc_layers_number; i++) {
            int weights_number = fc_params[i].n_out;
            float *spars_grads = new float[weights_number * fc_params[i].n_in];
            int *indices = new int[weights_number * fc_params[i].n_in];

            int sparse_grads_number = 0;
            for(int j = 0; j < fc_params[i].n_in; j++) {
                for(int k = 0; k < weights_number; k++) {
                    if(fc_weights_grad[i][j][k] > spars_threshold ||
                             fc_weights_grad[i][j][k] < -spars_threshold) {
                        spars_grads[sparse_grads_number] = fc_weights_grad[i][j][k];
                        indices[sparse_grads_number] = j * weights_number + k;;
                        sparse_grads_number++;
                    }
                }
            }

                
            int sparse_grads_number_left, sparse_grads_number_right;
            MPI_Sendrecv(&sparse_grads_number, 1, MPI_INT, neighbour_r, 0, 
                        &sparse_grads_number_left, 1, MPI_INT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(&sparse_grads_number, 1, MPI_INT, neighbour_l, 0, 
                        &sparse_grads_number_right, 1, MPI_INT, neighbour_r, 0, MPI_COMM_WORLD, NULL);

            float *spars_grads_left = new float[sparse_grads_number_left];
            int *indices_left = new int[sparse_grads_number_left];
            float *spars_grads_right = new float[sparse_grads_number_right];
            int *indices_right = new int[sparse_grads_number_right];


            MPI_Sendrecv(spars_grads, sparse_grads_number, MPI_FLOAT, neighbour_r, 0, 
                         spars_grads_left, sparse_grads_number_left, MPI_FLOAT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(spars_grads, sparse_grads_number, MPI_FLOAT, neighbour_l, 0, 
                         spars_grads_right, sparse_grads_number_right, MPI_FLOAT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);

            MPI_Sendrecv(indices, sparse_grads_number, MPI_INT, neighbour_r, 0, 
                         indices_left, sparse_grads_number_left, MPI_INT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(indices, sparse_grads_number, MPI_INT, neighbour_l, 0, 
                         indices_right, sparse_grads_number_right, MPI_INT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);



            for(int j = 0; j < fc_params[i].n_in; j++) {
                for(int k = 0; k < weights_number; k++) {
                    fc_weights_grad[i][j][k] /= 3;
                }
            }
            for(int k = 0; k < sparse_grads_number_left; k++) {
                int index_j = indices_left[k] / weights_number;
                int index_k = indices_left[k] % weights_number;
                fc_weights_grad[i][index_j][index_k] += spars_grads_left[k]/3;
            }
            for(int k = 0; k < sparse_grads_number_right; k++) {
                int index_j = indices_right[k] / weights_number;
                int index_k = indices_right[k] % weights_number;
                fc_weights_grad[i][index_j][index_k] += spars_grads_right[k]/3;
            }

            delete[] spars_grads_left;
            delete[] indices_left;
            delete[] spars_grads_right;
            delete[] indices_right;


            delete[] spars_grads;
            delete[] indices;


            int biases_number = fc_params[i].n_out;
            float *spars_bias_grads = new float[biases_number];
            int *indices_b = new int[biases_number];

            int sparse_bias_grads_number = 0;
            for(int k = 0; k < biases_number; k++) {
                if(fc_bias_grad[i][k] > spars_threshold || fc_bias_grad[i][k] < -spars_threshold) {
                    spars_bias_grads[sparse_bias_grads_number] = fc_bias_grad[i][k];
                    indices_b[sparse_bias_grads_number] = k;
                    sparse_bias_grads_number++;
                }
            }


            int sparse_bias_grads_number_left, sparse_bias_grads_number_right;
            MPI_Sendrecv(&sparse_bias_grads_number, 1, MPI_INT, neighbour_r, 0, 
                        &sparse_bias_grads_number_left, 1, MPI_INT, neighbour_l, 0, MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(&sparse_bias_grads_number, 1, MPI_INT, neighbour_l, 0, 
                        &sparse_bias_grads_number_right, 1, MPI_INT, neighbour_r, 0, MPI_COMM_WORLD, NULL);


            float *spars_bias_grads_left = new float[sparse_bias_grads_number_left];
            int *indices_b_left = new int[sparse_bias_grads_number_left];
            float *spars_bias_grads_right = new float[sparse_bias_grads_number_right];
            int *indices_b_right = new int[sparse_bias_grads_number_right];


            MPI_Sendrecv(spars_bias_grads, sparse_bias_grads_number, MPI_FLOAT, neighbour_r, 0, 
                         spars_bias_grads_left, sparse_bias_grads_number_left, MPI_FLOAT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(spars_bias_grads, sparse_bias_grads_number, MPI_FLOAT, neighbour_l, 0, 
                         spars_bias_grads_right, sparse_bias_grads_number_right, MPI_FLOAT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);

            MPI_Sendrecv(indices_b, sparse_bias_grads_number, MPI_INT, neighbour_r, 0, 
                         indices_b_left, sparse_bias_grads_number_left, MPI_INT, neighbour_l, 0,
                         MPI_COMM_WORLD, NULL);
            MPI_Sendrecv(indices_b, sparse_bias_grads_number, MPI_INT, neighbour_l, 0, 
                         indices_b_right, sparse_bias_grads_number_right, MPI_INT, neighbour_r, 0,
                         MPI_COMM_WORLD, NULL);



            for(int k = 0; k < biases_number; k++) {
                fc_bias_grad[i][k] /= 3;
            }
            for(int k = 0; k < sparse_bias_grads_number_left; k++) {
                int index = indices_b_left[k];
                fc_bias_grad[i][index] += spars_bias_grads_left[k]/3;
            }
            for(int k = 0; k < sparse_bias_grads_number_right; k++) {
                int index = indices_b_right[k];
                fc_bias_grad[i][index] += spars_bias_grads_right[k]/3;
            }


            delete[] spars_bias_grads_left;
            delete[] indices_b_left;
            delete[] spars_bias_grads_right;
            delete[] indices_b_right;

            delete[] spars_bias_grads;
            delete[] indices_b;

        }

        gossip_step++;
        if(gossip_step > np/2) {
            gossip_step = 1;
        }

    }

}


void Network::grad_zero() {

    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    for(int i = 0; i < conv_layers_number; i++) {
        int weights_number = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            for(int k = 0; k < weights_number; k++) {
                conv_weights_grad[i][j][k] = 0;
            }
        }

        int biases_number = conv_params[i].maps_out;
        for(int j = 0; j < biases_number; j++) {
            conv_bias_grad[i][j] = 0;
        }
    }

    for(int i = 0; i < fc_layers_number; i++) {
        int weights_number = fc_params[i].n_out;
        for(int j = 0; j < fc_params[i].n_in; j++) {
            for(int k = 0; k < weights_number; k++) {
                fc_weights_grad[i][j][k] = 0;
            }
        }

        int biases_number = fc_params[i].n_out;
        for(int j = 0; j < biases_number; j++) {
            fc_bias_grad[i][j] = 0;
        }
    }

}


void Network::im2col(float **x, float **res, ConvParameters &params) {
    int res_dim1 = params.kernel_size * params.kernel_size * params.maps_in;
    int locations_n_x = (params.img_size_x + 2*params.padding - params.kernel_size)/params.stride + 1;
    int locations_n_y = (params.img_size_y + 2*params.padding - params.kernel_size)/params.stride + 1;
    int res_dim2 = locations_n_x * locations_n_y;

    int kernel_size = params.kernel_size;
    int zero_loc__center_i = kernel_size/2 - params.padding;
    int zero_loc__center_j = kernel_size/2 - params.padding;

    for(int i = 0; i < res_dim1; i++) {
        //filter number
        int filter_i = i / (kernel_size*kernel_size);
        //kernel pixel coordinates
        int kernel_i = (i % (kernel_size*kernel_size)) / kernel_size - kernel_size/2;
        int kernel_j = (i % (kernel_size*kernel_size)) % kernel_size - kernel_size/2;
        for(int j = 0; j < res_dim2; j++) {
            //location coordinates
            int loc_i = j / locations_n_y;
            int loc_j = j % locations_n_y;
            //location center coordinates (position on image)
            int loc_center_i = zero_loc__center_i + loc_i * params.stride;
            int loc_center_j = zero_loc__center_j + loc_j * params.stride;
            //pixel coordinates
            int pixel_i = loc_center_i + kernel_i;
            int pixel_j = loc_center_j + kernel_j;

            float temp = 0;
            if(pixel_i >= 0 && pixel_j >= 0 && pixel_i < params.img_size_x && pixel_j < params.img_size_y) {
                temp = x[filter_i][pixel_i * params.img_size_y + pixel_j];
            }
            res[i][j] = temp;
        }
    }
}


void Network::col2im(float **x, float **res, ConvParameters &params, int size) {
    int res_dim1 = params.kernel_size * params.kernel_size * params.maps_in;
    int locations_n_x = (params.img_size_x + 2*params.padding - params.kernel_size)/params.stride + 1;
    int locations_n_y = (params.img_size_y + 2*params.padding - params.kernel_size)/params.stride + 1;
    int res_dim2 = locations_n_x * locations_n_y;

    int kernel_size = params.kernel_size;
    int zero_loc__center_i = kernel_size/2 - params.padding;
    int zero_loc__center_j = kernel_size/2 - params.padding;

    for(int sample_number = 0; sample_number < size; sample_number++) {
        for(int i = 0; i < res_dim1; i++) {
            //filter number
            int filter_i = i / (kernel_size*kernel_size);
            //kernel pixel coordinates
            int kernel_i = (i % (kernel_size*kernel_size)) / kernel_size - kernel_size/2;
            int kernel_j = (i % (kernel_size*kernel_size)) % kernel_size - kernel_size/2;
            for(int j = 0; j < res_dim2; j++) {
                //location coordinates
                int loc_i = j / locations_n_y;
                int loc_j = j % locations_n_y;
                //location center coordinates (position on image)
                int loc_center_i = zero_loc__center_i + loc_i * params.stride;
                int loc_center_j = zero_loc__center_j + loc_j * params.stride;
                //pixel coordinates
                int pixel_i = loc_center_i + kernel_i;
                int pixel_j = loc_center_j + kernel_j;
                
                if(pixel_i >= 0 && pixel_j >= 0 && pixel_i < params.img_size_x && pixel_j < params.img_size_y) {
                    res[filter_i][sample_number*params.img_size_x*params.img_size_y + pixel_i *
                     params.img_size_y + pixel_j] = x[i][sample_number*params.img_size_x*params.img_size_y + j];
                }

            }
        }
    }

}

void Network::matrix_mult(float const* const* a, float const* const* b, float** c,
                            int a_size1, int a_size2, int b_size1, int b_size2, int a_trans_flag, int b_trans_flag) {
    
    int c_size1 = 0, c_size2 = 0;
    if(!a_trans_flag && !b_trans_flag) {
        c_size1 = a_size1;
        c_size2 = b_size2;
    } else if (a_trans_flag && !b_trans_flag) {
        c_size1 = a_size2;
        c_size2 = b_size2;
    } else if (!a_trans_flag && b_trans_flag) {
        c_size1 = a_size1;
        c_size2 = b_size1;
    } else {
        c_size1 = a_size2;
        c_size2 = b_size1;
    }

    float*  A = new float[a_size1*a_size2];
    float*  B = new float[b_size1*b_size2];
    float*  C = new float[c_size1*c_size2];

    for(int i = 0; i < a_size1; i++) {
        for(int j = 0; j < a_size2; j++) {
            A[i*a_size2 + j] = a[i][j];
        }
    }
    for(int i = 0; i < b_size1; i++) {
        for(int j = 0; j < b_size2; j++) {
            B[i*b_size2 + j] = b[i][j];
        }
    }

    if(!a_trans_flag && !b_trans_flag) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c_size1, c_size2, a_size2, 1.0,
                     A, a_size2, B, b_size2, 0.0, C, c_size2);
    } else if (a_trans_flag && !b_trans_flag) {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, c_size1, c_size2, a_size1, 1.0,
                     A, a_size2, B, b_size2, 0.0, C, c_size2);
    } else if (!a_trans_flag && b_trans_flag) {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, c_size1, c_size2, a_size2, 1.0, 
                     A, a_size2, B, b_size2, 0.0, C, c_size2);
    } else {
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasTrans, c_size1, c_size2, a_size1, 1.0, 
                     A, a_size2, B, b_size2, 0.0, C, c_size2);
    }

    for(int i = 0; i < c_size1; i++) {
        for(int j = 0; j < c_size2; j++) {
            c[i][j] = C[i*c_size2 + j];
        }
    }

    delete[] A;
    delete[] B;
    delete[] C;
}

float Network::vectors_mult(const float* a, const float* b, int size) {
    float res = 0;
    for (int i = 0; i < size; i++) {
        res += a[i] * b[i];
    }
    return res;
}

void Network::vector_matrix_mult( const float* a,  float const* const* b, float* c, 
                                    int a_size, int b_size1, int b_size2) {
    if(a_size != b_size1) {
        std::cout << "Can't multiply such vector and matrix" << std::endl;
        return;
    }
    for(int i = 0; i < b_size2; i++) {
        c[i] = 0;
    }
    for(int k = 0; k < b_size1; k++) {
        for (int i = 0; i < b_size2; i++) {
            c[i] += a[k] * b[k][i]; 
        }
    } 
}

void Network::matrix_vector_mult(float const* const* a, const float* b, float* c, int a_size1,
                                     int a_size2, int b_size) {
    if(a_size2 != b_size) {
        std::cout << "Can't multiply such matrix and vector" << std::endl;
        return;
    }
    for(int i = 0; i < a_size1; i++) {
        c[i] = 0;
    }
    for(int i = 0; i < a_size1; i++) {
        for(int j = 0; j < a_size2; j++) {
            c[i] += a[i][j] * b[j];
        }
    }
}

void Network::matrix_trans(float const* const* m, float** res, int m_size1, int m_size2) {
    for(int i = 0; i < m_size1; i++) {
        for(int j = 0; j < m_size2; j++) {
            res[j][i] = m[i][j];
        }
    }
}

void Network::softmax(float* x, float* res, int x_size) {
    float exp_sum = 0;

    for (int i = 0; i < x_size; i++) {
        exp_sum += exp(x[i]);
    }
    for (int i = 0; i < x_size; i++) {
        res[i] = exp(x[i]) / exp_sum;
    }
    
}

void Network::relu(const float* x, float* res, int x_size) {
    for (int i = 0; i < x_size; i++) {
        if (x[i] > 0) {
            res[i] = x[i];
        } else {
            res[i] = 0;
        }
    }
}

void Network::relu_backward(float* x, float* d_x, int n) {
    for(int i = 0; i < n; i++) {
        if(x[i] <= 0.0000001 && x[i] >= -0.0000001) {
            d_x[i] = 0;
        }
    }
}

void Network::maxpool(float const* const* x, float** res, int** indices, int fmap_n, int img_size) {
    for(int i = 0; i < fmap_n; i++) {
        for(int j = 0; j < img_size/2; j++) {
            for(int k = 0; k < img_size/2; k++) {
                int ind1 = (j*2) * img_size + k*2;
                int ind2 = (j*2) * img_size + k*2 + 1;
                int ind3 = (j*2 + 1) * img_size + k*2;
                int ind4 = (j*2 + 1) * img_size + k*2 + 1;
                int max_ind = ind1;
                float max = x[i][ind1];
                if(x[i][ind2] > max) {
                    max_ind = ind2;
                    max = x[i][ind2];
                }
                if(x[i][ind3] > max) {
                    max_ind = ind3;
                    max = x[i][ind3];
                }
                if(x[i][ind4] > max) {
                    max_ind = ind4;
                    max = x[i][ind4];
                }
                res[i][j * img_size/2 + k] = max;
                indices[i][j * img_size/2 + k] = max_ind;
            }
        }
    }
}

void Network::maxpool_backward(float **x, float **res, int **indices, ConvParameters &params, int size) {
    int res_dim1 = params.maps_out;
    int res_dim2 = size * params.img_size_x * params.img_size_y;

    for(int i = 0; i < res_dim1; i++) {
        for(int j = 0; j < res_dim2; j++) {
            res[i][j] = 0;
        }
    }

    for(int i = 0; i < size; i++) {
        for(int j = 0; j < params.maps_out; j++) {
            for(int k = 0; k < params.img_size_x; k++) {
                for(int l = 0; l < params.img_size_y; l++) {
                    int loc_k = k/2;
                    int loc_l = l/2;
                    if(indices[j][loc_k * (params.img_size_y/2) + loc_l] == k*params.img_size_y + l) {
                        res[j][i*params.img_size_x*params.img_size_y + k*params.img_size_y + l] =
                             x[j][i*(params.img_size_x/2)*(params.img_size_y/2) + 
                             loc_k*(params.img_size_y/2) + loc_l];
                    }
                }
            }
        }
    }

}

void Network::shuffle(float*** X, int* y, int n) {
    
    for(int i = 0; i < n; i++) {
        int switch_ind = rand() % n;
        float **t;
        t = X[i];
        X[i] = X[switch_ind];
        X[switch_ind] = t;

        int t2;
        t2 = y[i];
        y[i] = y[switch_ind];
        y[switch_ind] = t2;
    }
}

void Network::fit_step(float*** X, int* y, int size) {

    auto start_time = std::chrono::high_resolution_clock::now();

    float **err = new float*[size];
    for (int i = 0; i < size; i++) {
        err[i] = new float[class_n];
    }


    float ***all_fc_inputs = new float**[fc_layers_number];
    for(int i = 0; i < fc_layers_number; i++) {
        all_fc_inputs[i] = new float*[size];
        for(int j = 0; j < size; j++) {
            all_fc_inputs[i][j] = new float[fc_params[i].n_in];
        }
    }

    float ***all_conv_inputs = new float**[conv_layers_number];
    for(int i = 0; i < conv_layers_number; i++) {
        int dim1_size = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        int dim2_size = size * conv_params[i].locations_n;
        all_conv_inputs[i] = new float*[dim1_size];
        for(int j = 0; j < dim1_size; j++) {
            all_conv_inputs[i][j] = new float[dim2_size];
        }
    }


    for (int i = 0; i < size; i++) {

        float *y_pred = new float[class_n];
        float **X_sample = new float*[number_of_channels];
        for(int j = 0; j < number_of_channels; j++) {
            X_sample[j] = new float[image_size];
            for(int k = 0; k < image_size; k++) {
                X_sample[j][k] =  X[i][j][k];
            }
        }


        predict(X_sample, image_size_x, image_size_y, number_of_channels, y_pred);
        
        for(int j = 0; j < number_of_channels; j++) {
            delete[] X_sample[j];
        }
        delete[] X_sample;

        for (int j = 0; j < class_n; j++) {
            if (j == y[i]) {
                err[i][j] = 1.0;
            } else {
                err[i][j] = 0;
            }

            err[i][j] = y_pred[j] - err[i][j];

        }

        for(int j = 0; j < fc_layers_number; j++) {
            for(int k = 0; k < fc_params[j].n_in; k++) {
                all_fc_inputs[j][i][k] = fc_inputs[j][k];
            }
        }

        for(int j = 0; j < conv_layers_number; j++) {
            int dim1_size = conv_params[j].kernel_size * conv_params[j].kernel_size * conv_params[j].maps_in;
            int dim2_size_short = conv_params[j].locations_n;
            for(int k = 0; k < dim1_size; k++) {
                for(int l = 0; l < dim2_size_short; l++) {
                    all_conv_inputs[j][k][i * dim2_size_short + l] = conv_inputs[j][k][l];
                }
            }
        }
        

        delete[] y_pred;
    }


    float **d_inputs;
    int d_inputs_dim1 = size;
    int d_inputs_dim2 = 0;
    for(int i = fc_layers_number - 1; i >= 0; i--) {

        float **grad = new float*[fc_params[i].n_in];
        for(int j = 0; j < fc_params[i].n_in; j++) {
            grad[j] = new float[fc_params[i].n_out];
        }

        float *bias_grad = new float[fc_params[i].n_out];
        for(int j = 0; j < fc_params[i].n_out; j++) {
            bias_grad[j] = 0;
        }


        if(i == fc_layers_number -1) {

            d_inputs_dim2 = fc_params[i].n_in;
            d_inputs = new float*[d_inputs_dim1];
            for(int j = 0; j < d_inputs_dim1; j++) {
                d_inputs[j] = new float[d_inputs_dim2];
            }

            matrix_mult(all_fc_inputs[i], err, grad, size, fc_params[i].n_in, size, class_n, 1, 0);
            for(int j = 0; j < size; j++) {
                for(int k = 0; k < fc_params[i].n_out; k++) {
                    bias_grad[k] += err[j][k];
                }
            }
            matrix_mult(err, fc_weights[i], d_inputs, size, class_n, fc_params[i].n_in,
                         fc_params[i].n_out, 0, 1);

        } else {

            float **d_outputs = d_inputs;
            int d_outputs_dim1 = d_inputs_dim1;
            int d_outputs_dim2 = d_inputs_dim2;

            matrix_mult(all_fc_inputs[i], d_outputs, grad, size, fc_params[i].n_in, d_inputs_dim1,
                         d_inputs_dim2, 1, 0);

            for(int j = 0; j < size; j++) {
                for(int k = 0; k < fc_params[i].n_out; k++) {
                    bias_grad[k] += d_outputs[j][k];
                }
            }

            d_inputs_dim2 = fc_params[i].n_in;
            d_inputs = new float*[d_inputs_dim1];
            for(int j = 0; j < d_inputs_dim1; j++) {
                d_inputs[j] = new float[d_inputs_dim2];
            }

            matrix_mult(d_outputs, fc_weights[i], d_inputs, d_outputs_dim1, d_outputs_dim2,
                         fc_params[i].n_in, fc_params[i].n_out, 0, 1);
            for(int j = 0; j < d_outputs_dim1; j++) {
                delete[] d_outputs[j];
            }
            delete[] d_outputs;
        }

        for(int j = 0; j < size; j++) {
            relu_backward(all_fc_inputs[i][j], d_inputs[j], d_inputs_dim2);
        }

        for(int j = 0; j < fc_params[i].n_in; j++) {
            for(int k = 0; k < fc_params[i].n_out; k++) {
                fc_weights_grad[i][j][k] = grad[j][k];
            }
        }

        for(int j = 0; j < fc_params[i].n_out; j++) {
            fc_bias_grad[i][j] = bias_grad[j];
        }

        for(int j = 0; j < fc_params[i].n_in; j++) {
            delete[] grad[j];
        }
        delete[] grad;
        delete[] bias_grad;

    }


    for(int i = 0; i < size; i++) {
        delete[] err[i];
    }
    delete[] err;
    for(int i = 0; i < fc_layers_number; i++) {
        for(int j = 0; j < size; j++) {
            delete[] all_fc_inputs[i][j];
        }
        delete[] all_fc_inputs[i];
    }
    delete[] all_fc_inputs;


    int d_outputs_dim1 = conv_params[conv_layers_number - 1].maps_out;
    int d_outputs_dim2 = size * conv_params[conv_layers_number - 1].img_size_x * 
                            conv_params[conv_layers_number - 1].img_size_y;
    if(conv_params[conv_layers_number - 1].maxpool_flag) {
        d_outputs_dim2 = size * (conv_params[conv_layers_number - 1].img_size_x/2) * 
                            (conv_params[conv_layers_number - 1].img_size_y/2);
    }
    float **d_outputs = new float*[d_outputs_dim1];
    for(int i = 0; i < d_outputs_dim1; i++) {
        d_outputs[i] = new float[d_outputs_dim2];
        for (int j = 0; j < d_outputs_dim2; j++) {
            int sample_number = j / (d_outputs_dim2/size);
            int pixel_number = j % (d_outputs_dim2/size);
            d_outputs[i][j] = d_inputs[sample_number][i * (d_outputs_dim2/size) + pixel_number];
        }
    }
    for(int i = 0; i < d_inputs_dim1; i++) {
        delete[] d_inputs[i];
    }
    delete[] d_inputs;



    for(int i = conv_layers_number - 1; i >= 0; i--) {
        
        
        if(i != conv_layers_number - 1) {
            int d_outputs_in_wrong_form_dim1 = d_outputs_dim1;
            int d_outputs_in_wrong_form_dim2 = d_outputs_dim2;
            float **d_outputs_in_wrong_form = d_outputs;

            d_outputs_dim1 = conv_params[i].maps_out;
            d_outputs_dim2 = size * conv_params[i].img_size_x * conv_params[i].img_size_y;
            if(conv_params[i].maxpool_flag) {
                d_outputs_dim2 = size * (conv_params[i].img_size_x/2) * (conv_params[i].img_size_y/2);
            } 
            d_outputs = new float*[d_outputs_dim1];
            for(int j = 0; j < d_outputs_dim1; j++) {
                d_outputs[j] = new float[d_outputs_dim2];
            }

            col2im(d_outputs_in_wrong_form, d_outputs, conv_params[i + 1], size);


            for(int j = 0; j < d_outputs_in_wrong_form_dim1; j++) {
                delete[] d_outputs_in_wrong_form[j];
            }
            delete[] d_outputs_in_wrong_form;
        }


        if(conv_params[i].maxpool_flag) {
            int d_outputs_small_dim1 = d_outputs_dim1;
            int d_outputs_small_dim2 = d_outputs_dim2;
            float **d_outputs_small = d_outputs;

            d_outputs_dim1 = conv_params[i].maps_out;
            d_outputs_dim2 = size * conv_params[i].img_size_x * conv_params[i].img_size_y;
            d_outputs = new float*[d_outputs_dim1];
            for(int j = 0; j < d_outputs_dim1; j++) {
                d_outputs[j] = new float[d_outputs_dim2];
            }

            maxpool_backward(d_outputs_small, d_outputs, maxpool_indices[i], conv_params[i], size);


            for(int j = 0; j < d_outputs_small_dim1; j++) {
                delete[] d_outputs_small[j];
            }
            delete[] d_outputs_small;
        }


        float **grad = new float*[conv_params[i].maps_out];
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            grad[j] = new float[conv_params[i].kernel_size * conv_params[i].kernel_size * 
                        conv_params[i].maps_in];
        }
        
        float *bias_grad = new float[conv_params[i].maps_out];

        int all_conv_inputs_dim1 = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                    conv_params[i].maps_in;
        int all_conv_inputs_dim2 = size * conv_params[i].locations_n;

        matrix_mult(d_outputs, all_conv_inputs[i], grad, d_outputs_dim1, d_outputs_dim2, 
                        all_conv_inputs_dim1, all_conv_inputs_dim2, 0, 1);
        
        for(int j = 0; j < d_outputs_dim1; j++) {
            bias_grad[j] = 0;
            for(int k = 0; k < d_outputs_dim2; k++) {
                bias_grad[j] += d_outputs[j][k];
            }
        }

        if(i != 0) {
            d_inputs_dim1 = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
            d_inputs_dim2 = size * conv_params[i].locations_n;
            d_inputs = new float*[d_inputs_dim1];
            for(int j = 0; j < d_inputs_dim1; j++) {
                d_inputs[j] = new float[d_inputs_dim2];
            }
            
            int conv_weights_dim1 = conv_params[i].maps_out;
            int conv_weights_dim2 = conv_params[i].kernel_size * conv_params[i].kernel_size * 
                                        conv_params[i].maps_in;
            matrix_mult(conv_weights[i], d_outputs, d_inputs, conv_weights_dim1, conv_weights_dim2, 
                            d_outputs_dim1, d_outputs_dim2, 1, 0);

            for(int j = 0; j < d_inputs_dim1; j++) {
                relu_backward(all_conv_inputs[i][j], d_inputs[j], d_inputs_dim2);
            }


            for(int j = 0; j < d_outputs_dim1; j++) {
                delete[] d_outputs[j];
            }
            delete[] d_outputs;

            d_outputs = d_inputs;
            d_outputs_dim1 = d_inputs_dim1;
            d_outputs_dim2 = d_inputs_dim2;

        }


        int weight_dim2 = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            for(int k = 0; k < weight_dim2; k++) {
                conv_weights_grad[i][j][k] = grad[j][k];
            }
        }

        for(int j = 0; j < conv_params[i].maps_out; j++) {
            conv_bias_grad[i][j] = bias_grad[j];
        }


        for(int j = 0; j < conv_params[i].maps_out; j++) {
            delete[] grad[j];
        }
        delete[] grad;
        delete[] bias_grad;
    }


    for(int i = 0; i < d_outputs_dim1; i++) {
        delete[] d_outputs[i];
    }
    delete[] d_outputs;

    for(int i = 0; i < conv_layers_number; i++) {
        int dim1_size = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < dim1_size; j++) {
            delete[] all_conv_inputs[i][j];
        }
        delete[] all_conv_inputs[i];
    }
    delete[] all_conv_inputs;



    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time-start_time).count();

    //std::cout << "time: " <<duration / 1000000.0 << " s" << std::endl;

}

void Network::sgd() {

    for(int i = 0; i < conv_layers_number; i++) {
        int weight_dim2 = conv_params[i].kernel_size * conv_params[i].kernel_size * conv_params[i].maps_in;
        for(int j = 0; j < conv_params[i].maps_out; j++) {
            for(int k = 0; k < weight_dim2; k++) {
                conv_weights[i][j][k] -= l_rate * conv_weights_grad[i][j][k];
            }
        }

        for(int j = 0; j < conv_params[i].maps_out; j++) {
            conv_bias[i][j] -= l_rate * conv_bias_grad[i][j];
        }
    }


    for(int i = 0; i < fc_layers_number; i++) {
        for(int j = 0; j < fc_params[i].n_in; j++) {
            for(int k = 0; k < fc_params[i].n_out; k++) {
                fc_weights[i][j][k] -= l_rate * fc_weights_grad[i][j][k];
            }
        }

        for(int j = 0; j < fc_params[i].n_out; j++) {
            fc_bias[i][j] -= l_rate * fc_bias_grad[i][j];
        }
    }

}




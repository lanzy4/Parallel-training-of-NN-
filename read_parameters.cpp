#include "read_parameters.hpp"

#include <iostream>
#include <cmath>
#include <chrono>
#include <fstream>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include "cblas.h"
#include <mpi.h>



int reverseInt (int i) {

    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}



void read_mnist_data(float*** &result, std::string filename)  {  
     
    const float MNIST_MAX_PIXEL_VALUE = 255.0;
    const float MNIST_MEAN = 0.1307;
    const float MNIST_STD_DEVIATION = 0.3081;
    std::ifstream file (filename);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
        result = new float**[number_of_images];
        for(int i = 0; i < number_of_images; i++) {
            result[i] = new float*[1];
            for(int r = 0; r < 1; r++) {
                result[i][r] = new float[n_rows * n_cols];
            }
        }
        for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows*n_cols;++r)
            {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                float value = (float)temp / MNIST_MAX_PIXEL_VALUE;
                value -= MNIST_MEAN;
                value /= MNIST_STD_DEVIATION;
                result[i][0][r] = value;
            }
        }

        file.close();
        
    }
    
}

void read_mnist_labels(int* &result, std::string filename)  {

    std::ifstream file (filename);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        result = new int[number_of_images];
        for(int i=0;i<number_of_images;++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            result[i] = temp;
        }

        file.close();
        
    }
    
}


void read_cifar10(float*** &x_train, int* &y_train, float*** &x_test, int* &y_test) {

    int train_images_number = 50000;
    int test_images_number = 10000;
    int channels_n = 3;
    int img_size = 32;

    float CIFAR10__MAX_PIXEL_VALUE = 255.0;
    float CIFAR10_MEAN[3]{0.49139968, 0.48215827 ,0.44653124};
    float CIFAR10_STD_DEVIATION[3]{0.24703233, 0.24348505, 0.26158768};

    x_train = new float**[train_images_number];
    for(int i = 0; i < train_images_number; i++) {
        x_train[i] = new float*[channels_n];
        for(int j = 0; j < channels_n; j++) {
            x_train[i][j] = new float[img_size*img_size];
        }
    }
    x_test = new float**[test_images_number];
    for(int i = 0; i < test_images_number; i++) {
        x_test[i] = new float*[channels_n];
        for(int j = 0; j < channels_n; j++) {
            x_test[i][j] = new float[img_size*img_size];
        }
    }
    y_train = new int[train_images_number];
    y_test = new int[test_images_number];

    int train_files_number = 5;
    int images_per_file = 10000;

    for(int i = 0; i < train_files_number; i++) {

        std::string filename = "CIFAR10/data_batch_";
        filename += std::to_string(i+1);
        filename += ".bin";

        std::ifstream file(filename);

        for(int j = 0; j < images_per_file; j++) {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            y_train[i*images_per_file + j] = temp;

            for(int k = 0; k < channels_n; k++) {
                for(int l = 0; l < img_size*img_size; l++) {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
                    float value = (float)temp / CIFAR10__MAX_PIXEL_VALUE;
                    value -= CIFAR10_MEAN[k];
                    value /= CIFAR10_STD_DEVIATION[k];
                    x_train[i*images_per_file + j][k][l] = value;
                }
            }
        }

        file.close();
    }



    std::string filename = "CIFAR10/test_batch.bin";
    std::ifstream file(filename);

    for(int j = 0; j < images_per_file; j++) {
        unsigned char temp=0;
        file.read((char*)&temp,sizeof(temp));
        y_test[j] = temp;

        for(int k = 0; k < channels_n; k++) {
            for(int l = 0; l < img_size*img_size; l++) {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                float value = (float)temp / CIFAR10__MAX_PIXEL_VALUE;
                value -= CIFAR10_MEAN[k];
                value /= CIFAR10_STD_DEVIATION[k];
                x_test[j][k][l] = value;
            }
        }
    }

    file.close();

}


void read_cifar100(float*** &x_train, int* &y_train, float*** &x_test, int* &y_test) {
    
    int train_images_number = 50000;
    int test_images_number = 10000;
    int channels_n = 3;
    int img_size = 32;

    float CIFAR100__MAX_PIXEL_VALUE = 255.0;
    float CIFAR100_MEAN[3]{0.5071, 0.4867, 0.4408};
    float CIFAR100_STD_DEVIATION[3]{0.2675, 0.2565, 0.2761};

    x_train = new float**[train_images_number];
    for(int i = 0; i < train_images_number; i++) {
        x_train[i] = new float*[channels_n];
        for(int j = 0; j < channels_n; j++) {
            x_train[i][j] = new float[img_size*img_size];
        }
    }
    x_test = new float**[test_images_number];
    for(int i = 0; i < test_images_number; i++) {
        x_test[i] = new float*[channels_n];
        for(int j = 0; j < channels_n; j++) {
            x_test[i][j] = new float[img_size*img_size];
        }
    }
    y_train = new int[train_images_number];
    y_test = new int[test_images_number];

    int images_per_file = 50000;
    std::string filename = "CIFAR100/train.bin";

    std::ifstream file(filename);

    for(int j = 0; j < images_per_file; j++) {
        unsigned char temp=0;
        file.read((char*)&temp,sizeof(temp));
        file.read((char*)&temp,sizeof(temp));
        y_train[j] = temp;

        for(int k = 0; k < channels_n; k++) {
            for(int l = 0; l < img_size*img_size; l++) {
                unsigned char temp=0;
                file.read((char*)&temp,sizeof(temp));
                float value = (float)temp / CIFAR100__MAX_PIXEL_VALUE;
                value -= CIFAR100_MEAN[k];
                value /= CIFAR100_STD_DEVIATION[k];
                x_train[j][k][l] = value;
            }
        }
    }

    file.close();


    filename = "CIFAR100/test.bin";
    std::ifstream file_test(filename);

    images_per_file = 10000;

    for(int j = 0; j < images_per_file; j++) {
        unsigned char temp=0;
        file_test.read((char*)&temp,sizeof(temp));
        file_test.read((char*)&temp,sizeof(temp));
        y_test[j] = temp;

        for(int k = 0; k < channels_n; k++) {
            for(int l = 0; l < img_size*img_size; l++) {
                unsigned char temp=0;
                file_test.read((char*)&temp,sizeof(temp));
                float value = (float)temp / CIFAR100__MAX_PIXEL_VALUE;
                value -= CIFAR100_MEAN[k];
                value /= CIFAR100_STD_DEVIATION[k];
                x_test[j][k][l] = value;
            }
        }
    }

    file_test.close();

}


void read_dataset(float*** &x_train, int* &y_train, float*** &x_test,
                     int* &y_test, DatasetParameters &params, std::string dataset_name) {
    if(dataset_name == "MNIST") {
        params.train_amount = 60000;
        params.test_amount = 10000;
        params.img_size = 28;
        params.channels_n = 1;
        params.classes_number = 10;
        read_mnist_data(x_train, "MNIST/train_images");
        read_mnist_labels(y_train, "MNIST/train_labels");
        read_mnist_data(x_test, "MNIST/test_images");
        read_mnist_labels(y_test, "MNIST/test_labels");
    } else if(dataset_name == "CIFAR10") {
        params.train_amount = 50000;
        params.test_amount = 10000;
        params.img_size = 32;
        params.channels_n = 3;
        params.classes_number = 10;
        read_cifar10(x_train, y_train, x_test, y_test);
    } else if(dataset_name == "CIFAR100") {
        params.train_amount = 50000;
        params.test_amount = 10000;
        params.img_size = 32;
        params.channels_n = 3;
        params.classes_number = 100;
        read_cifar100(x_train, y_train, x_test, y_test);
    }
}



void free_dataset(float*** &x_train, int* &y_train, float*** &x_test,
                     int* &y_test, DatasetParameters &params) {
    int train_dim1 = params.train_amount;
    int test_dim1 = params.test_amount;
    int dim2 = params.channels_n;

    for(int i = 0; i < train_dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            delete[] x_train[i][j];
        }
        delete[] x_train[i];
    }
    delete[] x_train;
    delete[] y_train;

    for(int i = 0; i < test_dim1; i++) {
        for(int j = 0; j < dim2; j++) {
            delete[] x_test[i][j];
        }
        delete[] x_test[i];
    }
    delete[] x_test;
    delete[] y_test;
}




void model_from_file(Network &model, std::string filename) {

    int rank, np; 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &np);

    std::ifstream file(filename, std::ios::in);

    std::string layer_type;
    std::string next_layer_type;
    std::string s;

    int stop_flag = 0;

    file >> layer_type;
    if(layer_type == "END") {
        stop_flag = 1;
    }


    while(!stop_flag) {

        if(layer_type == "CONV") {
            int img_size_x, img_size_y, in_channels, out_channels, kernel_size, stride, padding, maxpool_flag;
            file >> s;
            file >> img_size_x;
            file >> s;
            file >> img_size_y;
            file >> s;
            file >> in_channels;
            file >> s;
            file >> out_channels;
            file >> s;
            file >> kernel_size;
            file >> s;
            file >> stride;
            file >> s;
            file >> padding;

            file >> next_layer_type;
            maxpool_flag = 0;
            if(next_layer_type != "END") {
                if(next_layer_type == "MAXPOOL") {
                    maxpool_flag = 1;
                    file >> layer_type;
                } else {
                    layer_type = next_layer_type;
                }
            } else {
                stop_flag = 1;
            }

            model.add_conv_layer(img_size_x, img_size_y, in_channels, out_channels,
                                 kernel_size, stride, padding, maxpool_flag);

            if(rank == 0) {
                std::cout << "CONV LAYER " << in_channels << " " << out_channels << " " <<
                    kernel_size << " " << stride << " " << padding << " " << maxpool_flag << std::endl;
            }

        } else if(layer_type == "FC") {
            int in_features, out_features;
            file >> s;
            file >> in_features;
            file >> s;
            file >> out_features;

            file >> layer_type;
            if(layer_type == "END") {
                stop_flag = 1;
            }

            model.add_fc_layer(in_features, out_features);

            if(rank == 0) {
                std::cout << "FC LAYER " << in_features << " " << out_features << std::endl;
            }
        }

    }

    file.close();
}


void set_dataset_params(DatasetParameters &params, std::string dataset_name) {
    if(dataset_name == "IMAGENETTE") {
        params.train_amount = 9469;
        params.test_amount = 3925;
        params.img_size = 224;
        params.channels_n = 3;
        params.classes_number = 10;
        params.max_pixel_value = 255.0;

        params.mean[0] = 0.4625;
        params.mean[1] = 0.4580;
        params.mean[2] = 0.4298;

        params.std[0] = 0.2827;
        params.std[1] = 0.2795;
        params.std[2] = 0.3019;
    } else if(dataset_name == "IMAGEWOOF") {
        params.train_amount = 9025;
        params.test_amount = 3929;
        params.img_size = 224;
        params.channels_n = 3;
        params.classes_number = 10;
        params.max_pixel_value = 255.0;

        params.mean[0] = 0.4860;
        params.mean[1] = 0.4559;
        params.mean[2] = 0.3940;

        params.std[0] = 0.2575;
        params.std[1] = 0.2505;
        params.std[2] = 0.2591;
    } else if(dataset_name == "IMAGENETTE227") {
        params.train_amount = 9469;
        params.test_amount = 3925;
        params.img_size = 227;
        params.channels_n = 3;
        params.classes_number = 10;
        params.max_pixel_value = 255.0;

        params.mean[0] = 0.4625;
        params.mean[1] = 0.4580;
        params.mean[2] = 0.4298;

        params.std[0] = 0.2827;
        params.std[1] = 0.2795;
        params.std[2] = 0.3019;
    } else if(dataset_name == "IMAGEWOOF227") {
        params.train_amount = 9025;
        params.test_amount = 3929;
        params.img_size = 227;
        params.channels_n = 3;
        params.classes_number = 10;
        params.max_pixel_value = 255.0;

        params.mean[0] = 0.4860;
        params.mean[1] = 0.4559;
        params.mean[2] = 0.3940;

        params.std[0] = 0.2575;
        params.std[1] = 0.2505;
        params.std[2] = 0.2591;
    } else if(dataset_name == "CIFAR10") {
        params.train_amount = 50000;
        params.test_amount = 10000;
        params.img_size = 32;
        params.channels_n = 3;
        params.classes_number = 10;
        params.max_pixel_value = 255.0;

        params.mean[0] = 0.49139968;
        params.mean[1] = 0.48215827;
        params.mean[2] = 0.44653124;

        params.std[0] = 0.24703233;
        params.std[1] = 0.24348505;
        params.std[2] = 0.26158768;
    }
}


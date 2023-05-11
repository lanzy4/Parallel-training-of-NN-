#include <iostream>
#include <cstdlib>
#include <fstream>


int main(int argc, char **argv) {

    std::ofstream file("weights.bin", std::ios::out);
    int epoch = 0;
    file.write((char*)&epoch, sizeof(int));
    file.close();

    return 0;
}
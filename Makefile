all:
	mpiCC network.cpp read_parameters.cpp main.cpp -o prog -lopenblas -O4

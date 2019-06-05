CXX = g++
NVCC = nvcc
FLAG = -std=c++11 -O3
LIB = -lm
LIBCU = -lcublas -lcusparse
SRC = MLP.cpp
SRCCU = MLP_cuda.cu
TARGET = mlp mlp_cuda

all::$(TARGET)

mlp: $(SRC)
	$(CXX) $(FLAG) $(LIB) -o $@ $^

mlp_cuda: $(SRCCU)
	$(NVCC) $(FLAG) $(LIB) $(LIBCU) -o $@ $^

clean:
	rm -rf $(TARGET)

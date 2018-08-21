LINKFLAGS=-Ithird_party/xtensor/include -Ithird_party/xtensor-blas/include -Ithird_party/xsimd/include -Ithird_party/xtl/include
CC_FLAGS=-march=native -Ofast -lcblas 

all:
	g++-8 main.cpp $(LINKFLAGS) $(CC_FLAGS) -o main

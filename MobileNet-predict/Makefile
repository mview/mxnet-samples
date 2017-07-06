# Special thanks to https://github.com/pertusa for the Makefile
CFLAGS=-std=c++11 -Wno-unknown-pragmas -Wall

# Added for openblas
# export OPENBLAS_ROOT=/usr/local/opt/openblas

# CFLAGS+= -I${OPENBLAS_ROOT}/include
# LDFLAGS=-L${OPENBLAS_ROOT}/lib -lopenblas

# Added for opencv
CFLAGS+= `pkg-config --cflags opencv`
LDFLAGS+=`pkg-config --libs opencv`

# Added for mxnet
CFLAGS+=-Wall -I$(DIR_MXNET)/include -g
LDFLAGS+=-L$(DIR_MXNET)/lib -L/usr/local/cuda-8.0/targets/x86_64-linux/lib/ -lcudnn -lmxnet -lopenblas 
LDFLAGS += -Wl,-rpath-link /usr/local/cuda-8.0/targets/x86_64-linux/lib/

image-classification-predict: image-classification-predict.o
	$(CXX) -O3 -o image-classification-predict image-classification-predict.o $(LDFLAGS)

image-classification-predict.o: image-classification-predict.cc
	$(CXX) -O3 -c image-classification-predict.cc ${CFLAGS}

clean: 
	rm -f *.d *.o
	rm -f image-classification-predict 

lint:
	python ../../../dmlc-core/scripts/lint.py mxnet "cpp" ./

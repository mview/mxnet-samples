#############################################################
INCS += -I$(DIR_MXNET)/include
INCS += -I$(DIR_MXNET)/cpp-package/include
INCS += -I$(DIR_MXNET)/include/mxnet
INCS += -I$(DIR_MXNET)/mshadow
INCS += -I$(DIR_MXNET)/nnvm/include
INCS += -I$(DIR_MXNET)/dmlc-core/include
INCS += -I$(DIR_MXNET)/dlpack/include

#############################################################
CXX := g++
CXXFLAGS := -O0 -Wall -fPIC -std=c++11 -g
LOCAL_MODULE=mtcnn

ifeq (1,$(CPU_ONLY))
CXXFLAGS += -DCPU_ONLY
endif

ifeq (1,$(MSHADOW_USE_MKL))
   CXXFLAGS+=-DMSHADOW_USE_MKL=1
else
   CXXFLAGS+=-DMSHADOW_USE_MKL=0
endif

LIBS += -L$(DIR_MXNET)/lib -L/usr/local/cuda-8.0/targets/x86_64-linux/lib/ -lopencv_highgui -lopencv_core -lopencv_imgproc -lmxnet -ldl -lcudnn

CXXFLAGS += $(INCS) -g
LDFLAGS += -Wl,-rpath-link /usr/local/cuda-8.0/targets/x86_64-linux/lib/

#############################################################
all : $(LOCAL_MODULE) 

$(LOCAL_MODULE) : MTMain.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(LIBS)

MTMain.o: MTMain.cpp
	$(CXX) -O3 ${CXXFLAGS} -c $^ 

clean:
	rm -f $(LOCAL_MODULE) *.o 

.PHONY : all clean


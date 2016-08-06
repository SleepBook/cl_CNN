CFLAGS = -Iinclude/
OBJS = clcnn.o test.o
CL_DEPS = -L/usr/local/cuda/lib64 -lOpenCL
CL_HEAD = -I/usr/local/cuda/include/
TARGET = test
OBJDIR = bin

VPATH = src:include:data:.:bin

test:$(OBJS) $(OBJDIR)
	cd $(OBJDIR) && $(CXX) $(OBJS) $(CL_DEPS) -o $@

clcnn.o:clcnn.cpp clcnn.h $(OBJDIR)
	$(CXX) -c $< $(CL_HEAD) $(CFLAGS) -o $@ && mv $@ ./bin

test.o:test.cpp clcnn.h $(OBJDIR)
	$(CXX) -c $< $(CL_HEAD) $(CFLAGS) -o $@ && mv $@ ./bin

$(OBJDIR):
	mkdir -p ./bin
clean:
	cd ./bin && rm *

NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include -arch sm_20
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = four1
OBJ	        = main.o 

default: $(EXE)

test-mode: NVCC_FLAGS += -DTEST_MODE
test-mode: default

main.o: main.cu kernel.cu
	$(NVCC) -c -o $@ main.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

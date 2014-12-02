
all: pazam_cpu

pazam_cpu: pazam_cpu.c fft64.c 
	cc -lm -o pazam_cpu pazam_cpu.c fft64.c -I./

clean: 
	rm pazam_cpu

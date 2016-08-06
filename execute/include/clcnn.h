#ifndef CLCNN_H
#define CLCNN_H

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <ctime>
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

//#define WIN
#define UNIX
#ifdef WIN
#include<Windows.h>
#endif
#ifdef UNIX
#include <sys/time.h>
#endif

#define DEBUG_MODE 0 //will print extra information

#define TEST_PLAT 0
#define TEST_DEV 0

#define INPUT_WIDTH 28
#define INPUT_HEIGHT 28
#define INPUT_CHANNEL 1

//some knots
#define VISUAL
//#define DEV_DEBUG
#define FLEX_THREAD_NUM
//choose the OpenCl platform and device interactively
//#def INTERACTIVE

class clcnn
{
public:
	int getInput();//port to external
	int execute_device();
	int execute_cpu();
	int retrieve_result(float* container, int lang);
	int predict();
	float getKernelTime();
	float getCPUTime();
	float getLoadTime();

	clcnn(const char* net_file);
	~clcnn();
	cl_kernel convp_sig_avg_full, convp_sig_avg_mc, convp_sig_avg_pc;
	cl_kernel fc_sig, fcp_sig;
	float* layer_time;

private:
	int load_num;
	int cpu_load_num;
	int device_load_num;

	int load(const char* net_file);
	void load_cpu();
	void load_device();
	cl_program load_program(cl_context context, cl_device_id device, const char* filename);
	
	int exe_conv_device(int* layer_pos, int flag);
	int exe_pool_device(int* layer_pos, int flag);
	int exe_fc_device(int* layer_pos, int flag);
	/*
	int exe_conv_cpu(int mode, int non_linear);
	int exe_pool_cpu(int mode);
	int exe_fc_cpu(int mode, int non_linear);*/
	float non_linear(int mode, float input);
	

	/////////////////////////////////////////
	//varibles for CL environment
	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_uint platnum, devnum;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_event timer;
	cl_ulong start, end, elipse;

	////////////////////////////////////////////
	//varibles for the data structure on host
	int layer_num;
	FILE *net_config;
	FILE *input_data;

	int *layer_type;//internal code for layers: CONV 0 FULL 1 POOL 2
	int *feature_num;//include the input data, n+1

	float *****filters;
	int **kernel_dim;
	float **bias;
	int *stride;
	int *method;//record the pooling method only for pool layer

	//some extra var to record the link info and address filters
	//because CL only allow 1-d buffer
	int **link_info;
		//the 2D array flatted into 1D, the first link record how many kernel in each kernel group
		//the following lines record respective kernel's input data(last layer)'s offset in the 1D
		//array which stores the interanl result. 
	int **filter_offset;
		//the question is different kernel group may have different number of kernels, so this mark
		//the boundary among differnet kernel groups
	int *total_kernel;

	int **inter_res_dim;//include the input data, n+1
	float ****inter_res;//same, n+1
	int MAX_INTER_RES;

	int **launch_info;
	int *max_group_size;

	/////////////////////////////////////////////
	//varible for DS on device
	cl_mem *d_net;
	cl_mem *d_bias;
	cl_mem *d_flt_offset;
	cl_mem *d_flt_sz_lk;//stands for size_link
	cl_mem d_inter_res_a;
	cl_mem d_inter_res_b;
	
	float kernel_time, load_time, CPUtime;
};

#endif

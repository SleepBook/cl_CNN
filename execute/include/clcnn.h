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
//#define VERBOSE
#define VISUAL
//#define DEV_DEBUG

#ifdef WIN
#include<Windows.h>
#endif
#ifdef UNIX
#include <sys/time.h>
#endif

class clcnn
{
public:
	int execute_device(const char* infile);
	int execute_cpu(const char* infile);
	int retrieve_result(float* container, int lang);
	int predict();
	float get_execution_time();
	float getLoadTime();

	clcnn(const char* net_file);
	~clcnn();

private:
	int cpu_load_num;
	int device_load_num;

	int load(const char* net_file);
	void load_cpu();
	void load_device();
	cl_program load_program(cl_context context, cl_device_id device, const char* filename);
	
	float non_linear(int mode, float input);
	int getInput(const char* infile);
	int exe_conv_device(int* layer_pos, int flag);
	int exe_pool_device(int* layer_pos, int flag);
	int exe_fc_device(int* layer_pos, int flag);

	cl_platform_id *platforms;
	cl_device_id *devices;
	cl_uint platnum, devnum;
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_event timer;
	cl_ulong start, end, elapse;
	cl_kernel convp_sig_avg_full, convp_sig_avg_mc, convp_sig_avg_pc,fc_sig, fcp_sig;
	
	//varibles for the data structure on host
	int input_width, input_height, input_channel;
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
	//varible for DS on device
	cl_mem *d_net;
	cl_mem *d_bias;
	cl_mem *d_flt_offset;
	cl_mem *d_flt_sz_lk;//stands for size_link
	cl_mem d_inter_res_a;
	cl_mem d_inter_res_b;	
	float execution_time, load_time;
};

#endif

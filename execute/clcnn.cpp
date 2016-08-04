#include "clcnn.h"

clcnn::clcnn(const char* net_file)
{
	cpu_load_num = 0;
	device_load_num = 0;
	load_num = 0;

	layer_num = 0;
	MAX_INTER_RES = INPUT_WIDTH*INPUT_HEIGHT*INPUT_CHANNEL;

	load(net_file);
	layer_time = (float*)malloc(sizeof(float) * 7);
}

clcnn::~clcnn()
{
	free(layer_time);
	int i,j,k,m;
	if(cpu_load_num!=0)
	{
		for(i=0;i<layer_num-1;i++)
		{
			for(j=0;j<feature_num[i+1];j++)
			{
				for(k=0;k < inter_res_dim[i+1][1]; k++)
				{
					free(inter_res[i+1][j][k]);
				}
				free(inter_res[i+1][j]);
			}
			free(inter_res[i+1]);
		}
	}

	if(device_load_num!=0)
	{
		clReleaseMemObject(d_inter_res_a);
		clReleaseMemObject(d_inter_res_b);
	}
	//release input/ouput
	for(i=0;i<feature_num[0];i++)
	{
		for(j=0;j<inter_res_dim[0][1];j++)
		{
			free(inter_res[0][i][j]);
		}
		free(inter_res[0][i]);
	}
	free(inter_res[0]);
	free(inter_res[layer_num][0][0]);
	free(inter_res[layer_num][0]);
	free(inter_res[layer_num]);
	free(inter_res);
	//release the filters
	for (i = 0; i < layer_num; i++)
	{
		if (filters[i] != NULL){
			for (j = 0; j < feature_num[i + 1]; j++)
			{
				for (k = 0; k < feature_num[i]; k++)
				{
					if (filters[i][j][k] != NULL)
					{
						for (m = 0; m < kernel_dim[i][1]; m++)
						{
							free(filters[i][j][k][m]);
						}
						free(filters[i][j][k]);
					}
				}
				free(filters[i][j]);
			}
			free(bias[i]);
			free(filters[i]);
		}
		free(kernel_dim[i]);
		if(layer_type[i]==0)
		{
			free(link_info[i]);
			free(filter_offset[i]);
		}
		clReleaseMemObject(d_net[i]);
		clReleaseMemObject(d_bias[i]);
		clReleaseMemObject(d_flt_offset[i]);
		clReleaseMemObject(d_flt_sz_lk[i]);
	}
	for(i=0;i<layer_num+1;i++)
	{
		free(inter_res_dim[i]);
	}
	free(inter_res_dim);

	free(d_net);
	free(d_bias);
	free(d_flt_sz_lk);
	free(d_flt_offset);

	free(link_info);
	free(total_kernel);
	free(filter_offset);

	free(filters);
	free(bias);
	free(kernel_dim);
	//release others
	free(layer_type);
	free(feature_num);
	free(method);
	free(stride);
	for (i = 0; i < layer_num; i++)
	{
		free(launch_info[i]);
	}
	free(launch_info);
	free(max_group_size);

	//release the CL environment
	//clReleaseEvent(timer);
	clReleaseKernel(fcp_sig);
	clReleaseKernel(fc_sig);
	clReleaseKernel(convp_sig_avg_full);	
	clReleaseProgram(program);
	clReleaseCommandQueue(queue);
	clReleaseContext(context);
}

cl_program clcnn::load_program(cl_context context, cl_device_id device, const char* filename)
{
	std::ifstream in(filename, std::ios_base::binary);
	if (!in.good()) {
		return 0;
	}

	// get file length
	in.seekg(0, std::ios_base::end);
	size_t length = in.tellg();
	in.seekg(0, std::ios_base::beg);

	// read program source
	std::vector<char> data(length + 1);
	in.read(&data[0], length);
	data[length] = 0;

	// create and build program 
	const char* source = &data[0];
	cl_program program = clCreateProgramWithSource(context, 1, &source, 0, 0);
	if (program == 0) {
		std::cout << "create unsucc" << std::endl;
		return 0;
	}

	if (clBuildProgram(program, 0, 0, 0, 0, 0) != CL_SUCCESS) {
		std::cout << "build unsucc" << std::endl;
		char buildlog[16384];
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buildlog), buildlog, NULL);
		std::cout << buildlog << std::endl;
		return 0;
	}
	return program;
}

int clcnn::load(const char* net_file)
{
	if (load_num == 0)
	{
		///////////////////////////////////////
		//construct the DS on host
		int i, j, k, m, n;
		net_config = fopen(net_file, "r");
		fscanf(net_config, "%d", &layer_num);

		layer_type = (int*)malloc(sizeof(int)*layer_num);
		feature_num = (int*)malloc(sizeof(int)*(layer_num + 1));
		kernel_dim = (int**)malloc(sizeof(int*)*layer_num);
		stride = (int*)malloc(sizeof(int)*layer_num);
		method = (int*)malloc(sizeof(int)*layer_num);

		link_info = (int**)malloc(sizeof(int*)*layer_num);
		filter_offset = (int**)malloc(sizeof(int*)*layer_num);
		total_kernel = (int*)malloc(sizeof(int)*layer_num);

		inter_res_dim = (int**)malloc(sizeof(int*)*(layer_num + 1));
		bias = (float**)malloc(sizeof(float*)*layer_num);
		inter_res = (float****)malloc(sizeof(float***)*(layer_num + 1));
		filters = (float*****)malloc(sizeof(float****)*layer_num);
		for (i = 0; i < layer_num; i++)
		{
			//the convention for the x_dim varible is, [0] specify the width, [1]specify the height
			inter_res_dim[i] = (int*)malloc(sizeof(int) * 2);
			inter_res_dim[i][0] = 0;
			inter_res_dim[i][1] = 0;
			kernel_dim[i] = (int*)malloc(sizeof(int) * 2);//assume at present each layer has uniformed kernel size
		}
		inter_res_dim[layer_num] = (int*)malloc(sizeof(int) * 2);

		launch_info = (int**)malloc(sizeof(int*)*layer_num);
		for (i = 0; i < layer_num; i++)
		{
			launch_info[i] = (int*)malloc(sizeof(int) * 2);
			launch_info[i][0] = -1;
		}
		max_group_size = (int*)malloc(sizeof(int)*layer_num);
		for (i = 0; i < layer_num; i++)
		{
			max_group_size[i] = -1;
		}

		//the lenet net configure
		inter_res_dim[0][0] = INPUT_WIDTH;
		inter_res_dim[0][1] = INPUT_HEIGHT;
		feature_num[0] = INPUT_CHANNEL;

		for (i = 0; i < layer_num; i++)
		{
			fscanf(net_config, "%d", &layer_type[i]);
			if (layer_type[i] == 0)
			{
				if (DEBUG_MODE)
				{
					printf("read in a CONV layer\n");
				}
				int flt_size, front_feature_size, flt_w, flt_h, lstride;
				fscanf(net_config, "%d", &flt_size);
				fscanf(net_config, "%d", &front_feature_size);
				fscanf(net_config, "%d", &flt_w);
				fscanf(net_config, "%d", &flt_h);
				fscanf(net_config, "%d", &lstride);

				filters[i] = (float****)malloc(sizeof(float***)*flt_size);
				kernel_dim[i][0] = flt_w;
				kernel_dim[i][1] = flt_h;
				stride[i] = lstride;
				method[i] = -1;
				bias[i] = (float*)malloc(sizeof(float)*flt_size);
				feature_num[i + 1] = flt_size;
				inter_res_dim[i + 1][0] = (inter_res_dim[i][0] - flt_w + 1) / lstride;
				inter_res_dim[i + 1][1] = (inter_res_dim[i][1] - flt_h + 1) / lstride;//the user has the responsibility to make sure it can zhengchu
				if (inter_res_dim[i + 1][0] * inter_res_dim[i + 1][1] * flt_size > MAX_INTER_RES)
				{
					MAX_INTER_RES = inter_res_dim[i + 1][0] * inter_res_dim[i + 1][1] * flt_size;
				}
				link_info[i] = (int*)malloc(sizeof(int)*flt_size*(feature_num[i] + 1));
				filter_offset[i] = (int*)malloc(sizeof(int)*flt_size);
				total_kernel[i] = 0;

				//readin the kernel group
				int flt_num = 0;
				for (j = 0; j < flt_size; j++)
				{
					filters[i][j] = (float***)malloc(sizeof(float**)*front_feature_size);
					for (k = 0; k < front_feature_size; k++)
					{
						filters[i][j][k] = NULL;
					}
					if (j == 0)
						filter_offset[i][j] = 0;
					else
						filter_offset[i][j] = flt_num*flt_w*flt_h + filter_offset[i][j - 1];
					flt_num = 0;
					fscanf(net_config, "%d", &flt_num);
					link_info[i][j] = flt_num;
					total_kernel[i] += flt_num;
					for (k = 0; k < flt_num; k++)
					{
						int flt_squence;
						fscanf(net_config, "%d", &flt_squence);
						link_info[i][(k + 1) * flt_size + j] = inter_res_dim[i][0] * inter_res_dim[i][1] * flt_squence;
						filters[i][j][flt_squence] = (float**)malloc(sizeof(float*) * flt_h);
						for (m = 0; m < flt_h; m++)
						{
							filters[i][j][flt_squence][m] = (float*)malloc(sizeof(float) * flt_w);
							for (n = 0; n < flt_w; n++)
							{
								fscanf(net_config, "%f", &filters[i][j][flt_squence][m][n]);
							}
						}
					}
					//scan the bias item 
					fscanf(net_config, "%f", &bias[i][j]);
				}
			}
			else if (layer_type[i] == 1)
			{
				if (DEBUG_MODE)
				{
					printf("read in a FULL layer\n");
				}
				feature_num[i + 1] = 1;
				stride[i] = 0;
				method[i] = 0;

				link_info[i] = NULL;
				filter_offset[i] = NULL;
				total_kernel[i] = 1;

				int front_feature_num = feature_num[i];
				int input_num, output_num;
				fscanf(net_config, "%d", &input_num);
				fscanf(net_config, "%d", &output_num);
				inter_res_dim[i + 1][0] = output_num;
				inter_res_dim[i + 1][1] = 1;
				if (output_num > MAX_INTER_RES)
				{
					MAX_INTER_RES = output_num;
				}

				kernel_dim[i][0] = output_num;
				kernel_dim[i][1] = input_num;
				filters[i] = (float****)malloc(sizeof(float***) * 1);
				filters[i][0] = (float***)malloc(sizeof(float**) * front_feature_num);
				for (j = 0; j < front_feature_num; j++)
				{
					if (j == 0)
					{
						filters[i][0][j] = (float**)malloc(sizeof(float*)*input_num);
					}
					else
					{
						filters[i][0][j] = NULL;
					}
				}
				filters[i][0][0] = (float**)malloc(sizeof(float*)*input_num);
				bias[i] = (float*)malloc(sizeof(float)*output_num);
				for (j = 0; j < input_num; j++)
				{
					filters[i][0][0][j] = (float*)malloc(sizeof(float)*output_num);
					for (k = 0; k < output_num; k++)
					{
						fscanf(net_config, "%f", &filters[i][0][0][j][k]);
					}
				}
				for (k = 0; k < output_num; k++)
				{
					fscanf(net_config, "%f", &bias[i][k]);
				}
			}

			else if (layer_type[i] == 2)
			{
				if (DEBUG_MODE)
				{
					printf("read in a pooling layer\n");
				}
				int lmethod = 0;//default methid is MAX
				stride[i] = 0;
				int front_feature_size, width, height;
				fscanf(net_config, "%d", &lmethod);
				fscanf(net_config, "%d", &front_feature_size);
				fscanf(net_config, "%d", &width);
				fscanf(net_config, "%d", &height);

				inter_res_dim[i + 1][0] = inter_res_dim[i][0] / width;
				inter_res_dim[i + 1][1] = inter_res_dim[i][1] / height;
				method[i] = lmethod;

				filters[i] = NULL;
				bias[i] = NULL;
				feature_num[i + 1] = front_feature_size;

				kernel_dim[i][0] = width;
				kernel_dim[i][1] = height;

				filter_offset[i] = NULL;
				link_info[i] = NULL;
				total_kernel[i] = 0;
			}
			else
			{
				printf("unrecognized layer percepted, exiting now\n");
				if (DEBUG_MODE)
				{
					getchar();
				}
				return -1;
			}
		}
		fclose(net_config);

		////////////////////////////////////////////////
		//setting up the CL enviroment
		cl_int err;
		cl_uint num, numm;
		size_t size;
		char* name;

		err = clGetPlatformIDs(0, 0, &num);
		if (err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
		}
		platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id)*num);
		err = clGetPlatformIDs(num, platforms, NULL);
		if (err != CL_SUCCESS) {
			std::cerr << "Unable to get platforms\n";
		}
		for (i = 0; i < num; i++)
		{
			err = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_NAME,
				0,
				0,
				&size);
			name = (char*)malloc(size);
			err = clGetPlatformInfo(
				platforms[i],
				CL_PLATFORM_NAME,
				size,
				name,
				NULL);
			printf("plat %d: %s\n", i, name);
			free(name);
		}
		std::cout << "please spicify the platform to use" << std::endl;
		platnum = TEST_PLAT;
#ifdef INTERACTIVE
		std::cin >> platnum;
#endif
		err = clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, 0, 0, &num);
		devices = (cl_device_id*)malloc(sizeof(cl_device_id)*num);
		err = clGetDeviceIDs(platforms[platnum], CL_DEVICE_TYPE_ALL, num, devices, 0);
		printf("there are %u devices under this platform:\n", num);
		for (i = 0; i < num; i++)
		{
			err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, 0, NULL, &size);
			name = (char*)malloc(size);
			err = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, size, name, 0);
			err = clGetDeviceInfo(devices[i], CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &numm, &size);
			printf("device %u: %s\n\tMAX compute units: %u\n", i, name, numm);
			free(name);
		}
		std::cout << "please spicify the device to use" << std::endl;
		devnum = TEST_DEV;
#ifdef INTERACTIVE
		std::cin >> devnum;
#endif
		cl_context_properties prop[] = { CL_CONTEXT_PLATFORM, reinterpret_cast<cl_context_properties>(platforms[platnum]), 0 };
		context = clCreateContext(prop, 1, &devices[devnum], NULL, NULL, &err);
		if (context == 0) {
			std::cerr << "Can't create OpenCL context\n";
		}
		queue = clCreateCommandQueue(context, devices[devnum], CL_QUEUE_PROFILING_ENABLE, 0);
		if (queue == 0) {
			std::cerr << "Can't create command queue\n";
			//clReleaseContext(context);
			//return 0;
		}

		program = load_program(context, devices[devnum], "shader.cl");
		if (program == 0) {
			std::cerr << "Can't load or build program\n";
		}

		convp_sig_avg_full = clCreateKernel(program, "convp_sig_avg_full", 0);
		if (convp_sig_avg_full == 0) {
			std::cerr << "Can't load kernel convp_sig_avg_full\n";
		}
		convp_sig_avg_mc = clCreateKernel(program, "convp_sig_avg_mc", 0);
		if (convp_sig_avg_mc == 0) {
			std::cerr << "Can't load kernel convp_sig_avg_mc\n";
		}
		convp_sig_avg_pc = clCreateKernel(program, "convp_sig_avg_pc", 0);
		if (convp_sig_avg_pc == 0) {
			std::cerr << "Can't load kernel convp_sig_avg_pc\n";
		}



		fcp_sig = clCreateKernel(program, "fcp_sig", 0);
		if (fcp_sig == 0) {
			std::cerr << "Can't load kernel fcp_sig\n";
		}
		fc_sig = clCreateKernel(program, "fc_sig", 0);
		if (fc_sig == 0) {
			std::cerr << "Can't load kernel fc_sig\n";
		}

		free(platforms);
		free(devices);

		//////////////////////////////////////////////
		//construct the DS on device mem
		d_net = (cl_mem*)malloc(sizeof(cl_mem)*layer_num);
		d_bias = (cl_mem*)malloc(sizeof(cl_mem)*layer_num);
		d_flt_offset = (cl_mem*)malloc(sizeof(cl_mem)*layer_num);
		d_flt_sz_lk = (cl_mem*)malloc(sizeof(cl_mem)*layer_num);

		for (i = 0; i<layer_num; i++)
		{
			if (layer_type[i] == 0)
			{
				d_net[i] = clCreateBuffer(
					context,
					CL_MEM_READ_ONLY,
					sizeof(float)*total_kernel[i] * kernel_dim[i][0] * kernel_dim[i][1],
					NULL,
					NULL);
				int cursor = 0;
				for (j = 0; j<feature_num[i + 1]; j++)
				{
					for (k = 0; k<feature_num[i]; k++)
					{
						if (filters[i][j][k] != NULL)
						{
							for (m = 0; m<kernel_dim[i][1]; m++)
							{
								err = clEnqueueWriteBuffer(
									queue,
									d_net[i],
									CL_TRUE,
									cursor*sizeof(float),
									sizeof(float)*kernel_dim[i][0],
									filters[i][j][k][m],
									NULL,
									NULL,
									NULL);
								cursor += kernel_dim[i][0];
							}
						}
					}
				}
				d_bias[i] = clCreateBuffer(
					context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(float)*feature_num[i + 1],
					bias[i],
					NULL);
				d_flt_offset[i] = clCreateBuffer(
					context,
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					sizeof(int)*feature_num[i + 1],
					filter_offset[i],
					&err);
				d_flt_sz_lk[i] = clCreateBuffer(
					context,
					CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
					sizeof(int)*feature_num[i + 1] * (1 + feature_num[i]),
					link_info[i],
					&err);
			}

			if (layer_type[i] == 1)
			{
				int cursor = 0;
				d_net[i] = clCreateBuffer(
					context,
					CL_MEM_READ_ONLY,
					sizeof(float)*kernel_dim[i][0] * kernel_dim[i][1],
					NULL,
					NULL);
				for (j = 0; j<kernel_dim[i][1]; j++)
				{
					//attentino, here load by coloum is apparent more effective, but to comply with old struture, I need to load by row
					err = clEnqueueWriteBuffer(
						queue,
						d_net[i],
						CL_TRUE,
						sizeof(float)*cursor,
						sizeof(float)*kernel_dim[i][0],
						filters[i][0][0][j],
						NULL,
						NULL,
						NULL);
					cursor += kernel_dim[i][0];
				}
				d_bias[i] = clCreateBuffer(
					context,
					CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
					sizeof(float)*kernel_dim[i][0],
					bias[i],
					NULL);
				d_flt_sz_lk[i] = NULL;
				d_flt_offset[i] = NULL;
			}

			if (layer_type[i] == 2)
			{
				d_net[i] = NULL;
				d_bias[i] = NULL;
				d_flt_offset[i] = NULL;
				d_flt_sz_lk[i] = NULL;
			}
		}


		///////////////////////////////////////////////////////
		//construct the DS to store input data
		inter_res[0] = (float***)malloc(sizeof(float**)*feature_num[0]);
		inter_res[0][0] = (float**)malloc(sizeof(float*)*inter_res_dim[0][1]);
		for (i = 0; i < inter_res_dim[0][1]; i++)
		{
			inter_res[0][0][i] = (float*)malloc(sizeof(float)*inter_res_dim[0][0]);
		}
		//and the result
		inter_res[layer_num] = (float***)malloc(sizeof(float**) * 1);
		inter_res[layer_num][0] = (float**)malloc(sizeof(float*) * 1);
		inter_res[layer_num][0][0] = (float*)malloc(sizeof(float)*inter_res_dim[layer_num][0]);
		load_num++;
	}
}


int clcnn::getInput()
{
	//this is the function to convert the image to 
	//the format this class used

	//currently is only a tmp realization, which use
	//a file as input
	FILE* input_data;
	int i, j, k;
	input_data = fopen("test.cdat", "r");
	int input_channel, input_width, input_height;
	fscanf(input_data, "%d", &input_channel);
	fscanf(input_data, "%d", &input_width);
	fscanf(input_data, "%d", &input_height);
	if (feature_num[0] != input_channel || inter_res_dim[0][0] != input_width || inter_res_dim[0][1] != input_height)
	{
		printf("input data mismatch with the net, exiting now\n");
		if (DEBUG_MODE)
		{
			getchar();
		}
		return -1;
	}
	for (i = 0; i < input_channel; i++)
	{
		for (j = 0; j < input_height; j++)
		{
			for (k = 0; k < input_width; k++)
			{
				fscanf(input_data, "%f", &inter_res[0][i][j][k]);
#ifdef VISUAL
				if (inter_res[0][i][j][k]>0.1)
				{
					printf("*");
				}
				else
				{
					printf(" ");
				}
#endif
			}
#ifdef VISUAL
			printf("\n");
#endif
		}
	}
	fclose(input_data);
}

float clcnn::getKernelTime()
{
	printf("kernel time: %f us\n",kernel_time);
	/*
	printf("the execution time for each layer is:\n");
	int i;
	for (i = 0; i < 7; i++)
	{
		printf("%f ", layer_time[i]);
	}
	printf("\n");
	*/
	return kernel_time;
}

float clcnn::getCPUTime()
{
	printf("the execution time on cpu is %f us\n",CPUtime);
	return CPUtime;
}

int clcnn::execute_cpu()
{
	if(cpu_load_num == 0)
	{
		load_cpu();
		cpu_load_num++;
	}
#ifdef WIN
	LARGE_INTEGER large_int;
	double diff;
	__int64 c1, c2;

	QueryPerformanceFrequency(&large_int);
	diff = large_int.QuadPart;
	QueryPerformanceCounter(&large_int);
	c1 = large_int.QuadPart;
#endif
#ifdef UNIX
	long star, endd;
	struct timeval tv;
	gettimeofday(&tv, NULL);
	star = tv.tv_usec;
#endif
	int i,j,k,m,n;
	for(i=0;i<layer_num;i++)
	{
		if(layer_type[i]==0)
		{
			if (DEBUG_MODE){
				printf("execution convlution\n");
			}
			int pre_feature, next_feature;
			int ker_w, ker_h,strid;
			int res_w, res_h;
			res_w = inter_res_dim[i + 1][0];
			res_h = inter_res_dim[i + 1][1];
			pre_feature = feature_num[i];
			next_feature = feature_num[i+1];
			ker_w = kernel_dim[i][0];
			ker_h = kernel_dim[i][1];
			strid = stride[i];
			for(j=0;j<next_feature;j++)
			{				
				for(k=0;k<pre_feature;k++)
				{
					if(filters[i][j][k]!=NULL)
					{						
						for(m=0;m<res_h;m++)
						{
							for(n=0;n<res_w;n++)
							{
								float tmp_res= 0.0;
								int x, y;
								for(x=0;x<ker_h;x++)
								{
									for(y=0;y<ker_w;y++)
									{
										tmp_res += filters[i][j][k][x][y]*inter_res[i][k][m*strid+x][n*strid+y];
									}
								}
								inter_res[i+1][j][m][n] += tmp_res;
							}
						}

					}
				}
				for(m=0;m<res_h;m++)
				{
					for(n=0;n<res_w;n++)
					{
						inter_res[i+1][j][m][n] = non_linear(0,inter_res[i+1][j][m][n] + bias[i][j]);
					}
				}
			}
		}
		else if(layer_type[i]==2)
		{
			if (DEBUG_MODE)
			{
				printf("execution pooling\n");
			}
			int ker_w, ker_h;
			ker_w = kernel_dim[i][0];
			ker_h = kernel_dim[i][1];
			int lmethod = method[i];
			float result;
			int x, y;
			for(j=0;j<feature_num[i];j++)
			{
				for(m=0;m<inter_res_dim[i+1][1];m++)
				{
					for(n=0;n<inter_res_dim[i+1][0];n++)
					{
						result = 0.0;
						for(x=0;x<ker_h;x++)
						{
							for(y=0;y<ker_w;y++)
							{
								if (lmethod==0)
								{
									if(inter_res[i][j][m*ker_h+x][n*ker_w+y]>result)
									{
										result = inter_res[i][j][m*ker_h+x][n*ker_w+y];
									}
								}
								else if (lmethod == 1)
								{
									result += inter_res[i][j][m*ker_h + x][n*ker_w + y];
								}
							}
						}
						if (lmethod == 1)
						{
							result = result / (ker_h*ker_w);
						}
						inter_res[i + 1][j][m][n] = result;
					}
				}
			}
		}
		else
		{
			if (DEBUG_MODE)
			{
				printf("excution FULL conection\n");
			}
			//first flatten the front data
			float* local_input = NULL;
			float local_res = 0.0;
			if(inter_res_dim[i][1]!=1 || feature_num[i]!=1)
			{
				int pre_feature, pre_res_w, pre_res_h,input_dim;
				pre_feature = feature_num[i];
				pre_res_w = inter_res_dim[i][0];
				pre_res_h = inter_res_dim[i][1];
				
				input_dim = pre_res_h*pre_res_w*pre_feature;
				if(input_dim != kernel_dim[i][1])
				{
					printf("FULL layer mismatch with previous dimension, exiting now\n");
					printf("%d %d %d\n", pre_feature, pre_res_h, pre_res_w);
					printf("%d", input_dim);
					if (DEBUG_MODE)
					{
						getchar();
					}
					return -1;
				}
				local_input = (float*)malloc(sizeof(float)*input_dim);
				for(j=0;j<pre_feature;j++)
				{			
					for (m = 0; m < pre_res_h; m++)
					{
						for (n = 0; n < pre_res_w; n++)
						{
							local_input[j*pre_res_h*pre_res_w + m*pre_res_w + n] = inter_res[i][j][n][m];
							//attention here, the reason for [n][m] not [m][n] is the netdata i retrieved from matlab
							//flat the data by coloum first, not row first, that's 1 2 is flatted into 1 3 2 4 
							//                                                     3 4
						}
					}
				}
				for (j = 0; j<kernel_dim[i][0]; j++)
				{
					local_res = 0.0;
					for (k = 0; k<kernel_dim[i][1]; k++)
					{
						local_res += local_input[k] * filters[i][0][0][k][j];
					}
					local_res = non_linear(0,local_res + bias[i][j]);
					inter_res[i + 1][0][0][j] = local_res;
				}
				free(local_input);				
			}
			else
			{
				for (j = 0; j<kernel_dim[i][0]; j++)
				{
					local_res = 0.0;
					for (k = 0; k<kernel_dim[i][1]; k++)
					{
						local_res += inter_res[i][0][0][k] * filters[i][0][0][k][j];
					}
					local_res = non_linear(0,local_res + bias[i][j]);
					inter_res[i + 1][0][0][j] = local_res;
				}
			}		
		}
	}
#ifdef WIN
	QueryPerformanceCounter(&large_int);
	c2 = large_int.QuadPart;
	CPUtime = (float)((c2-c1)*1e06/diff);
#endif
#ifdef UNIX
	gettimeofday(&tv, NULL);
	endd = tv.tv_usec;
	CPUtime = (float)(endd - star);
#endif

	return 0;
}

float clcnn::non_linear(int mode, float input)
{
	if (mode == 0)
	{
		return 1.0 / (1.0 + exp(0.0 - input));
	}

}

int clcnn::execute_device()
{
	int i,j,m;
	size_t size;
	int flag = 0;//mark which pool is being used as input
	kernel_time = 0.0;
	elipse = 0;
	//0 represent a is input
	cl_int err;

	if(device_load_num == 0)
	{
		load_device();
		device_load_num++;
	}
	//moving the input data to device mem
	m = 0;//m act as a cursor here
	for(i=0;i<feature_num[0];i++)
	{
		for(j=0;j<inter_res_dim[0][1];j++)
		{
			err = clEnqueueWriteBuffer(
				queue,
				d_inter_res_a,
				CL_TRUE,
				sizeof(float)*m,
				sizeof(float)*inter_res_dim[0][0],
				inter_res[0][i][j],
				NULL,
				NULL,
				&timer);
			if (err != CL_SUCCESS)
			{
				printf("error written original data\n");
			}
			m += inter_res_dim[0][0];
			clFinish(queue);
			err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &size);
			if (err != CL_SUCCESS)
			{
				printf("profile error\n");
			}
			err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &size);
			elipse += (cl_ulong)(end - start);
		}
	}

	//start execution
	for(i=0;i<layer_num;i++)
	{
		printf("start execution an layer:\t");
		if(layer_type[i] == 0)
		{
			printf("conv layer ");
			err = exe_conv_device(&i,flag);
			flag = !flag;
		}
		else if(layer_type[i] == 1)
		{
			err = exe_fc_device(&i, flag);
			flag = !flag;
		}
		else if (layer_type[i] == 2)
		{
			err = exe_pool_device(&i, flag);
			flag = !flag;
			printf("havent's support pure pool layer\n");
		}
		else
		{
			printf("unrecognized layer %d, exiting now\n", i);
			return -1;
		}
	}

	if(flag)
	{
		err = clEnqueueReadBuffer(
		queue,
		d_inter_res_b,
		CL_TRUE, 
		0,
		sizeof(float)*inter_res_dim[layer_num][0],
		inter_res[layer_num][0][0],
		NULL,
		NULL,
		NULL);
	}
	else
	{
		err = clEnqueueReadBuffer(
		queue,
		d_inter_res_a,
		CL_TRUE, 
		0,
		sizeof(float)*inter_res_dim[layer_num][0],
		inter_res[layer_num][0][0],
		NULL,
		NULL,
		NULL);
	}
	kernel_time = (float)(elipse*1e-03);
	return 0;
}

int clcnn::exe_conv_device(int* layer_pos, int flag)
{
	int i = *layer_pos;
	cl_int err;
	size_t size;
	if (layer_type[i + 1] == 2)
	{
		printf("with fused mode\n");
		//start fused mode
		cl_int conv_w, conv_h, ker_w, ker_h, pool_w, pool_h, lstride;
		cl_int out_w, out_h,group_num;
		cl_int err;
		int max_grp_sz;
		int launch_mode, launch_batch_size;
		int linknum;

		ker_w = kernel_dim[i][0];
		ker_h = kernel_dim[i][1];
		pool_w = kernel_dim[i + 1][0];
		pool_h = kernel_dim[i + 1][1];
		lstride = stride[i];
		out_w = inter_res_dim[i + 2][0];
		out_h = inter_res_dim[i + 2][1];
		conv_w = inter_res_dim[i+1][0];
		conv_h = inter_res_dim[i+1][1];
		group_num = feature_num[i + 1];
		linknum = link_info[i][0];//warning, this should only be used when every kernelgroup has uniform forward link num. 		

		if ((max_grp_sz = max_group_size[i]) == -1)
		{
			cl_device_id dev;
			size_t max;
			err = clGetContextInfo(
				context,
				CL_CONTEXT_DEVICES,
				sizeof(cl_device_id),
				&dev,
				NULL);

			err = clGetKernelWorkGroupInfo(
				convp_sig_avg_full,
				dev,
				CL_KERNEL_WORK_GROUP_SIZE,
				sizeof(size_t),
				&max,
				NULL);
			max_grp_sz = (int)max;
			err = clGetKernelWorkGroupInfo(
				convp_sig_avg_mc,
				dev,
				CL_KERNEL_WORK_GROUP_SIZE,
				sizeof(size_t),
				&max,
				NULL);
			if ((int)max != max_grp_sz)
			{
				printf("warnning, ununiformed Max_GROUP_SIZE in each kernel\n");
				if ((int)max < max_grp_sz)
				{
					max_grp_sz = (int)max;
				}
			}
			err = clGetKernelWorkGroupInfo(
				convp_sig_avg_pc,
				dev,
				CL_KERNEL_WORK_GROUP_SIZE,
				sizeof(size_t),
				&max,
				NULL);
			if ((int)max != max_grp_sz)
			{
				printf("warnning, ununiformed Max_GROUP_SIZE in each kernel\n");
				if ((int)max < max_grp_sz)
				{
					max_grp_sz = (int)max;
				}
			}
			max_group_size[i] = max_grp_sz;
		}
#ifdef FLEX_THREAD_NUM
		//for temporial test purpose
		if(i==0)
		{
			max_grp_sz = 330;
			max_group_size[i] = 330;

		}		
#endif
		if((launch_mode = launch_info[i][0]) == -1)
		{
			if(max_grp_sz >= conv_w*conv_h*linknum)
			{
				launch_info[i][0] = 1; //means full speed kernel;
				launch_mode = 1;
			}
			else if(max_grp_sz > conv_w*conv_h)
			{
				launch_info[i][0] = 2; //means multi channel process at once, mc;
				launch_mode = 2;
				launch_info[i][1] = max_grp_sz/(conv_h * conv_w);//how many input channel can be compute simontaneously
			}
			else
			{
				launch_info[i][0] = 3;//partial channel at once, pc;
				launch_mode = 3;
				if(max_grp_sz >= out_w*out_h)
				{
					printf("\tentering alaigned partial-chennel mode\n");
					launch_info[i][1] = 0;
				}
				else
				{
					printf("\tenter standard partial-channel mode\n");
					launch_info[i][1] = 2;
				}
			}
		}
		launch_batch_size = launch_info[i][1];

		if(launch_mode == 1)
		{
			printf("\tentering full speed mode\n");
			clSetKernelArg(convp_sig_avg_full, 0, sizeof(cl_int), &conv_w);
			clSetKernelArg(convp_sig_avg_full, 1, sizeof(cl_int), &conv_h);
			clSetKernelArg(convp_sig_avg_full, 2, sizeof(cl_int), &out_w);
			clSetKernelArg(convp_sig_avg_full, 3, sizeof(cl_int), &out_h);
			clSetKernelArg(convp_sig_avg_full, 4, sizeof(cl_int), &ker_w);
			clSetKernelArg(convp_sig_avg_full, 5, sizeof(cl_int), &ker_h);
			clSetKernelArg(convp_sig_avg_full, 6, sizeof(cl_int), &pool_w);
			clSetKernelArg(convp_sig_avg_full, 7, sizeof(cl_int), &pool_h);
			clSetKernelArg(convp_sig_avg_full, 8, sizeof(cl_int), &lstride);
			clSetKernelArg(convp_sig_avg_full, 9, sizeof(cl_mem), &d_net[i]);
			clSetKernelArg(convp_sig_avg_full, 10, sizeof(cl_mem), &d_bias[i]);
			clSetKernelArg(convp_sig_avg_full, 11, sizeof(cl_mem), &d_flt_offset[i]);
			clSetKernelArg(convp_sig_avg_full, 12, sizeof(cl_mem), &d_flt_sz_lk[i]);
			//local
			//clSetKernelArg(convp_sig_avg_full, 15, sizeof(float)*feature_num[i] * linknum * kernel_dim[i][0] * kernel_dim[i][1], NULL);
			//clSetKernelArg(convp_sig_avg_full, 16, sizeof(float)*feature_num[i] * inter_res_dim[i][0] * inter_res_dim[i][1], NULL);
			clSetKernelArg(convp_sig_avg_full, 15, sizeof(float) * 1, NULL);
			clSetKernelArg(convp_sig_avg_full, 16, sizeof(float) * 1, NULL);
			clSetKernelArg(convp_sig_avg_full, 17, sizeof(float)*conv_h*conv_w*linknum, NULL);
			clSetKernelArg(convp_sig_avg_full, 18, sizeof(float) * 1, NULL);
			if (flag)
			{
				clSetKernelArg(convp_sig_avg_full, 13, sizeof(cl_mem), &d_inter_res_b);
				clSetKernelArg(convp_sig_avg_full, 14, sizeof(cl_mem), &d_inter_res_a);
			}
			else
			{
				clSetKernelArg(convp_sig_avg_full, 14, sizeof(cl_mem), &d_inter_res_b);
				clSetKernelArg(convp_sig_avg_full, 13, sizeof(cl_mem), &d_inter_res_a);
			}
			//for ed0
			//size_t global_work_size[2] = { kernel_dim[i][0] * group_num, kernel_dim[i][1] };
			//size_t local_work_size[2] = { kernel_dim[i][0], kernel_dim[i][1] };
			//for ed1
			size_t local_work_size[2] = { inter_res_dim[i + 1][0] * linknum, inter_res_dim[i + 1][1] };
			size_t global_work_size[2] = { inter_res_dim[i + 1][0] * linknum*group_num, inter_res_dim[i + 1][1] };
			err = clEnqueueNDRangeKernel(queue, convp_sig_avg_full, 2, 0, global_work_size, local_work_size, NULL, NULL, &timer);			
		}
		else if(launch_mode == 2)
		{
			printf("\tentering multi-channel mode\n");
			cl_int many = linknum/(max_grp_sz/(conv_h * conv_w));//how many channels a thread should compute
			if (linknum % (max_grp_sz / (conv_h * conv_w)) != 0)
			{
				many += 1;
			}//actually a hand write cell() for int.
			#ifdef FLEX_THREAD_NUM
			printf("%d %d\n",many, launch_batch_size);
			#endif
			size_t local_work_size[2] = {conv_h*launch_batch_size, conv_w};
			size_t global_work_size[2] = {conv_h*launch_batch_size*group_num, conv_w};
			
			clSetKernelArg(convp_sig_avg_mc, 0, sizeof(cl_int), &conv_w);
			clSetKernelArg(convp_sig_avg_mc, 1, sizeof(cl_int), &conv_h);
			clSetKernelArg(convp_sig_avg_mc, 2, sizeof(cl_int), &out_w);
			clSetKernelArg(convp_sig_avg_mc, 3, sizeof(cl_int), &out_h);
			clSetKernelArg(convp_sig_avg_mc, 4, sizeof(cl_int), &ker_w);
			clSetKernelArg(convp_sig_avg_mc, 5, sizeof(cl_int), &ker_h);
			clSetKernelArg(convp_sig_avg_mc, 6, sizeof(cl_int), &pool_w);
			clSetKernelArg(convp_sig_avg_mc, 7, sizeof(cl_int), &pool_h);
			clSetKernelArg(convp_sig_avg_mc, 8, sizeof(cl_int), &lstride);
			clSetKernelArg(convp_sig_avg_mc, 9, sizeof(cl_int), &many);
			clSetKernelArg(convp_sig_avg_mc, 10, sizeof(cl_mem), &d_net[i]);
			clSetKernelArg(convp_sig_avg_mc, 11, sizeof(cl_mem), &d_bias[i]);
			clSetKernelArg(convp_sig_avg_mc, 12, sizeof(cl_mem), &d_flt_offset[i]);
			clSetKernelArg(convp_sig_avg_mc, 13, sizeof(cl_mem), &d_flt_sz_lk[i]);
			clSetKernelArg(convp_sig_avg_mc, 16, sizeof(float)*conv_w*conv_h*launch_batch_size,NULL);
			if (flag)
			{
				clSetKernelArg(convp_sig_avg_mc, 14, sizeof(cl_mem), &d_inter_res_b);
				clSetKernelArg(convp_sig_avg_mc, 15, sizeof(cl_mem), &d_inter_res_a);
			}
			else
			{
				clSetKernelArg(convp_sig_avg_mc, 15, sizeof(cl_mem), &d_inter_res_b);
				clSetKernelArg(convp_sig_avg_mc, 14, sizeof(cl_mem), &d_inter_res_a);
			}
			err = clEnqueueNDRangeKernel(queue,convp_sig_avg_mc,2,0,global_work_size,local_work_size,0,NULL,&timer);
			if(err != CL_SUCCESS)
			{
				printf("error enqueue using mode 2, %d\n",err);
			}
		}
		else if(launch_mode == 3)
		{
			if(launch_info[i][1] == 0)
			{
				size_t local_work_size[2] = {out_h, out_w};
				size_t global_work_size[2] = {out_h*group_num, out_w};

				clSetKernelArg(convp_sig_avg_pc, 0, sizeof(cl_int), &conv_w);
				clSetKernelArg(convp_sig_avg_pc, 1, sizeof(cl_int), &conv_h);
				clSetKernelArg(convp_sig_avg_pc, 2, sizeof(cl_int), &out_w);
				clSetKernelArg(convp_sig_avg_pc, 3, sizeof(cl_int), &out_h);
				clSetKernelArg(convp_sig_avg_pc, 4, sizeof(cl_int), &ker_w);
				clSetKernelArg(convp_sig_avg_pc, 5, sizeof(cl_int), &ker_h);
				clSetKernelArg(convp_sig_avg_pc, 6, sizeof(cl_int), &pool_w);
				clSetKernelArg(convp_sig_avg_pc, 7, sizeof(cl_int), &pool_h);
				clSetKernelArg(convp_sig_avg_pc, 8, sizeof(cl_int), &lstride);	
				clSetKernelArg(convp_sig_avg_pc, 9, sizeof(cl_mem), &d_net[i]);
				clSetKernelArg(convp_sig_avg_pc, 10, sizeof(cl_mem), &d_bias[i]);
				clSetKernelArg(convp_sig_avg_pc, 11, sizeof(cl_mem), &d_flt_offset[i]);
				clSetKernelArg(convp_sig_avg_pc, 12, sizeof(cl_mem), &d_flt_sz_lk[i]);
				clSetKernelArg(convp_sig_avg_pc, 15, sizeof(float)*out_w*out_h, NULL);
				if (flag)
				{
					clSetKernelArg(convp_sig_avg_pc, 13, sizeof(cl_mem), &d_inter_res_b);
					clSetKernelArg(convp_sig_avg_pc, 14, sizeof(cl_mem), &d_inter_res_a);
				}
				else
				{
					clSetKernelArg(convp_sig_avg_pc, 14, sizeof(cl_mem), &d_inter_res_b);
					clSetKernelArg(convp_sig_avg_pc, 13, sizeof(cl_mem), &d_inter_res_a);
				}
				err = clEnqueueNDRangeKernel(queue,convp_sig_avg_pc,2,0,global_work_size,local_work_size,0,NULL,&timer);
				if(err != CL_SUCCESS)
				{
					printf("error enqueue using mode 3, %d\n",err);
				}
			}
			else
			{
				printf("haven't suppported yet\n");
			}

		}
		else
		{
			printf("invalid launch mode exiting now\n");
			return -1;
		}
		
		clFinish(queue);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &size);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &size);
		elipse += (cl_ulong)(end - start);
		layer_time[i] = ((cl_ulong)(end - start))*1e-03;
		(*layer_pos)++;

#ifdef DEV_DEBUG
		float* itck;
		itck = (float*)malloc(sizeof(float)*group_num*out_w*out_h);
		if (flag)
		{
			err = clEnqueueReadBuffer(
				queue,
				d_inter_res_a,
				CL_TRUE,
				0,
				sizeof(float)*group_num*out_h*out_w,
				itck,
				0,
				NULL,
				NULL);
		}
		else
		{
			err = clEnqueueReadBuffer(
				queue,
				d_inter_res_b,
				CL_TRUE,
				0,
				sizeof(float)*group_num*out_h*out_w,
				itck,
				0,
				NULL,
				NULL);
		}
		clFinish(queue);
		printf("print the intemediate result from %d\n", i);
		int x, y, z;
		for (x = 0; x<group_num; x++)
		{
			for (y = 0; y<out_h; y++)
			{
				for (z = 0; z<out_w; z++)
				{
					printf("%f ", itck[x*out_h*out_w + y*out_w + z]);
				}
				printf("\n");
			}
			printf("==========================\n");
		}
		free(itck);
#endif
	}
	else
	{
		//normal conv
		printf("haven't supported\n");
	}
	return 0;
}

int clcnn::exe_fc_device(int* layer_pos, int flag)
{
	int i = *layer_pos;
	cl_int err;
	size_t size;
	if (layer_type[i - 1] != 1)
	{
		printf("fcp layer\n");
		clSetKernelArg(fcp_sig, 0, sizeof(int), &kernel_dim[i][0]);
		clSetKernelArg(fcp_sig, 1, sizeof(int), &feature_num[i]);
		clSetKernelArg(fcp_sig, 2, sizeof(int), &inter_res_dim[i][0]);
		clSetKernelArg(fcp_sig, 3, sizeof(int), &inter_res_dim[i][1]);
		clSetKernelArg(fcp_sig, 4, sizeof(cl_mem), &d_net[i]);
		clSetKernelArg(fcp_sig, 5, sizeof(cl_mem), &d_bias[i]);
		clSetKernelArg(fcp_sig, 8, sizeof(float)*inter_res_dim[i][0], NULL);
		if (flag)
		{
			clSetKernelArg(fcp_sig, 6, sizeof(cl_mem), &d_inter_res_b);
			clSetKernelArg(fcp_sig, 7, sizeof(cl_mem), &d_inter_res_a);
		}
		else
		{
			clSetKernelArg(fcp_sig, 6, sizeof(cl_mem), &d_inter_res_a);
			clSetKernelArg(fcp_sig, 7, sizeof(cl_mem), &d_inter_res_b);
		}

		size_t global_work_size[1] = { kernel_dim[i][0] * inter_res_dim[i][0] };
		size_t local_work_size[1] = { inter_res_dim[i][0] };
		err = clEnqueueNDRangeKernel(
			queue,
			fcp_sig,
			1,
			0,
			global_work_size,
			local_work_size,
			NULL,
			NULL,
			&timer);
		if (err != CL_SUCCESS)
		{
			printf("error execute fcp layer%d\n",err);
		}
		clFinish(queue);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &size);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &size);
		elipse += (cl_ulong)(end - start);
		layer_time[i] = ((cl_ulong)(end - start))*1e-03;
	}
	else
	{
		printf("pure fc layer\n");
		clSetKernelArg(fc_sig, 0, sizeof(int), &kernel_dim[i][1]);
		clSetKernelArg(fc_sig, 1, sizeof(cl_mem), &d_net[i]);
		clSetKernelArg(fc_sig, 2, sizeof(cl_mem), &d_bias[i]);
		if (flag)
		{
			clSetKernelArg(fc_sig, 3, sizeof(cl_mem), &d_inter_res_b);
			clSetKernelArg(fc_sig, 4, sizeof(cl_mem), &d_inter_res_a);
		}
		else
		{
			clSetKernelArg(fc_sig, 3, sizeof(cl_mem), &d_inter_res_a);
			clSetKernelArg(fc_sig, 4, sizeof(cl_mem), &d_inter_res_b);
		}

		size_t global_work_size[1] = { kernel_dim[i][0] };
		size_t local_work_size[1] = { 1 };
		err = clEnqueueNDRangeKernel(
			queue,
			fc_sig,
			1,
			0,
			global_work_size,
			local_work_size,
			NULL,
			NULL,
			&timer);
		if (err != CL_SUCCESS)
		{
			printf("error enqueue pure fc layer %d\n", err);
		}
		clFinish(queue);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, &size);
		err = clGetEventProfilingInfo(timer, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, &size);
		elipse += (cl_ulong)(end - start);
		layer_time[i] = ((cl_ulong)(end - start))*1e-03;
	}
	
	return 0;
	
}

int clcnn::exe_pool_device(int* layer_pos, int flag)
{
	printf("havent's support pure pool layer\n");
	return 0;
}


void clcnn::load_device()
{
	d_inter_res_a = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*MAX_INTER_RES, NULL, NULL);
	d_inter_res_b = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float)*MAX_INTER_RES, NULL, NULL);

}

void clcnn::load_cpu()
{
	int i,j,k,m,n;
	for (i = 0; i < layer_num-1; i++)
	{
		if (layer_type[i] == 0)
		{
			//conv:
			inter_res[i + 1] = (float***)malloc(sizeof(float**)*feature_num[i+1]);
			for (j = 0; j < feature_num[i + 1]; j++)
			{
				inter_res[i + 1][j] = (float**)malloc(sizeof(float*)*inter_res_dim[i + 1][1]);
				for (m = 0; m < inter_res_dim[i + 1][1]; m++)
				{
					inter_res[i + 1][j][m] = (float*)malloc(sizeof(float)*inter_res_dim[i + 1][0]);
					for (n = 0; n < inter_res_dim[i + 1][0]; n++)
					{
						inter_res[i + 1][j][m][n] = 0.0;
					}
				}
			}
		}
		else if (layer_type[i] == 1)
		{
			//fc
			inter_res[i + 1] = (float***)malloc(sizeof(float**) * 1);
			inter_res[i + 1][0] = (float**)malloc(sizeof(float*) * 1);
			inter_res[i + 1][0][0] = (float*)malloc(sizeof(float)*inter_res_dim[i+1][0]);
			for (j = 0; j < inter_res_dim[i + 1][0]; j++)
			{
				inter_res[i + 1][0][0][j] = 0.0;
			}
		}
		else if (layer_type[i] == 2)
		{
			//pool
			inter_res[i + 1] = (float***)malloc(sizeof(float**) * feature_num[i]);
			for (j = 0; j < feature_num[i]; j++)
			{
				inter_res[i + 1][j] = (float**)malloc(sizeof(float*)*inter_res_dim[i + 1][1]);
				for (k = 0; k < inter_res_dim[i + 1][1]; k++)
				{
					inter_res[i + 1][j][k] = (float*)malloc(sizeof(float)*inter_res_dim[i + 1][0]);
					for (m = 0; m < inter_res_dim[i + 1][0]; m++)
					{
						inter_res[i + 1][j][k][m] = 0.0;
					}
				}
			}
		}
		else
		{
			printf("unrecognized layer, can't allocate mem\n");
		}
	}

}


int clcnn::retrieve_result(float* container, int lang)
{
	int i;
	if(lang > inter_res_dim[layer_num][0])
	{
		printf("retrieve lenght excede the output dimension\n");
		for(i=0;i<inter_res_dim[layer_num][0];i++)
		{
			container[i] = inter_res[layer_num][0][0][i];
		}
	}
	else
	{
		for(i=0;i<lang;i++)
		{
			container[i] = inter_res[layer_num][0][0][i];
		}
	}
	return 0;
}

int clcnn::predict()
{
	float tmp_base = 0.0;
	int predict;
	int i;
	for (i = 0; i < inter_res_dim[layer_num][0]; i++)
	{
		if (inter_res[layer_num][0][0][i] > tmp_base)
		{
			predict = i;
			tmp_base = inter_res[layer_num][0][0][i];
		}
	}
	predict = (predict + 1)%10;
	printf("\nthe prediction is: %d\n", predict);
	return predict;
}

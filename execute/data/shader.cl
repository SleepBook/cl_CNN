float sigmoid(float input)
{
	return 1.0/(1+ exp(0.0 - input));
}

__kernel void convp_sig_avg_full(
	const int conv_w,
	const int conv_h,
	const int out_w,
	const int out_h,
	const int ker_w, 
	const int ker_h, 
	const int pool_w, 
	const int pool_h, 
	const int stride,
	__global const float* net,
	__global const float* bias,
	__global const int* flt_offset, 
	__global const int* flt_sz_lk,
	__global float* input, 
	__global float* output,
	__local float* flt_buf, 
	__local float* input_buf, 
	__local float* conv_buf,
	__local float* pool_buf)
{
	int global_idx = get_global_id(0);
	int global_idy = get_global_id(1);

	int total_group = get_num_groups(0);
	//printf("the group size is %d %d\n", get_num_groups(0), get_num_groups());
	int group_id = get_group_id(0);
	int total_local_group = get_local_size(0)/conv_w;
	int local_group_id = get_local_id(0)/conv_w;
	int local_idx = get_local_id(0)%conv_w;
	int local_idy = get_local_id(1);	

	int net_offset = flt_offset[group_id] + ker_h*ker_w*local_group_id;
	int input_offset = flt_sz_lk[(local_group_id+1)*total_group + group_id];

	int i,j,k;
	float res;

	res = 0.0;
	for(i=0;i<ker_h;i++)
	{
		for(j=0;j<ker_w;j++)
		{
			res += net[net_offset+i*ker_w + j] * input[input_offset + (local_idx+i)*(conv_w+ker_w-1) + local_idy + j];
		}
	}	
	conv_buf[local_group_id*conv_w*conv_h + local_idx*conv_w + local_idy] = res;	
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_group_id == 0)
	{
		for(i=0;i<total_local_group-1;i++)
		{
			conv_buf[local_idx*conv_w + local_idy] += conv_buf[(i+1)*conv_w*conv_h + local_idx*conv_w + local_idy];
		}
		conv_buf[local_idx*conv_w + local_idy] = sigmoid(conv_buf[local_idx*conv_w + local_idy] + bias[group_id])/(pool_h*pool_w);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	
	if(local_group_id==0 && (local_idy%pool_w) == 0)
	{
		for(i=0;i<(pool_w-1);i++)
		{
			conv_buf[local_idx*conv_w + local_idy] += conv_buf[local_idx*conv_w + local_idy + i + 1];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(local_group_id==0 && local_idx%pool_h == 0 && local_idy%pool_w == 0)
	{
		for(i=0;i<(pool_h-1);i++)
		{
			conv_buf[local_idx*conv_w + local_idy] += conv_buf[(local_idx+i+1)*conv_w + local_idy];
		}
		output[group_id*out_w*out_h + (local_idx/pool_h)*out_w + local_idy/pool_w] = conv_buf[local_idx*conv_w + local_idy];
	}
}
__kernel void convp_sig_avg_mc(
	const int conv_w,
	const int conv_h,
	const int out_w,
	const int out_h,
	const int ker_w,
	const int ker_h,
	const int pool_w,
	const int pool_h,
	const int stride,
	const int many, //how many channel each thread need compute
	__global const float* net,
	__global const float* bias,
	__global const int* flt_offset,
	__global const int* flt_sz_lk,
	__global const float* input,
	__global  float* output,
	__local float* buf 
	)
{
	int group_id = get_group_id(0);
	int group_size = get_num_groups(0);

	int local_idx = get_local_id(0);
	int batch_h = get_local_size(0)/conv_h;
	int batch_num = local_idx/conv_h;
	int batch_pos = local_idx%conv_h;
	int local_idy = get_local_id(1);

	int net_offset = flt_offset[group_id];
	int total_fwd_lk = flt_sz_lk[group_id];

	int m,i,j,k;
	float res = 0.0;
	//for testing purpose
	if (get_global_id(0) == 0 && get_global_id(1) == 0)
	{
		//printf("%d %d %d %d %d %d %d %d\n",conv_w,conv_h,out_w,out_h,ker_w,ker_h,pool_w,pool_h);
		//printf("%d %d\n",get_num_groups(0),get_num_groups(1));
		//printf("%d\n", many);
		//printf("%d\n", flt_sz_lk[0]);
		//printf("%d %d %d %d %d %d %d %d\n", conv_w, conv_h, out_w, out_h, ker_w, ker_h, pool_w, pool_h);
		//printf("%d %d\n", batch_h, batch_num);
		/*
		for(i=0;i<6;i++)
		{
			for(j=0;j<16;j++)
			{
				printf("%d ",flt_sz_lk[i*16+j]);
			}
			printf("\n");
		}
		*/
	}

	for(m=0;m<many;m++)
	{
		if(m * batch_h + batch_num < total_fwd_lk)
		{
			for(i=0;i<ker_h;i++)
			{
				for(j=0;j<ker_w;j++)
				{
					res += net[net_offset+(m * batch_h + batch_num)*ker_w*ker_h	+ i* ker_w +j] * \
						input[(flt_sz_lk[(m * batch_h + batch_num + 1) * group_size + group_id]) +\
						(batch_pos+i)*(conv_w + ker_w -1) + local_idy +j];
					//if (get_global_id(0) == 0 && get_global_id(1) == 0)
					//	printf("%d %d\n",net_offset+(m * batch_h + batch_num)*ker_w*ker_h,(flt_sz_lk[(m * batch_h + batch_num + 1) * total_fwd_lk + group_id]));
				}
			}
		}
	}
	buf[batch_num*conv_w*conv_h + batch_pos*conv_w + local_idy] = res;

	barrier(CLK_LOCAL_MEM_FENCE);
	if(batch_num == 0)
	{
		for(i=0;i<batch_h-1;i++)
		{
			buf[batch_pos*conv_w + local_idy] +=  buf[(i+1)*conv_w*conv_h+batch_pos*conv_w + local_idy];
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(batch_num == 0)
	{
		buf[batch_pos*conv_w + local_idy] = sigmoid(buf[batch_pos*conv_w + local_idy]+bias[group_id])/(pool_w*pool_h);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(batch_num == 0)
	{
		if(batch_pos%pool_h == 0 && local_idy%pool_w == 0)
		{
			for(i=0;i<pool_h-1;i++)
			{
				for(j=0;j<pool_w-1;j++)
				{
					buf[batch_pos*conv_w + local_idy] += buf[(batch_pos+i+1)*conv_w + local_idy + j + 1];
				}
			}
			output[group_id*out_w*out_h + (batch_pos/pool_h)*out_w +\
						local_idy/pool_w] = buf[batch_pos*conv_w + local_idy];
		}
	}
}

__kernel void convp_sig_avg_pc(
	const int conv_w,
	const int conv_h,
	const int out_w,
	const int out_h,
	const int ker_w,
	const int ker_h,
	const int pool_w,
	const int pool_h,
	const int stride,
	__global const float* net,
	__global const float* bias,
	__global const int* flt_offset,
	__global const int* flt_sz_lk,
	__global const float* input,
	__global float* output,
	__local float* buf)
{
	int local_idh = get_local_id(0);
	int local_idw = get_local_id(1);
	int group_id = get_group_id(0);
	int group_size = get_num_groups(0);

	int net_offset = flt_offset[group_id];
	int total_fwd_lk = flt_sz_lk[group_id];

	int i,j,m,n,k;
	int cnv_h;
	int cnv_w;
	float res;
	buf[local_idh*out_w + local_idw] = 0.0;

	for(m=0;m<pool_h;m++)
	{
		for(n=0;n<pool_w;n++)
		{
			res = 0.0;
			cnv_h = local_idh*pool_h + m;
			cnv_w = local_idw*pool_w + n;
			for(i=0;i<ker_h;i++)
			{
				for(j=0;j<ker_w;j++)
				{
					for(k=0;k<total_fwd_lk;k++)
					{
						res += net[net_offset + ker_w*ker_h*k + i*ker_w + j] * \
								input[flt_sz_lk[(k+1)*group_size + group_id]+\
								(cnv_h+i)*(conv_w + ker_w -1) + cnv_w + j];
					}
				}
			}
			buf[local_idh*out_w + local_idw] += sigmoid(res + bias[group_id]);
		}
	}
	output[group_id*out_w*out_h + local_idh*out_w + local_idw] = buf[local_idh*out_w + local_idw]/(pool_w*pool_h);
	
}

	

//for FULL layer there are two kind of kernels, fcp and fc
//fcp stands for port, it's used when its front layer is not a FULL
__kernel void fcp_sig(
	const int output_dim,
	const int front_feature_num,
	const int front_feature_w,
	const int front_feature_h,
	__global const float *net,
	__global const float *bias,
	__global float *input,
	__global float *output,
	__local float* buffer
	)
	//each group number equal outputnum
	//the number of thread in each group equals the width per feature of last layer
	//the reason is to comply the net-data's flat mode
{
	int local_id = get_local_id(0);
	int out_n = get_group_id(0);
	int i, j;
	float tmp_res = 0.0;
	float ipt,flt;
	
	//printf("%d %d %d %d %d %d\n", output_dim, front_feature_num, front_feature_w, front_feature_h, out_n, local_id);
	for(i=0;i<front_feature_num;i++)
	{
		for(j=0;j<front_feature_h;j++)
		{
			ipt = input[i*front_feature_h*front_feature_w + j*front_feature_w + local_id];
			//flt = net[out_n*output_dim + i*front_feature_h*front_feature_w + local_id*front_feature_h + j];
			flt = net[(i*front_feature_h*front_feature_w + local_id * front_feature_w + j)*output_dim + out_n];
			tmp_res += ipt * flt;
			//printf("%f  %f  %f\n", ipt, flt, tmp_res);
		}		
	}	
	buffer[local_id] = tmp_res;
	barrier(CLK_LOCAL_MEM_FENCE);
	//printf("the loca tmp res is %f \n",tmp_res);
	
	if(local_id == 0)
	{
		for(i=0;i<front_feature_w-1;i++)
		{
			buffer[0] += buffer[i+1];
		}
		//printf("%f  %f  \n",buffer[0],bias[out_n]);
		output[out_n] = sigmoid(buffer[0]+bias[out_n]);
	}
	
}


__kernel void fc_sig(
	const int input_dim,	
	__global const float *net,
	__global const float *bias,
	__global float *input,
	__global float *output
	)
//one thread sum one output
{
	int id = get_global_id(0);
	int output_dim = get_global_size(0);
	//printf("global %d id %d\n",output_dim,id);
	float res = 0.0;
	int i;
	for(i=0;i<input_dim;i++)
	{
		res += input[i] * net[i*output_dim + id];
	}
	res = sigmoid(res + bias[id]);
	output[id] = res;
}


/*
	int graph_size = (conv_w+ker_w-1)*(conv_h+ker_h-1);
	int ipt_offset = flt_sz_lk[total_group*(local_group+1)+group_id];
	input_buf[graph_size*local_group + local_idx*(conv_w+ker_w-1) + local_idy] = input[ipt_offset+local_idx*(conv_w+ker_w-1) + local_idy];
	if(local_idy == conv_w-1)
	{
		for(i=0;i<ker_w-1;i++)
		{
			input_buf[graph_size*local_group + local_idx*(conv_w + ker_w -1) + local_idy + i] = input[ipt_offset + local_idx*(conv_w + ker_w -1) + local_idy + i];
		}
	}
	if(local_idx == conv_h -1)
	{
		for(i=0;i<ker_h-1;i++)
		{
			input_buf[graph_size*local_group + ]
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	if(group_id == 0 &&local_group==0 && local_idx == 0 && local_idy ==0)
	{
		for(i=0;i<get_local_size(0)/conv_w;i++)
		{
			for(j=0;j<conv_h)
		}
	}
*/
#include <clcnn.h>

int main()
{
	int i,j;
	float* res;
	res = (float*)malloc(sizeof(float)*10);
	int pred;
	float kernel_time,CPUtime;
	clcnn lenet("../data/test.cnet");
	for(i=0;i<1;i++)
	{
		printf(" %d cnn call\n",i);
		lenet.getInput();
		lenet.execute_device();
		//lenet.execute_cpu();
		pred = lenet.predict();
		lenet.retrieve_result(res, 10);		
		kernel_time = lenet.getKernelTime();
		//CPUtime = lenet.getCPUTime();
		
		for (j = 0; j < 10; j++)
		{
			printf("%f ", res[j]);
		}
		printf("\n");
		
	}
	getchar();
	return 0;
}

#include <clcnn.h>

int main()
{
	int i,j;
	float* res;
	res = (float*)malloc(sizeof(float)*10);
	int pred;
	float exe_time;
	clcnn lenet("../data/test.cnet");
	for(i=0;i<1;i++)
	{
		lenet.execute_device("../data/test.cdat");
	//	lenet.execute_cpu("../data/test.cdat");
		pred = lenet.predict();
		lenet.retrieve_result(res, 10);		
		exe_time = lenet.get_execution_time();
		
		for (j = 0; j < 10; j++)
		{
			printf("%f ", res[j]);
		}
		printf("\n");
		
	}
	return 0;
}

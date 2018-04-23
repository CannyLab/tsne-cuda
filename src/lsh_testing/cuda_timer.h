#ifndef __CUDA_TIMER_H_
#define __CUDA_TIMER_H_

#include <cutil.h>

void startTimer(unsigned *timer)
{
	CUT_SAFE_CALL(cutCreateTimer(timer));
	CUT_SAFE_CALL(cutStartTimer(*timer));
}


double endTimer(char *info, unsigned *timer)
{
	cudaThreadSynchronize();
	CUT_SAFE_CALL(cutStopTimer(*timer));
	double result = cutGetTimerValue(*timer);
	printf("%s costs, %3f ms\n", info, result);
	CUT_SAFE_CALL(cutDeleteTimer(*timer));
	return result;
}


#endif
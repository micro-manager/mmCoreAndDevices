#include "comdef.h"

/*int max(int a, int b) {
   return a>b ? a:b;
}

int min(int a, int b) {
   return a<b ? a:b;
}


unsigned long GetTickCount()
{
#ifdef _MAC
   struct timeval  now;
    gettimeofday(&now, NULL);
    unsigned long ul_ms = now.tv_usec/1000 + now.tv_sec*1000;
    return ul_ms;
#else   
   struct timespec ts;
   clock_gettime(CLOCK_MONOTONIC,&ts);
   return (ts.tv_sec*1000 + ts.tv_nsec/(1000*1000));
#endif
}*/


#ifndef COMMO
#define COMMO

#ifdef _LIN

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdarg.h>

#include <unistd.h>//usleep


int max(int,int);
int min(int,int);

unsigned long GetTickCount();

#define Sleep(a) usleep((a)*1000)
#endif
#endif


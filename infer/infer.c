#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "../netvisa.h"

/* we will store the events and detections in chunks */
#define INFER_TIME_CHUNK   300

/* we will infer events in window sizes */
#define INFER_WINDOW_SIZE 1800
/* the window will move forward in steps */
#define INFER_WINDOW_STEP 300

void infer(NetModel_t * p_netmodel, int numsamples)
{
  
}

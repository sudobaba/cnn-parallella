#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include "common.h"
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include <e-hal.h>

int main(int argc, char *argv[]){

	unsigned row, col, coreid;
	e_platform_t platform;
	e_epiphany_t dev;
	e_mem_t   mbuf;
	int i, j, k, l, pos, CH;
	unsigned all_done, done[CORES], clk_cyc;

	double cpu_time_used=0;
	struct timespec start, end;

	// Denominator for softmax
	float den;

	// Weights
	float c1w[32*1*5*5], c1b[32], c2w[32*32*5*5], c2b[32], c3w[64*32*3*3], c3b[64];
	float d1w[256*3136], d1b[256], d2w[10*256], d2b[10];

	// Image
	float im[28*28];
	float flat_im[3136], last_im[256], output[10];

/******************************************************************************************************/

	FILE * fp;
	char ch[50], comma[] = ",", neg[] = "-";
	const char s[2] = "&";
	char *token;

	// IMAGE
	fp = fopen(img, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
	while(i<50) {
		ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &im[pos]);
			if(ch[0] == neg[0]) {im[pos] = -1*im[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	for (i=0;i<28;i++) {
		for (j=0;j<28;j++) {
			printf("%.1f ", im[28*i + j]);
		} printf("\n");
	}

	//CONV1_BIAS
	fp = fopen(conv1_b, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c1b[pos]);
			if(ch[0] == neg[0]) {c1b[pos] = -1*c1b[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);
	// CONV1_WEIGHTS
	fp = fopen(conv1_w, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c1w[pos]);
			if(ch[0] == neg[0]) {c1w[pos] = -1*c1w[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	//CONV2_BIAS
	fp = fopen(conv2_b, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c2b[pos]);
			if(ch[0] == neg[0]) {c2b[pos] = -1*c2b[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);
	// CONV2_WEIGHTS
	fp = fopen(conv2_w, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c2w[pos]);
			if(ch[0] == neg[0]) {c2w[pos] = -1*c2w[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	//CONV3_BIAS
	fp = fopen(conv3_b, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c3b[pos]);
			if(ch[0] == neg[0]) {c3b[pos] = -1*c3b[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);
	// CONV3_WEIGHTS
	fp = fopen(conv3_w, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &c3w[pos]);
			if(ch[0] == neg[0]) {c3w[pos] = -1*c3w[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	//DENSE1_BIAS
	fp = fopen(dense1_b, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &d1b[pos]);
			if(ch[0] == neg[0]) {d1b[pos] = -1*d1b[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);
	// DENSE1_WEIGHTS
	fp = fopen(dense1_w, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &d1w[pos]);
			if(ch[0] == neg[0]) {d1w[pos] = -1*d1w[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	//DENSE2_BIAS
	fp = fopen(dense2_b, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &d2b[pos]);
			if(ch[0] == neg[0]) {d2b[pos] = -1*d2b[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);
	// DENSE2_WEIGHTS
	fp = fopen(dense2_w, "r");
  	if (fp == NULL) {
    		printf("failed to open file\n");
   		return 0;
  	}

	i=0;
	pos = 0;
        while(i<50) {
                ch[i] = getc(fp);
		if(ch[i] == comma[0]) {
			for (j=29;j>17;j--) {ch[j] = '0';}
			token = strtok(ch, s);
			sscanf(token, "%f", &d2w[pos]);
			if(ch[0] == neg[0]) {d2w[pos] = -1*d2w[pos];}
			i=-1;
			pos++;
		}
		i++;
	}
	fclose(fp);

	printf("Params and img read\n");

/******************************************************************************************************/

	//Initalize Epiphany device
  	e_init(NULL);                      
  	e_reset_system();                                      //reset Epiphany
  	e_get_platform_info(&platform);

  	e_open(&dev, 0, 0, platform.rows, platform.cols); //open all cores
	e_reset_group(&dev);

  	//Load program to cores
  	e_load_group("e_task.elf", &dev, 0, 0, platform.rows, platform.cols, E_FALSE);

  	for (i=0; i<platform.rows; i++){
		for (j=0; j<platform.cols;j++){
  			e_write(&dev, i, j, 0x2758, &im, sizeof(im));
			e_write(&dev, i, j, 0x5418, &c1w[(i*4+j)*2*5*5], 2*5*5*sizeof(float));
			e_write(&dev, i, j, 0x54E0, &c1b[(i*4+j)*2], 2*sizeof(float));
			e_write(&dev, i, j, 0x54E8, &c2w[(i*4+j)*2*32*5*5], 2*32*5*5*sizeof(float));
			e_write(&dev, i, j, 0x6DE8, &c2b[(i*4+j)*2], 2*sizeof(float));
			e_write(&dev, i, j, 0x6DF0, &c3w[(i*4+j)*4*32*3*3], 4*32*3*3*sizeof(float));
			e_write(&dev, i, j, 0x7FF0, &c3b[(i*4+j)*2], 4*sizeof(float));
		}
  	}

	e_start_group(&dev);

	while(1){    
    		all_done=0;
    		for (i=0; i<platform.rows; i++){
      			for (j=0; j<platform.cols;j++){
					e_read(&dev, i, j, 0x2000, &done[i*platform.cols+j], sizeof(int));
					all_done+=done[i*platform.cols+j];
      			}
    		}
    		if(all_done==CORES){
      			break;
    		}
  	}

	for (i=0;i<platform.rows;i++) {
		for (j=0;j<platform.cols;j++) {
			e_read(&dev, i, j, 0x2758, &flat_im[(i*4+j)*7*7*4], 7*7*4*sizeof(float));
		}
	}

	clock_gettime(CLOCK_MONOTONIC, &start);

	for (i=0;i<256;i++) {
		last_im[i] = 0;
		for (j=0;j<3136;j++) {
			last_im[i] += d1w[3136*i+j]*flat_im[j];
		}
		last_im[i] = (last_im[i]+d1b[i]) > 0 ? last_im[i]+d1b[i] : 0;
	}

	// DROP OUT

	for (i=0;i<10;i++) {
		output[i] = 0;
		for (j=0;j<256;j++) {
			output[i] += d2w[256*i+j]*last_im[j];
		}
		output[i] = (output[i]+d2b[i]) > 0 ? output[i]+d2b[i] : 0;
		output[i] = 1/(1 + pow(ee,-output[i]));
	}

	//den = pow(ee,output[0])+pow(ee,output[1])+pow(ee,output[2])+pow(ee,output[3])+pow(ee,output[4])+
	//	pow(ee,output[5])+pow(ee,output[6])+pow(ee,output[7])+pow(ee,output[8])+pow(ee,output[9]);

	//for (i=0;i<10;i++) {
	//	output[i] = pow(ee,output[i])/den;
	//}

	clock_gettime(CLOCK_MONOTONIC, &end);
        cpu_time_used = (end.tv_sec - start.tv_sec)*1e9;
        cpu_time_used += (cpu_time_used + (end.tv_nsec - start.tv_nsec))*1e-9;

	for (i=0;i<10;i++){ 
		printf("%d:%f, ", i, output[i]);
	} printf("\n");

        //printf("Time taken in sec is %.10f\n", cpu_time_used);

/*
	float tmp_im[4*7*7];

	k=0; l=0;

	for (k=0;k<4;k++) {
		for (l=0;l<4;l++) {
			printf("CORE %d, %d :\n", k, l);
			e_read(&dev, k, l, 0x2758, &tmp_im, 4*7*7*sizeof(float));	// im
			//e_read(&dev, 3, 3, 0x3FD8, &tmp_im, 2*9*9*sizeof(float));	// tmp_im
			//e_read(&dev, 0, 0, 0x49F8, &tmp_im, sizeof(float));		// dma_im
	
			for (CH=0;CH<4;CH++) {
				printf("Channel %d :\n", CH);
				for (i=0;i<7;i++) {
					for (j=0;j<7;j++) {
						printf("%.4f ", tmp_im[49*CH+7*i+j]);
					} printf("\n");
				} printf("\n");
			}
		}
	}
*/

	int timings[16];
	float total_time;

	for (i=0;i<4;i++) {
		for (j=0;j<4;j++) {
			e_read(&dev, i, j, 0x2004, &timings[i*4+j], sizeof(int));
			total_time += (float)timings[i*4+j];
		}
	}

	total_time = ((total_time/16)/600000000.0) + cpu_time_used;

	printf("The approx time taken (in sec) is %f\n", total_time);
	// Comes out to be 21.071 msec

	//Close down Epiphany device
  	e_close(&dev);

  	e_finalize();

	return EXIT_SUCCESS;
}

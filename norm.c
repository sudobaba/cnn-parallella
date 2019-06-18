#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "common.h"
#include <math.h>

#include <time.h>
#include <sys/time.h>

#include <e-hal.h>

int main(int argc, char *argv[]){

	int i, j, k, l, f, pos, CH;
	unsigned all_done, done[CORES], clk_cyc;
	float c1, c2;

	double cpu_time_used=0;
	struct timespec start,end;

	// Denominator for softmax
	float den;

	// Weights
	float c1w[32*1*5*5], c1b[32], c2w[32*32*5*5], c2b[32], c3w[64*32*3*3], c3b[64];
	float d1w[256*3136], d1b[256], d2w[10*256], d2b[10];

	// Image
	float im[28*28], im1[32*28*28], im2[32*14*14], im3[64*7*7];
	float last_im[256], output[10];
	float tmp_im1[32*32], tmp_im2[32*18*18], tmp_im3[32*9*9];

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

	clock_gettime(CLOCK_MONOTONIC, &start);

	// ZERO PADDING by 2 - tmp_im is 32*32
	for (i=0;i<1024;i++) { tmp_im1[i] = 0; }
	for (i=0;i<28;i++) {
		for (j=0;j<28;j++) {
			tmp_im1[66 + 32*i + j] = im[28*i + j];
		}
	}

	// CONV 1 - im1 is 32*28*28
	for (CH=0;CH<32;CH++) {
		for (i=0;i<28;i++) {
			for (j=0;j<28;j++) {
				im1[784*CH + 28*i + j] = 0;
				for (k=0;k<5;k++) {
					for (l=0;l>5;l++) {
						im1[784*CH + 28*i + j] += tmp_im1[32*i+j + 32*k+l]*c1w[25*CH + 5*k+l];
					}
				}
				c1 = im1[784*CH + 28*i + j] + c1b[CH];
				im1[784*CH + 28*i + j] = (c1 > 0) ? c1 : 0;
			}
		}
	}

	// MAX POOL + ZERO PADDING by 2 - tmp_im is 32*18*18
	for (CH=0;CH<32;CH++) {
		for (i=0;i<36;i++) {
			tmp_im2[324*CH + i] = 0;
			tmp_im2[324*(CH+1) - 1 - i] = 0;	
		}
		for (i=0;i<14;i++) {
			tmp_im2[324*CH + 18*i + 36] = 0;
			tmp_im2[324*CH + 18*i + 37] = 0;
			tmp_im2[324*CH + 18*i + 52] = 0;
			tmp_im2[324*CH + 18*i + 53] = 0;
			for (j=0;j<14;j++) {
				c1 = (im1[784*CH + 28*(2*i) + (2*j)] > im1[784*CH + 28*(2*i) + (2*j+1)]) ? im1[784*CH + 28*(2*i) + (2*j)] : im1[784*CH + 28*(2*i) + (2*j+1)];
                                c2 = (im1[784*CH + 28*(2*i+1) + (2*j)] > im1[784*CH + 28*(2*i+1) + (2*j+1)]) ? im1[784*CH + 28*(2*i+1) + (2*j)] : im1[784*CH + 28*(2*i+1) + (2*j+1)];
                                tmp_im2[324*CH + 38 + 18*i+j] = (c1 > c2) ? c1 : c2;
			}
		}
	}

	// CONV 2 - im2 is 32*14*14
	for (CH=0;CH<32;CH++) {
		for (i=0;i<14;i++) {
			for (j=0;j<14;j++) {
				im2[196*CH + 14*i + j] = 0;
				for (f=0;f<32;f++) {
					for (k=0;k<5;k++) {
						for (l=0;l<5;l++) {
							im2[196*CH + 14*i + j] += tmp_im2[324*f + 18*i+j + 18*k+l]*c2w[32*25*CH + 25*f + 5*k+l];
						}
					}
				}
				c1 = im2[196*CH + 14*i + j] + c2b[CH];
				im2[196*CH + 14*i + j] = (c1 > 0) ? c1 : 0;
			}
		}
	}

	// MAX POOL + ZERO PADDING by 2 - tmp_im is 32*9*9
	for (CH=0;CH<32;CH++) {
		for (i=0;i<9;i++) {
			tmp_im3[81*CH + i] = 0;
			tmp_im3[81*(CH+1) - 1 - i] = 0;
		}
		for (i=0;i<7;i++) {
			tmp_im3[81*CH + 9*i + 9] = 0;
			tmp_im3[81*CH + 9*i + 17] = 0;
			for (j=0;j<7;j++) {
				c1 = (im2[196*CH + 14*(2*i) + (2*j)] > im2[196*CH + 14*(2*i) + (2*j+1)]) ? im2[196*CH + 14*(2*i) + (2*j)] : im2[196*CH + 14*(2*i) + (2*j+1)];
                                c2 = (im2[196*CH + 14*(2*i+1) + (2*j)] > im2[196*CH + 14*(2*i+1) + (2*j+1)]) ? im2[196*CH + 14*(2*i+1) + (2*j)] : im2[196*CH + 14*(2*i+1) + (2*j+1)];
                                tmp_im3[81*CH + 10 + 9*i+j] = (c1 > c2) ? c1 : c2;
			}
		}
	}

	// CONV 3 - im3 is 64*7*7
	for (CH=0;CH<64;CH++) {
		for (i=0;i<7;i++) {
			for (j=0;j<7;j++) {
				im3[49*CH + 7*i + j] = 0;
				for (f=0;f<32;f++) {
					for (k=0;k<3;k++) {
						for (l=0;l<3;l++) {
							im3[49*CH + 7*i + j] += tmp_im3[81*f + 9*i+j + 9*k+l]*c3w[32*9*CH + 9*f + 3*k+l];
						}
					}
				}
				c1 = im3[49*CH + 7*i + j] + c3b[CH];
				im3[49*CH + 7*i + j] = (c1 > 0) ? c1 : 0;
			}
		}
	}

	for (i=0;i<256;i++) {
                last_im[i] = 0;
                for (j=0;j<3136;j++) {
                        last_im[i] += d1w[3136*i+j]*im3[j];
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
        }

	den = pow(ee,output[0]) + pow(ee,output[1]) + pow(ee,output[2]) + pow(ee,output[3]) + pow(ee,output[4]) +
		pow(ee,output[5]) + pow(ee,output[6]) + pow(ee,output[7]) + pow(ee,output[8]) + pow(ee,output[9]);

	for (i=0;i<10;i++) {
		output[i] = pow(ee,output[i])/den;
	}

	clock_gettime(CLOCK_MONOTONIC, &end);
	cpu_time_used = (end.tv_sec - start.tv_sec)*1e9;
	cpu_time_used += (cpu_time_used + (end.tv_nsec - start.tv_nsec))*1e-9;

	printf("Time taken in sec is %.10f\n", cpu_time_used);

	for (i=0;i<10;i++) {
		printf("%d:%f, ", i, output[i]);
	} printf("\n");

	return EXIT_SUCCESS;
}

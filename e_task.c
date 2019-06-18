#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "e-lib.h"
#include "common.h"

volatile e_barrier_t  barriers[16]; // barriers array
volatile e_barrier_t *tgt_bars[16]; // barriers array

int main(void)
{
	e_coreid_t coreid;

	float c1,c2;
	unsigned my_row, my_col, nc, c;
	unsigned *done, *time;
	float *im, *tmp_im, *dma_im, *neighbour;
	float *c1w, *c1b, *c2w, *c2b, *c3w, *c3b;
	
	time = (unsigned *) 0x2004;
	done = (unsigned *) 0x2000;

	im = (float *) 0x2758;
	tmp_im = (float *) 0x3FD8;
	dma_im = (float *) 0x49F8;
	c1w = (float *) 0x5418;
	c1b = (float *) 0x54E0;
	c2w = (float *) 0x54E8;
	c2b = (float *) 0x6DE8;
	c3w = (float *) 0x6DF0;
	c3b = (float *) 0x7FF0;

	unsigned int time_p;
        unsigned int time_c;
        unsigned int time_compare;
        unsigned int time_taken = 0;

/*
im = 28*28
tmp_im = 32*32
c1w = 5*5*1*2
c1b = 2
im = 28*28*2
tmp_im = 18*18*2
c2w = 5*5*32*2
c2b = 2
im = 14*14*2
tmp_im = 9*9*2
c3w = 3*3*32*4
c3b = 4
im = 7*7*4
*/

	coreid = e_get_coreid();
	e_coords_from_coreid(coreid, &my_row, &my_col);
	c = nc = my_row*4 + my_col;
	neighbour = (float *) e_get_global_address(((15+nc)%16)/4,((15+nc)%16)%4, dma_im);

	// Initialize the barriers
	e_barrier_init(barriers, tgt_bars);

	// Get time waste on functions
        e_ctimer_set(E_CTIMER_0, E_CTIMER_MAX) ;
        e_ctimer_start(E_CTIMER_0, E_CTIMER_CLK);
        time_p = e_ctimer_get(E_CTIMER_0);
        time_c = e_ctimer_get(E_CTIMER_0);
        e_ctimer_stop(E_CTIMER_0);
        time_compare = time_p - time_c ;

        e_ctimer_set(E_CTIMER_0, E_CTIMER_MAX) ;
        e_ctimer_start(E_CTIMER_0, E_CTIMER_CLK);
        time_p = e_ctimer_get(E_CTIMER_0);

	// ZERO PADDING 2
	for (int i=0;i<28;i++) {
		for (int j=0;j<28;j++) {
			tmp_im[66 + 32*i + j] = im[28*i + j];
		}
	}

	//CONV 1
	for (int i=0;i<28;i++) {
		for (int j=0;j<28;j++) {
			im[28*i+j] = 0;
			im[28*28+28*i+j] = 0;
			for (int k=0;k<5;k++) {
				for (int l=0;l<5;l++) {
					im[28*i+j] += tmp_im[32*i+j + 32*k+l]*c1w[5*k+l];
					im[28*28+28*i+j] += tmp_im[32*i+j + 32*k+l]*c1w[5*5+5*k+l]; 
				}
			}
			c1 = im[28*i+j] + c1b[0];
			c2 = im[28*28+28*i+j] + c1b[1];
			im[28*i+j] = (c1>0) ? c1 : 0;
			im[28*28+28*i+j] = (c2>0) ? c2 : 0;
		}
	}

	// MAX POOL + 2 ZERO PADDING
	for (int i=0;i<36;i++) {
		tmp_im[i] = 0;
		tmp_im[323 - i] = 0;
		tmp_im[324 + i] = 0;
		tmp_im[647 - i] = 0;
	}

	for (int ch=0;ch<2;ch++) {
		for (int i=0;i<14;i++) {
			tmp_im[324*ch + 18*i + 36] = 0;
			tmp_im[324*ch + 18*i + 37] = 0;
			tmp_im[324*ch + 18*i + 52] = 0;
			tmp_im[324*ch + 18*i + 53] = 0;
			for (int j=0;j<14;j++) {
				c1 = (im[784*ch + 28*(2*i) + (2*j)] > im[784*ch + 28*(2*i) + (2*j+1)]) ? im[784*ch + 28*(2*i) + (2*j)] : im[784*ch + 28*(2*i) + (2*j+1)];
				c2 = (im[784*ch + 28*(2*i+1) + (2*j)] > im[784*ch + 28*(2*i+1) + (2*j+1)]) ? im[784*ch + 28*(2*i+1) + (2*j)] : im[784*ch + 28*(2*i+1) + (2*j+1)];
				tmp_im[324*ch + 38+18*i+j] = (c1 > c2) ? c1 : c2;
			}
		}
	}
	// DROP OUT

	// Sync with all other cores
	e_barrier(barriers, tgt_bars);

	// CONV 2			- im = 14*14*2
	for (int i=0;i<392;i++) { im[i] = 0; }

	for (int iter=0;iter<16;iter++) {

		for (int i=0;i<14;i++) {
			for (int j=0;j<14;j++) {
				for (int k=0;k<5;k++) {
					for (int l=0;l<5;l++) {
						im[14*i+j] += tmp_im[18*i+j + 18*k+l]*c2w[(2*nc)*5*5 + 5*k+l] + tmp_im[324 + 18*i+j + 18*k+l]*c2w[(2*nc+1)*5*5 + 5*k+l];
						im[196+14*i+j] += tmp_im[18*i+j + 18*k+l]*c2w[5*5*32+(2*nc)*5*5 + 5*k+l] + tmp_im[324 + 18*i+j + 18*k+l]*c2w[5*5*32+(2*nc+1)*5*5 + 5*k+l];
					}
				}
			}
		}

		e_dma_copy(neighbour, tmp_im, 18*18*2*sizeof(float));
		for (int i=0;i<648;i++) { tmp_im[i] = dma_im[i]; }
		nc = (17+nc)%16;

	}

	for (int i=0;i<196;i++) {
		im[i] = ((im[i]+c2b[0]) > 0) ? im[i]+c2b[0] : 0;
		im[196+i] = ((im[196+i]+c2b[1]) > 0) ? im[196+i]+c2b[1] : 0;
	}

	// MAX POOL + 1 ZERO PADDING	- tmp_im = 9*9*2
	for (int i=0;i<9;i++) {
		tmp_im[i] = 0;
		tmp_im[80-i] = 0;
		tmp_im[81+i] = 0;
		tmp_im[161-i] = 0;
	}
	for (int ch=0;ch<2;ch++) {
		for (int i=0;i<7;i++) {
			tmp_im[81*ch + 9 + 9*i] = 0;
			tmp_im[81*ch + 17 + 9*i] = 0;
			for (int j=0;j<7;j++) {
				c1 = (im[196*ch + 14*(2*i) + (2*j)] > im[196*ch + 14*(2*i) + (2*j+1)]) ? im[196*ch + 14*(2*i) + (2*j)] : im[196*ch + 14*(2*i) + (2*j+1)];
				c2 = (im[196*ch + 14*(2*i+1) + (2*j)] > im[196*ch + 14*(2*i+1) + (2*j+1)]) ? im[196*ch + 14*(2*i+1) + (2*j)] : im[196*ch + 14*(2*i+1) + (2*j+1)];
				tmp_im[81*ch + 10 + 9*i+j] = (c1 > c2) ? c1 : c2;
			}
		}
	}
	// DROP OUT

	// Sync with all other cores
	e_barrier(barriers, tgt_bars);

	// CONV 3			- im = 7*7*4
	for (int i=0;i<196;i++) { im[i] = 0; }

	for (int iter=0;iter<16;iter++) {
	
		for (int i=0;i<7;i++) {
			for (int j=0;j<7;j++) {
				for (int k=0;k<3;k++) {
					for (int l=0;l<3;l++) {
						im[7*i+j] += tmp_im[9*i+j + 9*k+l]*c3w[(2*nc)*3*3+3*k+l] + tmp_im[81 + 9*i+j + 9*k+l]*c3w[(2*nc+1)*3*3+3*k+l];
						im[49+7*i+j] += tmp_im[9*i+j + 9*k+l]*c3w[3*3*32+(2*nc)*3*3+3*k+l] + tmp_im[81 + 9*i+j + 9*k+l]*c3w[3*3*32+(2*nc+1)*3*3+3*k+l];
						im[2*49+7*i+j] += tmp_im[9*i+j + 9*k+l]*c3w[2*3*3*32+(2*nc)*3*3+3*k+l] + tmp_im[81 + 9*i+j + 9*k+l]*c3w[2*3*3*32+(2*nc+1)*3*3+3*k+l];
						im[3*49+7*i+j] += tmp_im[9*i+j + 9*k+l]*c3w[3*3*3*32+(2*nc)*3*3+3*k+l] + tmp_im[81 + 9*i+j + 9*k+l]*c3w[3*3*3*32+(2*nc+1)*3*3+3*k+l];
					}
				}
			}
		} 
		
		e_dma_copy(neighbour, tmp_im, 9*9*2*sizeof(float));
		for (int i=0;i<162;i++) { tmp_im[i] = dma_im[i]; }
		nc = (17+nc)%16;
	}
	for (int i=0;i<49;i++) {
		im[i] = ((im[i]+c3b[0]) > 0) ? im[i]+c3b[0] : 0;
		im[49+i] = ((im[49+i]+c3b[1]) > 0) ? im[49+i]+c3b[1] : 0;
		im[98+i] = ((im[98+i]+c3b[2]) > 0) ? im[98+i]+c3b[2] : 0;
		im[147+i] = ((im[147+i]+c3b[3]) > 0) ? im[147+i]+c3b[3] : 0;
	}
	// DROP OUT

	e_barrier(barriers, tgt_bars);

	time_c = e_ctimer_get(E_CTIMER_0);
	e_ctimer_stop(E_CTIMER_0);

	(*(time)) = time_p - time_c - time_compare;

	(*(done)) = 0x00000001;

	return 0;
}

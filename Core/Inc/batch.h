#include "datatypes.h"
#include <stdint.h>
#include <math.h>
#include "utils.h"
#include "arm_math.h"
void arm_maxpool_q15_HWC_ref(q15_t* Im_in, const uint16_t dim_im_in, const uint16_t ch_im_in, 
    const uint16_t dim_kernel, const uint16_t padding,const uint16_t stride, 
    const uint16_t dim_im_out, q15_t* Im_out);
void batch_norm_first_layer(q15_t* __restrict pAct, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, 
    pckDtype* __restrict pOut, pckDtype* __restrict thresh, pckDtype* sign, 
    pckDtype* __restrict offset, uint8_t out_bit);
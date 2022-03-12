#include "batch.h"
void arm_maxpool_q15_HWC_ref(q15_t* Im_in,
    const uint16_t dim_im_in,
    const uint16_t ch_im_in,
    const uint16_t dim_kernel,
    const uint16_t padding,
    const uint16_t stride, const uint16_t dim_im_out, q15_t* Im_out)
{
    int16_t   i_ch_in, i_x, i_y;
    int16_t   k_x, k_y;

    for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
    {
        for (i_y = 0; i_y < dim_im_out; i_y++)
        {
            for (i_x = 0; i_x < dim_im_out; i_x++)
            {
                int16_t      max = -32768; // i.e., -1 * (2^15+1) so that it is smaller than smallest q15_t
                for (k_y = i_y * stride - padding; k_y < i_y * stride - padding + dim_kernel; k_y++)
                {
                    for (k_x = i_x * stride - padding; k_x < i_x * stride - padding + dim_kernel; k_x++)
                    {
                        if (k_y >= 0 && k_x >= 0 && k_y < dim_im_in && k_x < dim_im_in)
                        {
                            if ((int16_t)Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)] > max)
                            {
                                max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in)];
                            }
                        }
                    }
                }
                Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out)] = max;
            }
        }
    }
}
void batch_norm_first_layer(q15_t* __restrict pAct, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, 
    pckDtype* __restrict pOut, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t out_bit) {
    uint16_t  yCoeff = wdth * dpth;
    uint8_t  xCoeff = dpth;
    pckDtype pckTemp[out_bit];
    for (int y = 0; y != hght; y++) {
        uint16_t y_offset = yCoeff * y;
        for (int x = 0; x != wdth; x++) {
            uint16_t x_offset = xCoeff * x;
            pckDtype* signs = sign;
            pckDtype* threshLoc = thresh;
            pckDtype* offsets = offset;
            for (int k = 0; k != dpth / pckWdt; k++) {
                memset(pckTemp, 0, sizeof(pckTemp));
                for (int ks = 0; ks != pckWdt; ks++) {
                    int out_temp = pAct[y_offset + x_offset + k * pckWdt + ks];/// pow(2, in_bit);
                    out_temp = out_temp << 16;
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        int temp = out_temp > *threshLoc;
                        // Shift 
                        pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                        out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                        //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                    }
                    threshLoc++;
                    offsets++;
                }
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pOut++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
    }
    
}
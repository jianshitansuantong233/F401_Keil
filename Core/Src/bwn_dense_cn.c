/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
* 
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

/*!
 * \file      bwn_dense_cn.c
 * \brief     Dense binary-weight convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "bwn_dense_cn.h"

void CnBwnWrap(int8_t* __restrict pAct, int8_t* __restrict pKrn, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt,
    const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool,
    pckDtype* __restrict pOut, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t out_bit) {
    if (thresh) {
        CnBnBwn(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pad, pool, pOut, thresh, sign, offset, out_bit);
    }
    else {
        CnBwn(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pad, pool, pOut, out_bit);
    }
}
/**
 * @details Dense binary-weight Convolutional (CN) layer with output binarization.
 * Pooling/padding support
 * 
 * @param[in] pAct - pointer to the packed activation vector (depth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels
 * @param[in] pad  - padding size 
 * @param[in] pool - pooling window size 
 * @param[out] pOut - pointer to the packed output vector (depth-width-height)
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnBwn(int8_t* __restrict pAct, int8_t * __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool, 
    pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * sign, pckDtype* __restrict offset, uint8_t out_bit) {

   int32_t     outTemp=0;
   int32_t  outTempInt = 0;
   pckDtype pckTemp [out_bit];
   uint8_t  yCoeff  = wdth*dpth;
   uint8_t  xCoeff  = dpth;
   uint8_t  kCoeff  = khgt*kwdt*kdpt;
   uint8_t  kyCoeff = kwdt*kdpt;
   uint8_t  kxCoeff = kdpt;
   // Starting indices for padding
   int8_t  xStart, yStart = 0;
   uint8_t  xxStart, yyStart = 0;
   // Ending indices for padding
   int8_t  xEnd, yEnd = 0;
   uint8_t  xxEnd, yyEnd = 0;
   pckDtype  *signs = sign;
   pckDtype   *threshLoc = thresh;
   pckDtype* offsets = offset;
   int   maxTemp = 0;
   int16_t   convAct=0;
   uint8_t  xyCount = 0;

 
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+2*pad+1)/pool; y++) {
      // Account for padding - skip padded values
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1)/pool; x++) {
         // Account for padding - skip padded values
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         offsets = offset;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            //pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum / pckWdt + x * knum / pckWdt + k] = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               maxTemp = -INFINITY;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     outTemp = 0;
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                           // Z dim
                           for (uint16_t z = 0; z < dpth; z++) {
                              convAct = pAct[(y*pool+yy+ky-pad)*yCoeff + (x*pool+xx+kx-pad)*xCoeff + z];
                              if (pKrn[(k*pckWdt+ks)*kCoeff + ky*kyCoeff + kx*kxCoeff + z] == 1) {
                                 outTemp +=  convAct;
                              }
                              else {
                                 outTemp -=  convAct; 
                              }
                           } // Z
                        } // K-X
                     } // K-Y
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Batch normalize/ binarize
               int out_temp = maxTemp << (16);/// pow(2, in_bit);
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
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pOut++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
   }
}
/**
 * @details Dense binary-weight Convolutional (CN) layer with output binarization.
 * Pooling/padding support
 *
 * @param[in] pAct - pointer to the packed activation vector (depth-width-height)
 * @param[in] pKrn - pointer to the packed kernel vector (depth-width-height-kernel)
 * @param[in] dpth - activation depth
 * @param[in] wdth - activation width
 * @param[in] hght - activation height
 * @param[in] kdpt - kernel depth
 * @param[in] kwdt - kernel width
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels
 * @param[in] pad  - padding size
 * @param[in] pool - pooling window size
 * @param[out] pOut - pointer to the packed output vector (depth-width-height)
 * @param[in] thresh - pointer to batch normalization threshold
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBwn(int8_t* __restrict pAct, int8_t* __restrict pKrn, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt,
    const uint16_t khgt, const uint16_t knum, const uint16_t pad, const uint16_t pool,
    pckDtype* __restrict pOut, uint8_t out_bit) {

    int32_t     outTemp = 0;
    int32_t  outTempInt = 0;
    pckDtype pckTemp[out_bit];
    uint8_t  yCoeff = wdth * dpth;
    uint8_t  xCoeff = dpth;
    uint8_t  kCoeff = khgt * kwdt * kdpt;
    uint8_t  kyCoeff = kwdt * kdpt;
    uint8_t  kxCoeff = kdpt;
    // Starting indices for padding
    int8_t  xStart, yStart = 0;
    uint8_t  xxStart, yyStart = 0;
    // Ending indices for padding
    int8_t  xEnd, yEnd = 0;
    uint8_t  xxEnd, yyEnd = 0;
    int   maxTemp = 0;
    int16_t   convAct = 0;
    uint8_t  xyCount = 0;


    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 2 * pad + 1) / pool; y++) {
        // Account for padding - skip padded values
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 2 * pad + 1) / pool; x++) {
            // Account for padding - skip padded values
            // Restart kernel bn pointer
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                //pOut[y * ((wdth - kwdt + 2 * pad + 1) / pool) * knum / pckWdt + x * knum / pckWdt + k] = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    maxTemp = -INFINITY;
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        if ((y * pool + yy) < pad) { yStart = pad - (y * pool + yy); }
                        else { yStart = 0; }
                        if ((y * pool + yy) > hght - khgt + pad) { yEnd = hght - ((y * pool + yy) - pad); }
                        else { yEnd = khgt; }
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            if ((x * pool + xx) < pad) { xStart = pad - (x * pool + xx); }
                            else { xStart = 0; }
                            if ((x * pool + xx) > wdth - kwdt + pad) { xEnd = wdth - ((x * pool + xx) - pad); }
                            else { xEnd = kwdt; }
                            outTemp = 0;
                            xyCount = 0;
                            // K-Y dim
                            for (uint16_t ky = yStart; ky < yEnd; ky++) {
                                // K-X dim
                                for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                    // Z dim
                                    for (uint16_t z = 0; z < dpth; z++) {
                                        convAct = pAct[(y * pool + yy + ky - pad) * yCoeff + (x * pool + xx + kx - pad) * xCoeff + z];
                                        if (pKrn[(k * pckWdt + ks) * kCoeff + ky * kyCoeff + kx * kxCoeff + z] == 1) {
                                            outTemp += convAct;
                                        }
                                        else {
                                            outTemp -= convAct;
                                        }
                                    } // Z
                                } // K-X
                            } // K-Y
                            if (outTemp > maxTemp) { maxTemp = outTemp; }
                        }
                    }
                    // Batch normalize/ binarize
                    int out_temp = maxTemp << (16);/// pow(2, in_bit);
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        int temp = out_temp > 0;
                        // Shift 
                        pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                        out_temp = (temp==0) ? out_temp + (1 << (out_bit - bitw - 1)) : out_temp - (1 << (out_bit - bitw - 1));
                        //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                    }
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pOut++ = pckTemp[bitw];
                }
            }
        }
    }
}
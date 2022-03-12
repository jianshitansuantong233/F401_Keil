/*
* MIT License
* 
* Copyright (c) 2019 UCLA NanoCAD Laboratory 
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without __restriction, including without limitation the rights
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
 * \file      xnor_dense_cn.c
 * \brief     Dense binarized (XNOR) convolutional layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "xnor_cn.h"
#include <limits.h>
#include <stdio.h>
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, NEON support)
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 * @return 0 - Success, 1 - Failure
 */
uint8_t CnXnorWrap(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint16_t pad, const uint16_t pool, pckDtype * __restrict thresh, pckDtype * sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {

   // Batch Norm present (thresh != NULL)
   if (thresh) {
      // Output or input depth not multiple of pack width - not supported atm
      if (dpth  % pckWdt != 0 || knum % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (dpth  % (4*pckWdt) == 0) {
         return 1;
      }
      // Check for dual SIMD support
      else if (dpth  % (2*pckWdt) == 0) {
         return 1;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         // No padding
         if (pad == 0) {
            // No pooling
            if (pool == 1) {
               CnBnXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, thresh, sign, offset, in_bit, out_bit);
            }
            // Pooling
            else {
               CnBnPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool, thresh, sign, offset, in_bit, out_bit);
            }
         }
         // Padding
         else {
            // No pooling
            if (pool == 1) {
               CnBnPdXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, thresh, sign, offset, in_bit, out_bit);
            }
            // Pooling
            else {
               CnBnPdPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, thresh, sign, offset, in_bit, out_bit);
            }

         }
         return 0;
      }
   }
   // No Batch Norm (bnDtype == NULL)
   else {
      // Output or input not depth multiple of pack width - not supported atm
      if (dpth % pckWdt != 0 || knum % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (dpth  % (4*pckWdt) == 0) {
         return 1;
      }
      // Check for dual SIMD support
      else if (dpth  % (2*pckWdt) == 0) {
         return 1;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         // No padding
         if (pad == 0) {
            // No pooling
            if (pool == 1) {
               CnXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, in_bit, out_bit);
               //CnXnorKOut(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut);
            }
            // Pooling
            else {
               CnPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool, in_bit, out_bit);
            }
         }
         // Padding
         else {
            // No pooling
            if (pool == 1) {
               CnPdXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, in_bit, out_bit);
            }
            // Pooling
            else {
               CnPdPlXnor(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, in_bit, out_bit);
            }

         }
         return 0;
      }
   }

}



uint8_t CnXnorNoBinWrap(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint16_t pad, const uint16_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta, uint8_t in_bit, uint8_t out_bit) {

    // Batch Norm present (thresh != NULL)
    if (mean) {
        // Output or input depth not multiple of pack width - not supported atm
        if (dpth % pckWdt != 0 ) {
            return 1;
        }
        // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
        else if (dpth % (4 * pckWdt) == 0) {
            return 1;
        }
        // Check for dual SIMD support
        else if (dpth % (2 * pckWdt) == 0) {
            return 1;
        }
#endif
#endif
        // Roll back to default implementation
        else {
            // No padding
            if (pad == 0) {
                // No pooling
                if (pool == 1) {
                    CnBnXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, mean, var, gamma, beta, in_bit, out_bit);
                }
                // Pooling
                else {
                    CnBnPlXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pool, mean,var,gamma,beta, in_bit, out_bit);
                }
            }
            // Padding
            else {
                // No pooling
                if (pool == 1) {
                    CnBnPdXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, mean, var,gamma,beta, in_bit, out_bit);
                }
                // Pooling
                else {
                    CnBnPdPlXnorNoBin(pAct, pKrn, dpth, wdth, hght, kdpt, kwdt, khgt, knum, pOut, pad, pool, mean, var,gamma,beta, in_bit, out_bit);
                }

            }
            return 0;
        }
    }

}



/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop
 * Outer loop: xy, Pad: no, Pool: no BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 */
void CnXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, uint8_t in_bit, uint8_t out_bit) {

    // Temporary variables
    uint32_t xnorTemp;
    int32_t  outTemp;
    int32_t pckTemp[out_bit];
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    pckDtype* pRes = pOut;
    uint16_t  yCoeff = wdth * dpth * in_bit / pckWdt;
    uint16_t  xCoeff = dpth * in_bit / pckWdt;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;
    int32_t out = 0;
    uint8_t output_bit = 0;
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    pIn = pAct + y * yCoeff + x * xCoeff;
                    outTemp = 0;
                    out = 0;
                    output_bit = 0;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        pIn += bitw;
                        // K-Y dim
                        for (uint16_t ky = 0; ky < khgt; ky++) {
                            // K-X dim
                            for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                                // XNOR multiplication
                                xnorTemp = ~(*(pIn) ^ *(pWgt++));
                                outTemp += popcount(xnorTemp);
                                pIn += in_bit;
                            }// K-X dim     
                            // Move the activation pointer one row down 
                            pIn += (wdth - kwdt) * dpth * in_bit / pckWdt - bitw;
                        }// K-Y dim   
                        pWgt -= cntCoeff / pckWdt;
                        // Adjust the output value
                        outTemp = outTemp - (cntCoeff - outTemp);
                        // Get the int full precision value 
                        out += (outTemp << (in_bit - bitw - 1));
                        int up_thresh = 0;
                        for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                            up_thresh += (cntCoeff << (in_bit - bitt - 1));
                        }
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {                            
                            if (out > up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out = out - (1 << (out_bit - bito - 1));
                                output_bit++;
                            }
                            else if (out < -up_thresh) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out = out + (1 << (out_bit - bito - 1));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }    
                    }                    
                    pWgt += cntCoeff / pckWdt;
                    // We've only counted ones, but we want a difference between +1s and -1s 
                    // so we need to adjust the result
                    // Below is shorter for
                    // outTemp = outTemp - (2*cntCoeff - outTemp);
                    // outTemp = outTemp >= 0;
                    /*
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (cntCoeff - outTemp[bitw]);
                        // Get the int full precision value
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Quantization
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        int temp = out > 0;
                        // Shift
                        pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                        out = (temp == 0 ? out + (1 << (out_bit - bitw - 1)) : out - (1 << (out_bit - bitw - 1)));
                    }
                 }*/
                }
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                            *pRes++ = pckTemp[bitw];
                }
            }
        }
    }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding
 * Outer loop: xy, Pad: yes, Pool: no BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 */
void CnPdXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, uint8_t in_bit, uint8_t out_bit) {
    
    // Temporary variables
    uint32_t xnorTemp;
    int32_t  outTemp;
    int32_t pckTemp[out_bit];
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   uint16_t  yCoeff  = wdth*dpth*in_bit/pckWdt;
   uint16_t  xCoeff  = dpth*in_bit/pckWdt;
   uint16_t  cntCoeff = khgt*kwdt*kdpt;
   int32_t out = 0;
   uint8_t output_bit = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (pad+y)*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt + pad*knum*out_bit/pckWdt;
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
             memset(pckTemp, 0, sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               outTemp = 0;
               out = 0;
               output_bit = 0;
               for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                   pIn= pAct + y * yCoeff + x * xCoeff;
                   pIn += bitw;
                    // K-Y dim
                   for (uint16_t ky = 0; ky < khgt; ky++) {
                       // K-X dim
                       for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                           // XNOR multiplication
                           xnorTemp = ~(*pIn ^ *(pWgt++));
                           outTemp += popcount(xnorTemp);
                           pIn += in_bit;
                       }// K-X dim
                       pIn += (wdth - kwdt) * dpth * in_bit / pckWdt - bitw;
                   } // K-Y dim
                  // Move the activation pointer one row down
                   pWgt -= cntCoeff / pckWdt;
                   outTemp = outTemp - (cntCoeff - outTemp);
                   // Get the int full precision value 
                   out += (outTemp << (in_bit - bitw - 1));
                   int up_thresh = 0;
                   for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                       up_thresh += (cntCoeff << (in_bit - bitt - 1));
                   }
                   // Quantization
                   for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                       
                       if (out > up_thresh) {
                           pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                           out = out - (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else if (out < -up_thresh) {
                           pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                           out = out + (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else {
                           break;
                       }
                   }
               } 
               pWgt += cntCoeff / pckWdt;
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }

   // Top
   pRes = pOut;
   // Y dim
   for (uint16_t y = 0; y < pad; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
             memset(pckTemp, 0, sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
                outTemp = 0;
                out = 0;
                output_bit = 0;               
               pckDtype* weight_temp = pWgt;
               for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                   xyCount = 0;
                   // K-Y dim
                   for (uint16_t ky = yStart; ky < yEnd; ky++) {
                       // Move the input pointer to the first non-padded activation block
                       pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                       pIn += bitw;
                       // K-X dim
                       for (uint16_t kx = xStart; kx < xEnd; kx++) {
                           xyCount++;
                           // Z dim
                           for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                               // XNOR multiplication
                               xnorTemp = ~(*pIn ^ *pWgt++);
                               outTemp += popcount(xnorTemp);
                               pIn += in_bit;
                           }// Z dim        
                       }// K-X dim 
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                   } // K-Y dim
                   if (bitw != in_bit - 1) pWgt = weight_temp;
                   outTemp = outTemp - (xyCount * kdpt - outTemp);
                   out += (outTemp << (in_bit - bitw - 1));
                   int up_thresh = 0;
                   for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                       up_thresh += ((xyCount*kdpt) << (in_bit - bitt - 1));
                   }
                   for (uint8_t bito = output_bit; bito != out_bit; bito++) {                       
                       if (out > up_thresh) {
                           pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                           out = out - (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else if (out < -up_thresh) {
                           pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                           out = out + (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else {
                           break;
                       }
                   }
               }
               // Shift the weight pointer to the next kernel
               pWgt += yStart*kwdt*kdpt/pckWdt;
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + (hght-khgt+pad+1)*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = hght-khgt+pad+1; y < hght-khgt+2*pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
             memset(pckTemp, 0, sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
                outTemp = 0;
                out = 0;
                output_bit = 0;            
                pckDtype* weight_temp = pWgt;
               for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                   xyCount = 0;
                   // K-Y dim
                   for (uint16_t ky = yStart; ky < yEnd; ky++) {
                       // Move the input pointer to the first non-padded activation block
                       pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                       pIn += bitw;
                       // K-X dim
                       for (uint16_t kx = xStart; kx < xEnd; kx++) {
                           xyCount++;
                           // Z dim
                           for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                               // XNOR multiplication
                               xnorTemp = ~(*pIn ^ *pWgt++);
                               outTemp += popcount(xnorTemp);
                               pIn += in_bit;
                           }// Z dim
                       } // K-X dim
                       // Move the weight poitner to the next row
                       pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                   } // K-Y dim
                   if (bitw != in_bit - 1) pWgt = weight_temp;
                   outTemp = outTemp - (xyCount * kdpt - outTemp);
                   out += (outTemp << (in_bit - bitw - 1));
                   int up_thresh = 0;
                   for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                       up_thresh += ((xyCount*kdpt) << (in_bit - bitt - 1));
                   }
                   for (uint8_t bito = output_bit; bito != out_bit; bito++) {                       
                       if (out > up_thresh) {
                           pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                           out = out - (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else if (out < -up_thresh) {
                           pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                           out = out + (1 << (out_bit - bito - 1));
                           output_bit++;
                       }
                       else {
                           break;
                       }
                   }
               } 
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }
  
   // Left 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < pad; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
             memset(pckTemp, 0, sizeof(pckTemp));
             for (uint16_t ks = 0; ks < pckWdt; ks++) {
                 outTemp = 0;
                 out = 0;
                 output_bit = 0;
                 pckDtype* weight_temp = pWgt;
                 for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                     xyCount = 0;
                     // K-Y dim
                     for (uint16_t ky = yStart; ky < yEnd; ky++) {
                         // Move the input pointer to the first non-padded activation block
                         pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                         pIn += bitw;
                         // K-X dim
                         for (uint16_t kx = xStart; kx < xEnd; kx++) {
                             xyCount++;
                             // Z dim
                             for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                 // XNOR multiplication
                                 xnorTemp = ~(*pIn ^ *pWgt++);
                                 outTemp += popcount(xnorTemp);
                             }// Z dim
                         } // K-X dim
                         // Move the weight poitner to the next row
                         pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                     } // K-Y dim
                     if (bitw != in_bit - 1) pWgt = weight_temp;
                     outTemp = outTemp - (xyCount * kdpt - outTemp);
                     out += (outTemp << (in_bit - bitw - 1));
                     int up_thresh = 0;
                     for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                         up_thresh += ((xyCount*kdpt) << (in_bit - bitt - 1));
                     }
                     for (uint8_t bito = output_bit; bito != out_bit; bito++) {                         
                         if (out > up_thresh) {
                             pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                             out = out - (1 << (out_bit - bito - 1));
                             output_bit++;
                         }
                         else if (out < -up_thresh) {
                             pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                             out = out + (1 << (out_bit - bito - 1));
                             output_bit++;
                         }
                         else {
                             break;
                         }
                     }
                 } 
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
             }
             for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                 *pRes++ = pckTemp[bitw];
             }
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt;
   }

   // Right 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt + (wdth-kwdt+pad+1)*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = wdth-kwdt+pad+1; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp, 0, sizeof(pckTemp));
            for (uint16_t ks = 0; ks < pckWdt; ks++) {
                outTemp = 0;
                out = 0; 
                pckDtype* weight_temp = pWgt;
                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                // XNOR multiplication
                                xnorTemp = ~(*pIn ^ *pWgt++);
                                outTemp += popcount(xnorTemp);
                                pIn += in_bit;
                            }// Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    if (bitw != in_bit - 1) pWgt = weight_temp;
                    outTemp = outTemp - (xyCount * kdpt - outTemp);
                    out += (outTemp << (in_bit - bitw - 1));
                    int up_thresh = 0;
                    for (uint8_t bitt = bitw; bitt != in_bit; bitt++) {
                        up_thresh += ((xyCount*kdpt) << (in_bit - bitt - 1));
                    }
                    for (uint8_t bito = output_bit; bito != out_bit; bito++) {                        
                        if (out > up_thresh) {
                            pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                            out = out - (1 << (out_bit - bito - 1));
                            output_bit++;
                        }
                        else if (out < -up_thresh) {
                            pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                            out = out + (1 << (out_bit - bito - 1));
                            output_bit++;
                        }
                        else {
                            break;
                        }
                    }
                } 
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum*out_bit/pckWdt + (wdth-kwdt+pad+1)*knum*out_bit/pckWdt;
   }
}


/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, pooling
 * Outer loop: xy, Pad: no, Pool: yes, BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pool - pooling size
 */
void CnPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pool, uint8_t in_bit, uint8_t out_bit) {

    // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    // For maxpooling
    int32_t  maxTemp = 0;
    int32_t out = 0;
    //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
    pckDtype pckTemp[out_bit];
    uint16_t  yCoeff = wdth * dpth * in_bit / pckWdt;
    uint16_t  xCoeff = dpth * in_bit / pckWdt;
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    pckDtype* pRes = pOut;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;

    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1) / pool; y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1) / pool; x++) {
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    // Mpool patches
                    maxTemp = -(khgt * kwdt * kdpt);
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            memset(outTemp, 0, sizeof(outTemp));
                            out = 0;
                            for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt;
                                pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff + bitw;
                                // K-Y dim
                                for (uint16_t ky = 0; ky < khgt; ky++) {
                                    // K-X dim
                                    for (uint16_t kx = 0; kx < kwdt; kx++) {
                                        // Z dim
                                        for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                            // XNOR multiplication
                                            xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                            outTemp[bitw] += popcount(xnorTemp[bitw]);
                                            pIn += in_bit;
                                        }// Z dim
                                    } // K-X dim
                                    pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                                } // K-Y dim
                                // Adjust the output value
                                outTemp[bitw] = outTemp[bitw] - (cntCoeff - outTemp[bitw]);
                                // Get the int full precision value 
                                out += (outTemp[bitw] << (in_bit - bitw - 1));
                                int temp_thresh = 0;
                                for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                                    temp_thresh += (cntCoeff << (in_bit - bitt - 1));
                                }
                                if (out + temp_thresh < maxTemp) {
                                    break;
                                }
                            }                             
                            // Maxpool
                            if (out > maxTemp) { maxTemp = out; }
                        } // X-MP
                    } // Y-MP
                    // Quantization
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        int temp = maxTemp > 0;
                        // Shift 
                        pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                        maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
                    }
                }
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = pckTemp[bitw];
                }
            }
        }
    }
}


/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, padding/pooling
 * Outer loop: xy, Pad: yes, Pool: yes, BatchNorm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 */
void CnPdPlXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, uint8_t in_bit, uint8_t out_bit) {

   // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    pckDtype pckTemp[out_bit];
   uint16_t  yCoeff  = wdth*dpth*in_bit/pckWdt;
   uint16_t  xCoeff  = dpth*in_bit/pckWdt;
   int32_t out = 0;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   uint16_t  cntCoeff = khgt*kwdt*kdpt;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For maxpooling
   int32_t  maxTemp = 0;


   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (y)*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt + ((pad+pool-1)/pool)*knum*out_bit/pckWdt;
      for (uint16_t x = ((pad+pool-1)/pool); x <= (wdth-kwdt+2*pad+1)/pool - 2*((pad+pool-1)/pool); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp,0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {
                      memset(outTemp, 0, sizeof(outTemp));
                      out = 0;
                      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                          pIn = pAct + (y * pool + yy - pad) * yCoeff + (x * pool + xx - pad) * xCoeff + bitw;
                          pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt;
                          // K-Y dim
                          for (uint16_t ky = 0; ky < khgt; ky++) {
                              // K-X dim
                              for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                                  // XNOR multiplication
                                  xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                  outTemp[bitw] += popcount(xnorTemp[bitw]);
                                  pIn += in_bit;
                              }// K-X dim
                              // Move the activation pointer one row down
                              pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                          }// K-Y dim
                          // Adjust the output value
                          outTemp[bitw] = outTemp[bitw] - (cntCoeff - outTemp[bitw]);
                          // Get the int full precision value 
                          out += (outTemp[bitw] << (in_bit - bitw - 1));
                      }
                     // Maxpool
                     if (out > maxTemp) { maxTemp = out;}
                  } // X-MP
               } // Y-MP
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = maxTemp > 0;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
               }
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }

   //// Top
   pRes = pOut;
   // Y dim
   // We need to make sure there's enough lines to do pooling
   //for (uint16_t y = 0; y < pad; y++) {
   for (uint16_t y = 0; y < (pad+pool-1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad+1)/pool; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     memset(outTemp,0,sizeof(outTemp));
                     out = 0;
                     for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                         // Move the wieight pointer to the fisrt useful (non-padded) weight block
                         pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
                         xyCount = 0;
                         // K-Y dim
                         for (uint16_t ky = yStart; ky < yEnd; ky++) {
                             // Move the input pointer to the first non-padded activation block
                             pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                             // K-X dim
                             for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                 xyCount++;
                                 // Z dim
                                 for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                     // XNOR multiplication
                                     xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                     outTemp[bitw] += popcount(xnorTemp[bitw]);
                                     pIn += in_bit;
                                 }// Z dim
                             } // K-X dim
                             // Move the weight poitner to the next row
                             pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                         } // K-Y dim
                         outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                         out += (outTemp[bitw] << (in_bit - bitw - 1));
                     }
                     // Maxpool
                     if (out > maxTemp) { maxTemp = out;}
                  }
               }
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = maxTemp > 0;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
               }
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + ((hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool))*((wdth-kwdt+2*pad+1)/pool)*out_bit*knum/pckWdt;
   // Y dim
   for (uint16_t y = (hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool); y < (hght-khgt+2*pad)/pool + 1; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp,0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     memset(outTemp, 0, sizeof(outTemp));
                     out = 0;
                     for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                         // Move the wieight pointer to the fisrt useful (non-padded) weight block
                         pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
                         xyCount = 0;
                         // K-Y dim
                         for (uint16_t ky = yStart; ky < yEnd; ky++) {
                             // Move the input pointer to the first non-padded activation block
                             pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                             // K-X dim
                             for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                 xyCount++;
                                 // Z dim
                                 for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                     // XNOR multiplication
                                     xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                     outTemp[bitw] += popcount(xnorTemp[bitw]);
                                     pIn += in_bit;
                                 }// Z dim
                             } // K-X dim
                              // Move the weight poitner to the next row
                             pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                         }// K-Y dim
                         outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                         out += (outTemp[bitw] << (in_bit - bitw - 1));
                     }
                     // Maxpool
                     if (out > maxTemp) { maxTemp = out; }
                  }
               }
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = maxTemp > 0;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
               }
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
   }
  
   //// Left 
   pRes = pOut + ((pad+pool-1)/pool)*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = 0; x < ((pad+pool-1)/pool); x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
                maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     memset(outTemp, 0,sizeof(outTemp));
                     out = 0;
                     for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                         xyCount = 0;
                         // Move the wieight pointer to the fisrt useful (non-padded) weight block
                         pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
                         // K-Y dim
                         for (uint16_t ky = yStart; ky < yEnd; ky++) {
                             // Move the input pointer to the first non-padded activation block
                             pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                             // K-X dim
                             for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                 xyCount++;
                                 // Z dim
                                 for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                     // XNOR multiplication
                                     xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                     outTemp[bitw] += popcount(xnorTemp[bitw]);
                                     pIn += in_bit;
                                 }// Z dim
                             } // K-X dim
                             // Move the weight poitner to the next row
                             pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                         } // K-Y dim
                         outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                         out += (outTemp[bitw] << (in_bit - bitw - 1));
                     }
                     // Maxpool
                     if (out > maxTemp) { maxTemp = out; }
                  }
               }
               // Binarize
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = maxTemp > 0;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
               }
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
      pRes = pOut + (y+1)*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt;
   }

   // Right 
   pRes = pOut + ((pad+pool-1)/pool)*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt + (((wdth-kwdt+2*pad+1)/pool) - ((pad+pool-1)/pool))*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= (hght-khgt+2*pad+1)/pool - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = (wdth-kwdt+2*pad)/pool + 1 - ((pad+pool-1)/pool); x < (wdth-kwdt+2*pad)/pool + 1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp,0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     // Account for padding - skip padded values
                     if ((x*pool+xx) < pad) { xStart = pad-(x*pool+xx); } else { xStart = 0; }
                     if ((x*pool+xx) > wdth-kwdt+pad) { xEnd = wdth - ((x*pool+xx)-pad); } else { xEnd = kwdt; }
                     memset(outTemp, 0,sizeof(outTemp));
                     out = 0;
                     for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                         // Move the wieight pointer to the fisrt useful (non-padded) weight block
                         pWgt = pKrn + (k * pckWdt + ks) * cntCoeff / pckWdt + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
                         xyCount = 0;
                         // K-Y dim
                         for (uint16_t ky = yStart; ky < yEnd; ky++) {
                             // Move the input pointer to the first non-padded activation block
                             pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                             // K-X dim
                             for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                 xyCount++;
                                 // Z dim
                                 for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                     // XNOR multiplication
                                     xnorTemp[bitw] = ~(*pIn ^ *pWgt++);
                                     outTemp[bitw] += popcount(xnorTemp[bitw]);
                                     pIn += in_bit;
                                 }// Z dim
                             } // K-X dim
                             // Move the weight poitner to the next row
                             pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                         }// K-Y dim
                         outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                         out += (outTemp[bitw] << (in_bit - bitw - 1));
                     } 
                     // Maxpool
                     if (out > maxTemp) { maxTemp = out; }
                  }
               }
               // Binarize
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = maxTemp > 0;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   maxTemp = (temp == 0 ? maxTemp + (1 << (out_bit - bitw - 1)) : maxTemp - (1 << (out_bit - bitw - 1)));
               }
            }
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = pckTemp[bitw];
            }
         }
      }
      pRes = pOut + (y+1)*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt + (((wdth-kwdt+2*pad+1)/pool) - ((pad+pool-1)/pool))*knum*out_bit/pckWdt;
   }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm
 * Outer loop: XY, Pad: no, Pool: no BatchNorm: yes, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnXnor(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {
    // Temporary variables
    pckDtype xnorTemp;
    int32_t  outTemp;
    pckDtype pckTemp[out_bit];
    memset(pckTemp, 0, sizeof(pckTemp));
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    pckDtype* pRes = pOut;
    pckDtype* signs = sign;
    pckDtype* offsets = offset;
    pckDtype* threshLoc = thresh;
    uint16_t  yCoeff = wdth * dpth * in_bit / pckWdt;
    uint16_t  xCoeff = dpth * in_bit / pckWdt;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;
    int out = 0;
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            // Outer loop - kernels
            pWgt = pKrn;
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                out = 0;
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        pIn = pAct + y * yCoeff + x * xCoeff;
                        outTemp = 0;
                        // K-Y dim
                        for (uint16_t ky = 0; ky < khgt; ky++) {
                            // K-X dim
                            for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                                // XNOR multiplication
                                xnorTemp = ~(*pIn ^ *pWgt++);
                                outTemp += popcount(xnorTemp);
                                pIn += in_bit;
                            }// K-X dim
                             // Move the activation pointer one row down
                            pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                        } // K-Y dim        
                        pWgt -= cntCoeff / pckWdt;
                        outTemp = outTemp - (cntCoeff - outTemp);
                        // Get the int full precision value 
                        out += (outTemp << (in_bit - bitw - 1));
                    }
                    int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                    int temp = 0;
                    for (int i = 0; i != in_bit; i++) {
                        temp |= (1 << i);
                    }
                    temp = temp & out;
                    out_temp += temp;
                    for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                        int temp = out_temp > *threshLoc;
                        // Shift 
                        pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                        //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                        out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                    }
                    // Batch normalize/ binarize
                    threshLoc++;
                    offsets++;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = pckTemp[bitw] & (*signs);
                    pckTemp[bitw] = 0;
                }
                signs++;
            }
        }
    }
}

void CnBnXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth,
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt,
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut,
    bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta, uint8_t in_bit, uint8_t out_bit) {
    // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    pckDtype pckTemp[out_bit];
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;
    uint16_t  yCoeff = wdth * dpth * in_bit / pckWdt;
    uint16_t  xCoeff = dpth * in_bit / pckWdt;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;
    int out = 0;
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            for (uint16_t k = 0; k < knum; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                pIn = pAct + y * yCoeff + x * xCoeff;
                memset(outTemp, 0, sizeof(outTemp));
                out = 0;
                // K-Y dim
                for (uint16_t ky = 0; ky < khgt; ky++) {
                    // K-X dim
                    for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                        for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                            // XNOR multiplication
                            xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                            outTemp[bitw] += popcount(xnorTemp[bitw]);
                        }
                        pWgt++;
                    } // K-X dim
                    // Move the activation pointer one row down
                    pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                } // K-Y dim
                // We've only counted ones, but we want a difference between +1s and -1s 
                // so we need to adjust the result
                // Below is shorter for
                // outTemp = outTemp - (2*cntCoeff - outTemp);
                // outTemp = outTemp >= 0;
                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                    // Adjust the output value
                    outTemp[bitw] = outTemp[bitw] - (cntCoeff - outTemp[bitw]);
                    // Get the int full precision value 
                    out += (outTemp[bitw] << (in_bit - bitw - 1));
                }
                float out_temp = out / (float)(1 << (in_bit));// pow(2, in_bit);
                float temp = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                pOut[k] = temp;
            }
        }
    }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding
 * Outer loop: XY, Pad: yes, Pool: no BatchNorm: yes, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pad, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {

    // Temporary variables
    pckDtype xnorTemp;
    int32_t  outTemp;
    pckDtype pckTemp[out_bit];
    int out = 0;
    uint8_t output_bit = 0;
    uint16_t  yCoeff = wdth * dpth * in_bit / pckWdt;
    uint16_t  xCoeff = dpth * in_bit / pckWdt;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* weight_temp = pWgt;
    pckDtype* pIn = pAct;
    pckDtype* pRes = pOut;
    pckDtype* signs = sign;
    pckDtype* threshLoc = thresh;
    pckDtype* offsets = offset;
    int out_acc = 0;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;
    // Starting indices for padding
    uint16_t  xStart, yStart = 0;
    // Ending indices for padding
    uint16_t  xEnd, yEnd = 0;
    // Divide the input into 5 regions - top, bottom, left, right, middle 
    // Middle has no padding
    // Middle - no padding
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        // Set the output pointer
        // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
        // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
        // Offset to this row pad*knum/pckWdt
        pRes = pOut + (pad + y) * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt + pad * knum * out_bit / pckWdt;
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {

                    output_bit = 0;
                    out_acc = 0;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        outTemp = 0;
                        pIn = pAct + y * yCoeff + x * xCoeff + bitw;
                        // K-Y dim
                        for (uint16_t ky = 0; ky < khgt; ky++) {
                            // K-X dim
                            for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                                // XNOR multiplication
                                xnorTemp = ~(*pIn ^ *pWgt++);
                                outTemp += popcount(xnorTemp);
                                pIn += in_bit;
                            }// K-X dim
                            // Move the activation pointer one row down
                            pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                        } // K-Y dim
                        pWgt -= cntCoeff / pckWdt;
                        outTemp = outTemp - (cntCoeff - outTemp);
                        // Get the int full precision value 
                        out = (outTemp << (in_bit - bitw - 1));
                        // Quantization
                        int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                        int temp = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp |= (1 << i);
                        }
                        temp = (temp & out) << (16 - in_bit);
                        out_acc += out_temp + temp;
                        int up_thresh = 0;
                        int temp_thresh = 0;
                        for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                            temp_thresh += (cntCoeff << (in_bit - bitt - 1));
                        }
                        int temp_fixed = temp_thresh >> (in_bit + 2) << (16);/// pow(2, in_bit);
                        int temp_pack = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp_pack |= (1 << i);
                        }
                        temp_pack = (temp_pack & temp_thresh) << (16 - in_bit);
                        up_thresh = temp_fixed + temp_pack;
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                            if (out_acc > *threshLoc + up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out_acc += (1 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else if (out_acc < *threshLoc - up_thresh || (bitw == in_bit - 1)) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out_acc += (0 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }
                        if (output_bit == out_bit) {
                            break;
                        }
                    }
                    threshLoc++;
                    offsets++;
                    pWgt += cntCoeff / pckWdt;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
    }
    // Top
    pRes = pOut;
    // Y dim
    for (uint16_t y = 0; y < pad; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {

                    out = 0;
                    out_acc = 0;
                    output_bit = 0;
                    weight_temp = pWgt;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        outTemp = 0;
                        pWgt = weight_temp;
                        xyCount = 0;
                        // K-Y dim
                        for (uint16_t ky = yStart; ky < yEnd; ky++) {
                            // Move the input pointer to the first non-padded activation block
                            pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff + bitw;
                            // K-X dim
                            for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                xyCount++;
                                // Z dim
                                for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                    // XNOR multiplication
                                    xnorTemp = ~(*pIn ^ *pWgt++);
                                    outTemp += popcount(xnorTemp);
                                    pIn += in_bit;
                                }// Z dim
                            } // K-X dim
                             // Move the weight poitner to the next row
                            pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                        } // K-Y dim
                        outTemp = outTemp - (xyCount * kdpt - outTemp);
                        // Get the int full precision value 
                        out = (outTemp << (in_bit - bitw - 1));
                        // Quantization
                        int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                        int temp = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp |= (1 << i);
                        }
                        temp = (temp & out) << (16 - in_bit);
                        out_acc += out_temp + temp;
                        int up_thresh = 0;
                        int temp_thresh = 0;
                        for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                            temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                        }
                        int temp_fixed = temp_thresh >> (in_bit + 2) << (16);/// pow(2, in_bit);
                        int temp_pack = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp_pack |= (1 << i);
                        }
                        temp_pack = (temp_pack & temp_thresh) << (16 - in_bit);
                        up_thresh = temp_fixed + temp_pack;
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                            if (out_acc > *threshLoc + up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out_acc += (1 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else if (out_acc < *threshLoc - up_thresh || (bitw == in_bit - 1)) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out_acc += (0 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }
                        if (output_bit == out_bit) {
                            break;
                        }
                    }
                    threshLoc++;
                    offsets++;
                    // Shift the weight pointer to the next kernel
                    pWgt += yStart * kwdt * kdpt / pckWdt;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
    }

    // Bottom 
    // Move the ouput pointer
    pRes = pOut + (hght - khgt + pad + 1) * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt;
    // Y dim
    for (uint16_t y = hght - khgt + pad + 1; y < hght - khgt + 2 * pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {

                    out = 0;
                    out_acc = 0;
                    output_bit = 0;
                    weight_temp = pWgt;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        outTemp = 0;
                        xyCount = 0;
                        pWgt = weight_temp;
                        // K-Y dim
                        for (uint16_t ky = yStart; ky < yEnd; ky++) {
                            // Move the input pointer to the first non-padded activation block
                            pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff + bitw;
                            // K-X dim
                            for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                xyCount++;
                                // Z dim
                                for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                    // XNOR multiplication
                                    xnorTemp = ~(*pIn ^ *pWgt++);
                                    outTemp += popcount(xnorTemp);
                                    pIn += in_bit;
                                }// Z dim                            
                            } // K-X dim
                              // Move the weight poitner to the next row
                            pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                        } // K-Y dim
                        outTemp = outTemp - (xyCount * kdpt - outTemp);
                        // Get the int full precision value 
                        out = (outTemp << (in_bit - bitw - 1));
                        // Quantization
                        int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                        int temp = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp |= (1 << i);
                        }
                        temp = (temp & out) << (16 - in_bit);
                        out_acc += out_temp + temp;
                        int up_thresh = 0;
                        int temp_thresh = 0;
                        for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                            temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                        }
                        int temp_fixed = temp_thresh >> (in_bit + 2) << (16);/// pow(2, in_bit);
                        int temp_pack = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp_pack |= (1 << i);
                        }
                        temp_pack = (temp_pack & temp_thresh) << (16 - in_bit);
                        up_thresh = temp_fixed + temp_pack;
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                            if (out_acc > *threshLoc + up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out_acc += (1 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else if (out_acc < *threshLoc - up_thresh || (bitw == in_bit - 1)) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out_acc += (0 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }
                        if (output_bit == out_bit) {
                            break;
                        }
                    }
                    threshLoc++;
                    offsets++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
    }

    // Left 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < pad; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {

                    out = 0;
                    out_acc = 0;
                    output_bit = 0;
                    weight_temp = pWgt;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        outTemp = 0;
                        pWgt = weight_temp;
                        xyCount = 0;
                        // K-Y dim
                        for (uint16_t ky = yStart; ky < yEnd; ky++) {
                            // Move the input pointer to the first non-padded activation block
                            pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff + bitw;
                            // K-X dim
                            for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                xyCount++;
                                // Z dim
                                for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                    // XNOR multiplication
                                    xnorTemp = ~(*pIn ^ *pWgt++);
                                    outTemp += popcount(xnorTemp);
                                    pIn += in_bit;
                                }// Z dim
                            } // K-X dim
                            // Move the weight poitner to the next row
                            pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                        } // K-Y dim
                        outTemp = outTemp - (xyCount * kdpt - outTemp);
                        // Get the int full precision value 
                        out = (outTemp << (in_bit - bitw - 1));
                        // Quantization
                        int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                        int temp = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp |= (1 << i);
                        }
                        temp = (temp & out) << (16 - in_bit);
                        out_acc += out_temp + temp;
                        int up_thresh = 0;
                        int temp_thresh = 0;
                        for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                            temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                        }
                        int temp_fixed = temp_thresh >> (in_bit + 2) << (16);/// pow(2, in_bit);
                        int temp_pack = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp_pack |= (1 << i);
                        }
                        temp_pack = (temp_pack & temp_thresh) << (16 - in_bit);
                        up_thresh = temp_fixed + temp_pack;
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                            if (out_acc > *threshLoc + up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out_acc += (1 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else if (out_acc < *threshLoc - up_thresh || (bitw == in_bit - 1)) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out_acc += (0 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }
                        if (output_bit == out_bit) {
                            break;
                        }
                    }
                    threshLoc++;
                    offsets++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt;
    }

    // Right 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt + (wdth - kwdt + pad + 1) * knum * out_bit / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = wdth - kwdt + pad + 1; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            threshLoc = thresh;
            signs = sign;
            offsets = offset;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0, sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {

                    out = 0;
                    out_acc = 0;
                    output_bit = 0;
                    weight_temp = pWgt;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        outTemp = 0;
                        pWgt = weight_temp;
                        xyCount = 0;
                        // K-Y dim
                        for (uint16_t ky = yStart; ky < yEnd; ky++) {
                            // Move the input pointer to the first non-padded activation block
                            pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff + bitw;
                            // K-X dim
                            for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                xyCount++;
                                // Z dim
                                for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                    // XNOR multiplication
                                    xnorTemp = ~(*pIn ^ *pWgt++);
                                    outTemp += popcount(xnorTemp);
                                    pIn += in_bit;
                                }// Z dim
                            } // K-X dim
                              // Move the weight poitner to the next row
                            pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                        } // K-Y dim
                        outTemp = outTemp - (xyCount * kdpt - outTemp);
                        // Get the int full precision value 
                        out = (outTemp << (in_bit - bitw - 1));
                        // Quantization
                        int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
                        int temp = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp |= (1 << i);
                        }
                        temp = (temp & out) << (16 - in_bit);
                        out_acc += out_temp + temp;
                        int up_thresh = 0;
                        int temp_thresh = 0;
                        for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                            temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                        }
                        int temp_fixed = temp_thresh >> (in_bit + 2) << (16);/// pow(2, in_bit);
                        int temp_pack = 0;
                        for (int i = 0; i != in_bit; i++) {
                            temp_pack |= (1 << i);
                        }
                        temp_pack = (temp_pack & temp_thresh) << (16 - in_bit);
                        up_thresh = temp_fixed + temp_pack;
                        for (uint8_t bito = output_bit; bito != out_bit; bito++) {
                            if (out_acc > *threshLoc + up_thresh) {
                                pckTemp[bito] |= (1 << (pckWdt - ks - 1));
                                out_acc += (1 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else if (out_acc < *threshLoc - up_thresh || (bitw == in_bit - 1)) {
                                pckTemp[bito] |= (0 << (pckWdt - ks - 1));
                                out_acc += (0 ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? ((*offsets) >> (bito + 1)) : -((*offsets) >> (bito + 1)));
                                output_bit++;
                            }
                            else {
                                break;
                            }
                        }
                        if (output_bit == out_bit) {
                            break;
                        }
                    }
                    threshLoc++;
                    offsets++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
                //pckTemp = ~(pckTemp ^ *signs++);
                for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                    *pRes++ = ~(pckTemp[bitw] ^ (*signs));
                }
                signs++;
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum * out_bit / pckWdt + (wdth - kwdt + pad + 1) * knum * out_bit / pckWdt;
    }
    //printf("%d\n", exit_time);
}

void CnBnPdXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad,bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta, uint8_t in_bit, uint8_t out_bit) {
    // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    pckDtype pckTemp[out_bit];
    int out = 0;
    uint16_t  yCoeff = wdth * dpth*in_bit / pckWdt;
    uint16_t  xCoeff = dpth *in_bit/ pckWdt;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    // Moving kernel pointer
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;
    uint16_t  cntCoeff = khgt * kwdt * kdpt;
    // Starting indices for padding
    uint16_t  xStart, yStart = 0;
    // Ending indices for padding
    uint16_t  xEnd, yEnd = 0;

    // Divide the input into 5 regions - top, bottom, left, right, middle 
    // Middle has no padding

    // Middle - no padding
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1); y++) {
        // X dim
        // Set the output pointer
        // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
        // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
        // Offset to this row pad*knum/pckWdt
        pRes = pOut + (pad + y) * (hght - khgt + 2 * pad + 1) * knum / pckWdt + pad * knum / pckWdt;
        for (uint16_t x = 0; x < (wdth - kwdt + 1); x++) {
            // Outer loop - kernels
            pWgt = pKrn;
            //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    pIn = pAct + y * yCoeff + x * xCoeff;
                    memset(outTemp, 0,sizeof(outTemp));
                    out = 0;
                    // K-Y dim
                    for (uint16_t ky = 0; ky < khgt; ky++) {
                        // K-X dim
                        for (uint16_t kx = 0; kx < kwdt * dpth / pckWdt; kx++) {
                            for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                // XNOR multiplication
                                xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                outTemp[bitw] += popcount(xnorTemp[bitw]);
                            }
                            pWgt++;
                        } // K-X dim
                        // Move the activation pointer one row down
                        pIn += (wdth - kwdt) * dpth *in_bit/ pckWdt;
                    } // K-Y dim
                    // We've only counted ones, but we want a difference between +1s and -1s 
                    // so we need to adjust the result
                    // Below is shorter for
                    // outTemp = outTemp - (2*cntCoeff - outTemp);
                    // outTemp = outTemp >= 0;
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (cntCoeff - outTemp[bitw]);
                        // Get the int full precision value 
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Batch normalize/ binarize
                    float out_temp = out/ (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                    // Shift based on current kernel slice
                }
            }
        }
    }

    // Top
    pRes = pOut;
    // Y dim
    for (uint16_t y = 0; y < pad; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    memset(outTemp, 0,sizeof(outTemp));
                    out = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                    // XNOR multiplication
                                    xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                    outTemp[bitw] += popcount(xnorTemp[bitw]);
                                }
                                pWgt++;
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                        // Get the int full precision value 
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Batch normalize/ binarize
                    float out_temp = out / (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += yStart * kwdt * kdpt / pckWdt;
                }
            }
        }
    }

    // Bottom 
    // Move the ouput pointer
    pRes = pOut + (hght - khgt + pad + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = hght - khgt + pad + 1; y < hght - khgt + 2 * pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    memset(outTemp,0,sizeof(outTemp));
                    out = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                    // XNOR multiplication
                                    xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                    outTemp[bitw] += popcount(xnorTemp[bitw]);
                                }
                                pWgt++;
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                        // Get the int full precision value 
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Batch normalize/ binarize
                    float out_temp = out / (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
    }

    // Left 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = 0; x < pad; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp,0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    memset(outTemp, 0,sizeof(outTemp));
                    out = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                    // XNOR multiplication
                                    xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                    outTemp[bitw] += popcount(xnorTemp[bitw]);
                                }
                                pWgt++;
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                        // Get the int full precision value 
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Batch normalize/ binarize
                    float out_temp = out / (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt;
    }

    // Right 
    pRes = pOut + pad * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + (wdth - kwdt + pad + 1) * knum / pckWdt;
    // Y dim
    for (uint16_t y = pad; y < hght - khgt + pad + 1; y++) {
        // Account for padding - skip padded values
        if (y < pad) { yStart = pad - y; }
        else { yStart = 0; }
        if (y > hght - khgt + pad) { yEnd = hght - (y - pad); }
        else { yEnd = khgt; }
        // X dim
        for (uint16_t x = wdth - kwdt + pad + 1; x < wdth - kwdt + 2 * pad + 1; x++) {
            // Account for padding - skip padded values
            if (x < pad) { xStart = pad - x; }
            else { xStart = 0; }
            if (x > wdth - kwdt + pad) { xEnd = wdth - (x - pad); }
            else { xEnd = kwdt; }
            // Move the wieight pointer to the fisrt useful (non-padded) weight block
            pWgt = pKrn + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp, 0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    memset(outTemp, 0,sizeof(outTemp));
                    out = 0;
                    xyCount = 0;
                    // K-Y dim
                    for (uint16_t ky = yStart; ky < yEnd; ky++) {
                        // Move the input pointer to the first non-padded activation block
                        pIn = pAct + (y + ky - pad) * yCoeff + (x + xStart - pad) * xCoeff;
                        // K-X dim
                        for (uint16_t kx = xStart; kx < xEnd; kx++) {
                            xyCount++;
                            // Z dim
                            for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                    // XNOR multiplication
                                    xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                    outTemp[bitw] += popcount(xnorTemp[bitw]);
                                }
                                pWgt++;
                            } // Z dim
                        } // K-X dim
                        // Move the weight poitner to the next row
                        pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                    } // K-Y dim
                    // Adjust the output value
                    for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                        // Adjust the output value
                        outTemp[bitw] = outTemp[bitw] - (xyCount * kdpt - outTemp[bitw]);
                        // Get the int full precision value 
                        out += (outTemp[bitw] << (in_bit - bitw - 1));
                    }
                    // Batch normalize/ binarize
                    float out_temp = out / (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                    // Shift the weight pointer to the next kernel
                    pWgt += (khgt - yEnd + yStart) * kwdt * kdpt / pckWdt;
                }
            }
        }
        pRes = pOut + (y + 1) * (wdth - kwdt + 2 * pad + 1) * knum / pckWdt + (wdth - kwdt + pad + 1) * knum / pckWdt;
    }
}


#ifdef NEON
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, NEON
 * Outer loop: XY, Pad: yes, Pool: no BatchNorm: yes, SIMD: NEON (128) 
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint16_t  yCoeff  = wdth*dpth/pckWdt;
   uint16_t  xCoeff  = dpth/pckWdt;
   uint16_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint16_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint16_t  kxCoeff = kdpt/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Output X/Y dimensions
   uint16_t outYDim = hght-khgt+2*pad+1;
   uint16_t outXDim = wdth-kwdt+2*pad+1;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   uint16_t  cntCoeff = khgt*kwdt*kdpt/2;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;

   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding

   // Middle - no padding
   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1); y++) {
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + (pad+y)*(hght-khgt+2*pad+1)*knum/pckWdt + pad*knum/pckWdt;
      for (uint16_t x = 0; x < (wdth-kwdt+1); x++) {
         // Outer loop - kernels
         pWgt = pKrn;   
         threshLoc = thresh;
         signs = sign;
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               pIn = pAct + y*yCoeff + x*xCoeff;
               vecOut[0] = 0;
               vecOut[1] = 0;
               // K-Y dim
               for (uint16_t ky = 0; ky < khgt; ky++) {
                  // K-X dim
                  for (uint16_t kx = 0; kx < kwdt*(dpth/128)/pckWdt; kx++) {
                     // Load values
                     vecAct = vld1q_s32(*pIn);
                     vecWgt = vld1q_s32(*pWgt);
                     pIn += 4;
                     pWgt += 4;
                     // XNOR
                     vecAct = veorq_s32(vecAct, vecWgt);
                     vecAct = vmvnq_s32(vecAct);
                     // popcount
                     // popcount only works on 8-bit vectors, so needs some casting
                     // Not a problem here because those are binary vectors not values
                     vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                     // Now we need to do addition
                     // 16x8b reduce to 8x16b
                     vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                     // 8x16b to 4x32b
                     vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                     // 4x32b to a two values
                     vecOut += vpaddlq_s32(vecAct);
                  } // K-X dim
                  // Move the activation pointer one row down
                  pIn += (wdth-kwdt)*dpth/pckWdt;
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // We've only counted ones, but we want a difference between +1s and -1s 
               // so we need to adjust the result
               // Below is shorter for
               // outTemp = outTemp - (2*cntCoeff - outTemp);
               // outTemp = outTemp >= 0;
               outTemp = outTemp - cntCoeff;
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }

   // Top
   pRes = pOut;
   // Y dim
   for (uint16_t y = 0; y < pad; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
                           // popcount
                           // popcount only works on 8-bit vectors, so needs some casting
                           // Not a problem here because those are binary vectors not values
                           vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                           // Now we need to do addition
                           // 16x8b reduce to 8x16b
                           vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                           // 8x16b to 4x32b
                           vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                           // 4x32b to a two values
                           vecOut += vpaddlq_s32(vecAct);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += yStart*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + (hght-khgt+pad+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = hght-khgt+pad+1; y < hght-khgt+2*pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
                           // popcount
                           // popcount only works on 8-bit vectors, so needs some casting
                           // Not a problem here because those are binary vectors not values
                           vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                           // Now we need to do addition
                           // 16x8b reduce to 8x16b
                           vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                           // 8x16b to 4x32b
                           vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                           // 4x32b to a two values
                           vecOut += vpaddlq_s32(vecAct);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
   }
  
   // Left 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < pad; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
                           // popcount
                           // popcount only works on 8-bit vectors, so needs some casting
                           // Not a problem here because those are binary vectors not values
                           vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                           // Now we need to do addition
                           // 16x8b reduce to 8x16b
                           vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                           // 8x16b to 4x32b
                           vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                           // 4x32b to a two values
                           vecOut += vpaddlq_s32(vecAct);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt;
   }

   // Right 
   pRes = pOut + pad*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   // Y dim
   for (uint16_t y = pad; y < hght-khgt+pad+1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; }
      if (y > hght-khgt+pad) { yEnd = hght - (y-pad); } else { yEnd = khgt; }
      // X dim
      for (uint16_t x = wdth-kwdt+pad+1; x < wdth-kwdt+2*pad+1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; }
         if (x > wdth-kwdt+pad) { xEnd = wdth - (x-pad); } else { xEnd = kwdt; }
         // Move the wieight pointer to the fisrt useful (non-padded) weight block
         pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;   
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               vecOut[0] = 0;
               vecOut[1] = 0;
               xyCount = 0;
               // K-Y dim
               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                  // Move the input pointer to the first non-padded activation block
                  pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                  // K-X dim
                  for (uint16_t kx = xStart; kx < xEnd; kx++) {
                        xyCount++;
                        // Z dim
                        for (uint16_t z = 0; z < (dpth/128)/pckWdt; z++) {
                           // Load values
                           vecAct = vld1q_s32(*pIn);
                           vecWgt = vld1q_s32(*pWgt);
                           pIn += 4;
                           pWgt += 4;
                           // XNOR
                           vecAct = veorq_s32(vecAct, vecWgt);
                           vecAct = vmvnq_s32(vecAct);
                           // popcount
                           // popcount only works on 8-bit vectors, so needs some casting
                           // Not a problem here because those are binary vectors not values
                           vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                           // Now we need to do addition
                           // 16x8b reduce to 8x16b
                           vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                           // 8x16b to 4x32b
                           vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                           // 4x32b to a two values
                           vecOut += vpaddlq_s32(vecAct);
                        } // Z dim
                  } // K-X dim
                  // Move the weight poitner to the next row
                  pWgt += (kwdt-xEnd+xStart)*kdpt/pckWdt; 
               } // K-Y dim
               outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
               // Adjust the output value
               outTemp = outTemp - (xyCount*kdpt - outTemp);
               // Batch normalize/ binarize
               outTemp = (bnPrec) outTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= outTemp << (pckWdt-1-ks);
               // Shift the weight pointer to the next kernel
               pWgt += (khgt-yEnd+yStart)*kwdt*kdpt/pckWdt;
            }
            pckTemp = ~(pckTemp ^ *signs++);
            *pRes++ = pckTemp;
         }
      }
      pRes = pOut + (y+1)*(wdth-kwdt+2*pad+1)*knum/pckWdt + (wdth-kwdt+pad+1)*knum/pckWdt;
   }
}
#endif

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, pooling
 * Outer loop: XY, Pad: no, Pool: yes BatchNorm: yes, SIMD: no  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPlXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pool, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit){

   // Temporary variables
   pckDtype xnorTemp;
   int32_t  outTemp;
   int out = 0;
   // For maxpooling
   int  maxTemp=0;
   //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
   pckDtype pckTemp[out_bit];
   uint16_t  yCoeff  = wdth*dpth*in_bit/pckWdt;
   uint16_t  xCoeff  = dpth*in_bit/pckWdt;
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   pckDtype   *threshLoc = thresh;
   pckDtype* offsets = offset;
   uint16_t  cntCoeff = khgt * kwdt * kdpt;

   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+1)/pool; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+1)/pool; x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         offsets = offset;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            memset(pckTemp,0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = -cntCoeff;
               for (uint16_t yy = 0; yy < pool; yy++) {
                  for (uint16_t xx = 0; xx < pool; xx++) {                     
                     out = 0;
                     for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                         outTemp=0;
                         pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt;
                         pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff + bitw;
                         // K-Y dim
                         for (uint16_t ky = 0; ky < khgt; ky++) {
                             // K-X dim
                             for (uint16_t kx = 0; kx < kwdt; kx++) {
                                 // Z dim
                                 for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                     // XNOR multiplication
                                     xnorTemp = ~(*pIn ^ *pWgt++);
                                     outTemp += popcount(xnorTemp);
                                     pIn += in_bit;
                                 }// Z dim
                             } // K-X dim
                             pIn += (wdth - kwdt) * dpth * in_bit / pckWdt;
                         }// K-Y dim
                         outTemp = outTemp - (cntCoeff - outTemp);
                         out += (outTemp << (in_bit - bitw - 1));
                         int temp_thresh = 0;
                         for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                             temp_thresh += (cntCoeff << (in_bit - bitt - 1));
                         }
                         if (out + temp_thresh < maxTemp) {
                             break;
                         }
                     } 
                     // Maxpool
                     if (out > maxTemp) {
                         maxTemp = out;
                         int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                         int temp = 0;
                         for (int i = 0; i != in_bit; i++) {
                             temp |= (1 << i);
                         }
                         temp = (temp & maxTemp) << (16 - in_bit);
                         out_temp += temp;
                         int exit_condition = 0;
                         for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                             int temp = out_temp > *threshLoc;
                             // Shift 
                             pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                             exit_condition += temp;
                             //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                             out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                         }
                         if (exit_condition == out_bit) {
                             goto next;
                         }
                     }
                  } // X-MP
               } // Y-MP
               // Batch normalize/ binarize
               //goto end;
               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
               int temp = 0;
               for (int i = 0; i != in_bit; i++) {
                   temp |= (1 << i);
               }
               temp = temp & maxTemp;
               out_temp += temp;
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = out_temp > *threshLoc;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
               }
               next: threshLoc++;
               offsets++;
            }
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
   }
}

void CnBnPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, 
    const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta, uint8_t in_bit, uint8_t out_bit) {
    // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    // For maxpooling
    int32_t  maxTemp = 0;
    int out = 0;
    //int32_t  *outTemp = malloc(pool*pool*sizeof(int32_t));
    pckDtype pckTemp[out_bit];
    uint16_t  yCoeff = wdth * dpth *in_bit/ pckWdt;
    uint16_t  xCoeff = dpth *in_bit/ pckWdt;
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;
    
    // Y dim
    for (uint16_t y = 0; y < (hght - khgt + 1) / pool; y++) {
        // X dim
        for (uint16_t x = 0; x < (wdth - kwdt + 1) / pool; x++) {
            // Outer loop - kernels
            for (uint16_t k = 0; k < knum / pckWdt; k++) {
                // Packed slices
                memset(pckTemp,0,sizeof(pckTemp));
                for (uint16_t ks = 0; ks < pckWdt; ks++) {
                    // Mpool patches
                    maxTemp = -(khgt * kwdt * kdpt);
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            memset(outTemp,0,sizeof(outTemp));
                            out = 0;
                            pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt;
                            pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff;
                            // K-Y dim
                            for (uint16_t ky = 0; ky < khgt; ky++) {
                                // K-X dim
                                for (uint16_t kx = 0; kx < kwdt; kx++) {
                                    // Z dim
                                    for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                        for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                            // XNOR multiplication
                                            xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                            outTemp[bitw] += popcount(xnorTemp[bitw]);
                                        }
                                        pWgt++;
                                    } // Z dim
                                } // K-X dim
                                pIn += (wdth - kwdt) * dpth *in_bit/ pckWdt;
                            } // K-Y dim
                            // Adjust the output value
                            for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                // Adjust the output value
                                outTemp[bitw] = outTemp[bitw] - (khgt * kwdt * kdpt - outTemp[bitw]);
                                // Get the int full precision value 
                                out += (outTemp[bitw] << (in_bit - bitw - 1));
                            }
                            // Maxpool
                            if (out > maxTemp) { maxTemp = out; }
                        } // X-MP
                    } // Y-MP
                    // Batch normalize/ binarize
                    //goto end;
                    float out_temp = maxTemp / (float)(1 << (in_bit));// pow(2, in_bit);
                    *pRes++ = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                }
            }
        }
    }
}

/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling
 * Outer loop: XY, Pad: yes, Pool: yes BatchNorm: yes, SIMD: no  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPlXnor(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype* __restrict pOut, const uint8_t pad, const uint8_t pool, pckDtype* __restrict thresh, pckDtype* sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit){

   // Temporary variables
   pckDtype xnorTemp;
   int32_t  outTemp;
   pckDtype pckTemp[out_bit];
   int out = 0;
   uint16_t  yCoeff  = wdth*dpth*in_bit/pckWdt;
   uint16_t  xCoeff  = dpth*in_bit/pckWdt;
   uint16_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint16_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint16_t  kxCoeff = kdpt/pckWdt;
   // XY count for padding adjustment
   uint8_t  xyCount = 0;
   // Moving kernel pointer
   pckDtype *pWgt = pKrn;
   pckDtype *pIn  = pAct;
   pckDtype *pRes = pOut;
   pckDtype  *signs = sign;
   pckDtype   *threshLoc = thresh;
   pckDtype* offsets = offset;
   uint16_t  cntCoeff = khgt*kwdt*kdpt;
   // Starting indices for padding
   uint16_t  xStart, yStart = 0;
   // Ending indices for padding
   uint16_t  xEnd, yEnd = 0;
   // For maxpooling
   int  maxTemp = 0;
   int16_t  oHgt = (hght-khgt+2*pad+1)/pool;
   int16_t  oWdt = (wdth-kwdt+2*pad+1)/pool;
   int16_t  knCoeff = knum/pckWdt;
   int16_t  pInStrd = (wdth-kwdt)*kxCoeff*in_bit;
   //int one = 0;
   //int two = 0;
   //int one_time = 0;
   //int two_time = 0;
   // Divide the input into 5 regions - top, bottom, left, right, middle 
   // Middle has no padding
   //// Top
   pRes = pOut;
   // Y dim
   // We need to make sure there's enough lines to do pooling
   //for (uint16_t y = 0; y < pad; y++) {
   for (uint16_t y = 0; y < (pad + pool - 1) / pool; y++) {
       // X dim
       for (uint16_t x = 0; x < oWdt; x++) {
           // Restart kernel bn pointer
           threshLoc = thresh;
           signs = sign;
           offsets = offset;
           // Outer loop - kernels
           for (uint16_t k = 0; k < knCoeff; k++) {
               // Packed slices
               memset(pckTemp, 0, sizeof(pckTemp));
               for (uint16_t ks = 0; ks < pckWdt; ks++) {
                   // Mpool patches
                   maxTemp = INT_MIN;   
                   for (uint16_t yy = 0; yy < pool; yy++) {
                       // Account for padding - skip padded values
                       if ((y * pool + yy) < pad) { yStart = pad - (y * pool + yy); }
                       else { yStart = 0; }
                       if ((y * pool + yy) > hght - khgt + pad) { yEnd = hght - ((y * pool + yy) - pad); }
                       else { yEnd = khgt; }
                       for (uint16_t xx = 0; xx < pool; xx++) {
                           // Account for padding - skip padded values
                           if ((x * pool + xx) < pad) { xStart = pad - (x * pool + xx); }
                           else { xStart = 0; }
                           if ((x * pool + xx) > wdth - kwdt + pad) { xEnd = wdth - ((x * pool + xx) - pad); }
                           else { xEnd = kwdt; }
                           out = 0;
                           for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                               outTemp = 0;
                               // Move the wieight pointer to the fisrt useful (non-padded) weight block
                               //pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                               pWgt = pKrn + (k * pckWdt + ks) * kCoeff + yStart * kyCoeff + xStart * kxCoeff;
                               xyCount = 0;
                               // K-Y dim
                               for (uint16_t ky = yStart; ky < yEnd; ky++) {
                                   // Move the input pointer to the first non-padded activation block
                                   pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                                   // K-X dim
                                   for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                       xyCount++;
                                       // Z dim
                                       for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                           // XNOR multiplication
                                           xnorTemp = ~(*pIn ^ *pWgt++);
                                           outTemp += popcount(xnorTemp);
                                           pIn += in_bit;
                                       }// Z dim
                                   } // K-X dim
                                   // Move the weight poitner to the next row
                                   pWgt += (kwdt - xEnd + xStart) * kxCoeff;
                               }// K-Y dim
                               outTemp = outTemp - (xyCount * kdpt - outTemp);
                               out += (outTemp << (in_bit - bitw - 1));
                               int temp_thresh = 0;
                               for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                                   temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                               }
                               if (out + temp_thresh < maxTemp) {
                                   //if (bitw < in_bit - 1)exit_time++;
                                   break;
                               }
                           }                           
                           //printf("%d, ",outTemp);
                           // Maxpool
                           if (out > maxTemp) { 
                               //two++;
                               maxTemp = out; 
                               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                               int temp = 0;
                               for (int i = 0; i != in_bit; i++) {
                                   temp |= (1 << i);
                               }
                               temp = (temp & maxTemp) << (16 - in_bit);
                               out_temp += temp;
                               int exit_condition = 0;
                               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                                   int temp = out_temp > *threshLoc;
                                   // Shift 
                                   if (temp == 1) {
                                        exit_condition++;
                                        pckTemp[bitw] |= (1 << (pckWdt - ks - 1));
                                        //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                                        out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                                   }
                                   else {
                                       break;
                                   }
                               }
                               if (exit_condition == out_bit) {
                                   //two_time += ((pool * pool - xx - yy * pool - 1) * in_bit != 0);
                                   //printf("et pooling top: window wise skip: %d, y: %d, x: %d, k: %d, ks: %d\n", (pool * pool - xx - yy * pool - 1)*in_bit, y, x, k, ks);
                                   goto next_middle;
                               }
                           }
                       }
                   }
                   // Binarize
                   int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                   int temp = 0;
                   for (int i = 0; i != in_bit; i++) {
                       temp |= (1 << i);
                   }
                   temp = (temp & maxTemp) << (16 - in_bit);
                   out_temp += temp;
                   for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                       int temp = out_temp > *threshLoc;
                       // Shift 
                       pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                       //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                       out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                   }
                   next_middle: threshLoc++;
                   offsets++;
                   // Shift the weight pointer to the next kernel
               }
               //pckTemp = ~(pckTemp ^ *signs++);
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   *pRes++ = ~(pckTemp[bitw]^(*signs));
               }
               signs++;
           }
       }
   }
   // Middle - no padding
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      //printf("Y: %d\n", y);
      // X dim
      // Set the output pointer
      // First n padded rows pad*(hght-khgt+2*pad+1)*knum/pckWdt
      // Already completed rows y*(hght-khgt+2*pad+1)*knum/pckWdt
      // Offset to this row pad*knum/pckWdt
      pRes = pOut + y*oHgt*knCoeff*out_bit + ((pad+pool-1)/pool)*knCoeff*out_bit;
      //printf("%d %d %d\n", pOut, pRes, (y)*((hght-khgt+2*pad+1)/pool)*knum/pckWdt + ((pad+pool-1)/pool)*knum/pckWdt);
      for (uint16_t x = ((pad+pool-1)/pool); x <= oWdt - 2*((pad+pool-1)/pool); x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         offsets = offset;
         //printf("X: %d\n", x);
         // Outer loop - kernels
         pWgt = pKrn;   
         //pRes = pOut + (y*(wdth-kwdt+1)+x)*knum/pckWdt;
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = INT_MIN;
               for (uint16_t yy = 0; yy < pool; yy++) {
                   for (uint16_t xx = 0; xx < pool; xx++) {
                       out = 0;
                       for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                           outTemp = 0;
                           pIn = pAct + (y * pool + yy - pad) * yCoeff + (x * pool + xx - pad) * xCoeff + bitw;
                           pWgt = pKrn + (k * pckWdt + ks) * kCoeff;
                           // K-Y dim
                           for (uint16_t ky = 0; ky < khgt; ky++) {
                               // K-X dim
                               for (uint16_t kx = 0; kx < kyCoeff; kx++) {
                                   // XNOR multiplication
                                   xnorTemp = ~(*pIn ^ *pWgt++);
                                   outTemp += popcount(xnorTemp);
                                   pIn += in_bit;
                               }// K-X dim
                                // Move the activation pointer one row down
                               pIn += pInStrd;
                           } // K-Y dim
                           //one++;
                           outTemp = outTemp - (cntCoeff - outTemp);
                           out += (outTemp << (in_bit - bitw - 1));
                           int temp_thresh = 0;
                           for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                               temp_thresh += (cntCoeff << (in_bit - bitt - 1));
                           }
                           if (out + temp_thresh < maxTemp) {
                               //if (bitw < in_bit - 1)exit_time++;
                               break;
                           }
                       }
                       //printf("OT: %d\n", outTemp);
                       // Maxpool
                       if (out > maxTemp) {
                           //two++;
                           maxTemp = out;
                           int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                           int temp = 0;
                           for (int i = 0; i != in_bit; i++) {
                               temp |= (1 << i);
                           }
                           temp = (temp & maxTemp) << (16 - in_bit);
                           out_temp += temp;
                           int exit_condition = 0;
                           for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                               int temp = out_temp > *threshLoc;
                               // Shift 
                               if (temp == 1) {
                                   exit_condition++;
                                   pckTemp[bitw] |= (1 << (pckWdt - ks - 1));
                                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                               }
                               else {
                                   break;
                               }
                           }
                           if (exit_condition == out_bit) {
                               //two_time += ((pool * pool - xx - yy * pool - 1) * in_bit != 0);
                               //printf("et pooling middle: window wise skip: %d, y: %d, x: %d, k: %d, ks: %d\n", (pool* pool - xx - yy * pool - 1)* in_bit, y, x, k, ks);
                               goto next_top;
                           }
                       }
                       // Shift based on current kernel slice
                   } // X-MP
               } // Y-MP
               // Binarize
               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
               int temp = 0;
               for (int i = 0; i != in_bit; i++) {
                   temp |= (1 << i);
               }
               temp = (temp & maxTemp) << (16 - in_bit);
               out_temp += temp;
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = out_temp > *threshLoc;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
               }
               // Batch normalize/ binarize
               next_top: threshLoc++;
               offsets++;
            }
            //printf("%d\n", pRes);
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
   }
   
   // Bottom 
   // Move the ouput pointer
   pRes = pOut + ((hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool))*((wdth-kwdt+2*pad+1)/pool)*knum*out_bit/pckWdt;
   // Y dim
   //for (uint16_t y = hght-khgt+((pad+pool-1)/pool)+1; y < hght-khgt+2*((pad+pool-1)/pool)+1; y++) {
   for (uint16_t y = (hght-khgt+2*pad)/pool + 1 - ((pad+pool-1)/pool); y < (hght-khgt+2*pad)/pool + 1; y++) {
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         offsets = offset;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = (INT_MIN);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                      // Account for padding - skip padded values
                      if ((x * pool + xx) < pad) { xStart = pad - (x * pool + xx); }
                      else { xStart = 0; }
                      if ((x * pool + xx) > wdth - kwdt + pad) { xEnd = wdth - ((x * pool + xx) - pad); }
                      else { xEnd = kwdt; }
                      // Move the wieight pointer to the fisrt useful (non-padded) weight block
                      //pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;                       
                      out = 0;
                      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                          outTemp = 0;
                          pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt + yStart * kwdt * kdpt / pckWdt + xStart * kdpt / pckWdt;
                          xyCount = 0;
                          //printf("%d %d %d %d %d %d %d %d\n", y, yy, x, xx, yStart, yEnd, xStart, xEnd);
                          // K-Y dim
                          for (uint16_t ky = yStart; ky < yEnd; ky++) {
                              // Move the input pointer to the first non-padded activation block
                              //pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                              pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                              // K-X dim
                              for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                  xyCount++;
                                  // Z dim
                                  for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                      // XNOR multiplication
                                      xnorTemp = ~(*pIn ^ *pWgt++);
                                      outTemp += popcount(xnorTemp);
                                      pIn += in_bit;
                                  }// Z dim
                              } // K-X dim
                             // Move the weight poitner to the next row
                              pWgt += (kwdt - xEnd + xStart) * kdpt / pckWdt;
                          } // K-Y dim
                          //one++;
                          outTemp = outTemp - (xyCount * kdpt - outTemp);
                          out += (outTemp << (in_bit - bitw - 1));
                          int temp_thresh = 0;
                          for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                              temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                          }
                          if (out + temp_thresh < maxTemp) {
                              //if (bitw < in_bit - 1)exit_time++;
                              break;
                          }
                      }
                      // Maxpool
                      if (out > maxTemp) {
                          //two++;
                          maxTemp = out;
                          int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                          int temp = 0;
                          for (int i = 0; i != in_bit; i++) {
                              temp |= (1 << i);
                          }
                          temp = (temp & maxTemp) << (16 - in_bit);
                          out_temp += temp;
                          int exit_condition = 0;
                          for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                              int temp = out_temp > *threshLoc;
                              // Shift 
                              if (temp == 1) {
                                  exit_condition++;
                                  pckTemp[bitw] |= (1 << (pckWdt - ks - 1));
                                  //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                                  out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                              }
                              else {
                                  break;
                              }
                          }
                          if (exit_condition == out_bit) {
                              //two_time += ((pool * pool - xx - yy * pool - 1) * in_bit != 0);
                              //printf("et pooling bottom: window wise skip: %d, y: %d, x: %d, k: %d, ks: %d\n", (pool* pool - xx - yy * pool - 1)* in_bit, y, x, k, ks);
                              goto next_bot;
                          }
                      }
                  }
               }
               // Binarize
               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
               int temp = 0;
               for (int i = 0; i != in_bit; i++) {
                   temp |= (1 << i);
               }
               temp = (temp & maxTemp) << (16 - in_bit);
               out_temp += temp;
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = out_temp > *threshLoc;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
               }
               next_bot: threshLoc++;
               offsets++;
            }
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
   }
  
   //// Left 
   pRes = pOut + ((pad+pool-1)/pool)*(oWdt)*out_bit*knCoeff;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = 0; x < ((pad+pool-1)/pool); x++) {
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         offsets = offset;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = (INT_MIN);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                      // Account for padding - skip padded values
                      if ((x * pool + xx) < pad) { xStart = pad - (x * pool + xx); }
                      else { xStart = 0; }
                      if ((x * pool + xx) > wdth - kwdt + pad) { xEnd = wdth - ((x * pool + xx) - pad); }
                      else { xEnd = kwdt; }

                      out = 0;
                      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                          outTemp = 0;
                          xyCount = 0;
                          // Move the wieight pointer to the fisrt useful (non-padded) weight block
                          //pWgt = pKrn + (k*pckWdt + ks)*(khgt*kwdt*kdpt)/pckWdt;
                          pWgt = pKrn + (k * pckWdt + ks) * kCoeff + yStart * kyCoeff + xStart * kxCoeff;
                          // K-Y dim
                          for (uint16_t ky = yStart; ky < yEnd; ky++) {
                              // Move the input pointer to the first non-padded activation block
                              pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                              // K-X dim
                              for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                  xyCount++;
                                  // Z dim
                                  for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                      // XNOR multiplication
                                      xnorTemp = ~(*pIn ^ *pWgt++);
                                      outTemp += popcount(xnorTemp);
                                      pIn += in_bit;
                                  }// Z dim
                              } // K-X dim
                              // Move the weight poitner to the next row
                              pWgt += (kwdt - xEnd + xStart) * kxCoeff;
                          } // K-Y dim
                          //one++;
                          outTemp = outTemp - (xyCount * kdpt - outTemp);
                          out += (outTemp << (in_bit - bitw - 1));
                          int temp_thresh = 0;
                          for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                              temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                          }
                          if (out + temp_thresh < maxTemp) {
                              //if (bitw < in_bit - 1)exit_time++;
                              break;
                          }
                      }
                      // Maxpool
                      if (out > maxTemp) {
                          //two++;
                          maxTemp = out;
                          int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                          int temp = 0;
                          for (int i = 0; i != in_bit; i++) {
                              temp |= (1 << i);
                          }
                          temp = (temp & maxTemp) << (16 - in_bit);
                          out_temp += temp;
                          int exit_condition = 0;
                          for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                              int temp = out_temp > *threshLoc;
                              // Shift 
                              if (temp == 1) {
                                  exit_condition++;
                                  pckTemp[bitw] |= (1 << (pckWdt - ks - 1));
                                  //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                                  out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                              }
                              else {
                                  break;
                              }
                          }
                          if (exit_condition == out_bit) {
                              //two_time += ((pool * pool - xx - yy * pool - 1) * in_bit != 0);
                              //printf("et pooling left: window wise skip: %d, y: %d, x: %d, k: %d, ks: %d\n", (pool* pool - xx - yy * pool - 1)* in_bit, y, x, k, ks);
                              goto next_left;
                          }
                      }
                  }
               }
               // Binarize
               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
               int temp = 0;
               for (int i = 0; i != in_bit; i++) {
                   temp |= (1 << i);
               }
               temp = (temp & maxTemp) << (16 - in_bit);
               out_temp += temp;
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = out_temp > *threshLoc;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
               }
               next_left: threshLoc++;
               offsets++;
            }
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
      pRes = pOut + (y+1)*(oWdt)*knum*out_bit/pckWdt;
   }

   // Right 
   pRes = pOut + ((pad+pool-1)/pool)*(oWdt)*knum*out_bit/pckWdt + ((oWdt) - ((pad+pool-1)/pool))*knum*out_bit/pckWdt;
   // Y dim
   for (uint16_t y = ((pad+pool-1)/pool); y <= oHgt - 2*((pad+pool-1)/pool); y++) {
      // X dim
      for (uint16_t x = (wdth-kwdt+2*pad)/pool + 1 - ((pad+pool-1)/pool); x < (wdth-kwdt+2*pad)/pool + 1; x++) {
      // Restart kernel bn pointer
      threshLoc = thresh;
      signs = sign;
      offsets = offset;
      //for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
      //for (uint16_t x = (wdth-kwdt+2*pad+1)/pool; x < wdth-kwdt+2*pad+1; x++) {
         // Outer loop - kernels
         for (uint16_t k = 0; k<knCoeff; k++) {
            // Packed slices
            memset(pckTemp, 0,sizeof(pckTemp));
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               // Mpool patches
               maxTemp = (INT_MIN);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  // Account for padding - skip padded values
                  if ((y*pool+yy) < pad) { yStart = pad-(y*pool+yy); } else { yStart = 0; }
                  if ((y*pool+yy) > hght-khgt+pad) { yEnd = hght - ((y*pool+yy)-pad); } else { yEnd = khgt; }
                  for (uint16_t xx = 0; xx < pool; xx++) {
                      // Account for padding - skip padded values
                      if ((x * pool + xx) < pad) { xStart = pad - (x * pool + xx); }
                      else { xStart = 0; }
                      if ((x * pool + xx) > wdth - kwdt + pad) { xEnd = wdth - ((x * pool + xx) - pad); }
                      else { xEnd = kwdt; }
                      // Move the wieight pointer to the fisrt useful (non-padded) weight block
                      //pWgt = pKrn + yStart*kwdt*kdpt/pckWdt + xStart*kdpt/pckWdt;
                      out = 0;
                      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                          outTemp = 0;
                          pWgt = pKrn + (k * pckWdt + ks) * kCoeff + yStart * kyCoeff + xStart * kxCoeff;
                          xyCount = 0;
                          // K-Y dim
                          for (uint16_t ky = yStart; ky < yEnd; ky++) {
                              // Move the input pointer to the first non-padded activation block
                              //pIn = pAct + (y+ky-pad)*yCoeff + (x+xStart-pad)*xCoeff;
                              pIn = pAct + ((y * pool + yy) + ky - pad) * yCoeff + ((x * pool + xx) + xStart - pad) * xCoeff + bitw;
                              // K-X dim
                              for (uint16_t kx = xStart; kx < xEnd; kx++) {
                                  xyCount++;
                                  // Z dim
                                  for (uint16_t z = 0; z < dpth / pckWdt; z++) {

                                      // XNOR multiplication
                                      xnorTemp = ~(*pIn ^ *pWgt++);
                                      outTemp += popcount(xnorTemp);
                                      pIn += in_bit;
                                  }// Z dim
                              } // K-X dim
                              // Move the weight poitner to the next row
                              pWgt += (kwdt - xEnd + xStart) * kxCoeff;
                          } // K-Y dim
                          //one++;
                          outTemp = outTemp - (xyCount * kdpt - outTemp);
                          out += (outTemp << (in_bit - bitw - 1));
                          int temp_thresh = 0;
                          for (uint8_t bitt = bitw + 1; bitt != in_bit; bitt++) {
                              temp_thresh += ((xyCount * kdpt) << (in_bit - bitt - 1));
                          }
                          if (out + temp_thresh < maxTemp) {
                              //if (bitw < in_bit - 1)exit_time++;
                              break;
                          }
                      }
                      // Maxpool
                      if (out > maxTemp) {
                          //two++;
                          maxTemp = out;
                          int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
                          int temp = 0;
                          for (int i = 0; i != in_bit; i++) {
                              temp |= (1 << i);
                          }
                          temp = (temp & maxTemp) << (16 - in_bit);
                          out_temp += temp;
                          int exit_condition = 0;
                          for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                              int temp = out_temp > *threshLoc;
                              // Shift 
                              if (temp == 1) {
                                  exit_condition++;
                                  pckTemp[bitw] |= (1 << (pckWdt - ks - 1));
                                  //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                                  out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
                              }
                              else {
                                  break;
                              }
                          }
                          if (exit_condition == out_bit) {
                              //two_time += ((pool * pool - xx - yy * pool - 1) * in_bit != 0);
                              //printf("et pooling right: window wise skip: %d, y: %d, x: %d, k: %d, ks: %d\n", (pool* pool - xx - yy * pool - 1)* in_bit, y, x, k, ks);
                              goto next_right;
                          }
                      }
                  }
               }
               // Binarize
               int out_temp = maxTemp >> (in_bit) << (16);/// pow(2, in_bit);
               int temp = 0;
               for (int i = 0; i != in_bit; i++) {
                   temp |= (1 << i);
               }
               temp = (temp & maxTemp) << (16 - in_bit);
               out_temp += temp;
               for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                   int temp = out_temp > *threshLoc;
                   // Shift 
                   pckTemp[bitw] |= (temp << (pckWdt - ks - 1));
                   //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
                   out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offsets) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
               }
               next_right: threshLoc++;
               offsets++;
            }
            //pckTemp = ~(pckTemp ^ *signs++);
            for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
                *pRes++ = ~(pckTemp[bitw]^(*signs));
            }
            signs++;
         }
      }
      pRes = pOut + (y+1)*(oWdt)*knCoeff*out_bit+ ((oWdt) - ((pad+pool-1)/pool))*knCoeff*out_bit;
   }
   //printf("%d,%d, %d, %d\n", one, one_time,two,two_time);
}

void CnBnPdPlXnorNoBin(pckDtype* __restrict pAct, pckDtype* __restrict pKrn, const uint16_t dpth, 
    const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, 
    const uint16_t khgt, const uint16_t knum, bnDtype* __restrict pOut, const uint8_t pad, 
    const uint8_t pool, bnDtype* __restrict mean, bnDtype* __restrict var, bnDtype* __restrict gamma, bnDtype* __restrict beta, uint8_t in_bit, uint8_t out_bit) {
    // Temporary variables
    pckDtype xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    int out = 0;
    uint16_t  yCoeff = wdth * dpth *in_bit / pckWdt;
    uint16_t  xCoeff = dpth *in_bit / pckWdt;
    uint16_t  kCoeff = kdpt / pckWdt;
    uint16_t index = 0;
    // XY count for padding adjustment
    uint8_t  xyCount = 0;
    uint8_t  idxYY = 0;
    uint8_t  idxXX = 0;
    // For maxpooling
    int32_t  maxTemp = 0;
    pckDtype  signs = 0;
    pckDtype* pWgt = pKrn;
    pckDtype* pIn = pAct;
    bnDtype* pRes = pOut;

    // Outer loop - kernels
    for (uint16_t k = 0; k < knum / pckWdt; k++) {
        // Packed slices
        for (uint16_t ks = 0; ks < pckWdt; ks++) {
            // Y dim
            for (uint16_t y = 0; y < ((hght - khgt + 2 * pad + 1) / pool); y++) {
                // X dim
                for (uint16_t x = 0; x < ((wdth - kwdt + 2 * pad + 1) / pool); x++) {
                    maxTemp = -(khgt * kwdt * kdpt);
                    // Need to do that because we'll be oring into it 
                    for (uint16_t yy = 0; yy < pool; yy++) {
                        for (uint16_t xx = 0; xx < pool; xx++) {
                            memset(outTemp, 0,sizeof(outTemp));
                            xyCount = 0;
                            pWgt = pKrn + (k * pckWdt + ks) * (khgt * kwdt * kdpt) / pckWdt;
                            pIn = pAct + (y * pool + yy) * yCoeff + (x * pool + xx) * xCoeff;
                            // K-Y dim
                            for (uint16_t ky = 0; ky < khgt; ky++) {
                                // K-X dim
                                for (uint16_t kx = 0; kx < kwdt; kx++) {
                                    // Z dim
                                    for (uint16_t z = 0; z < dpth / pckWdt; z++) {
                                        for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                            // XNOR multiplication
                                            xnorTemp[bitw] = ~(*pIn++ ^ *pWgt);
                                            outTemp[bitw] += popcount(xnorTemp[bitw]);
                                        }
                                        pWgt++;
                                    } // Z dim
                                } // K-X dim
                                pIn += (wdth - kwdt) * dpth *in_bit/ pckWdt;
                            } // K-Y dim
                            // Adjust the output value
                            for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                                // Adjust the output value
                                outTemp[bitw] = outTemp[bitw] - (khgt * kwdt * kdpt - outTemp[bitw]);
                                // Get the int full precision value 
                                out += (outTemp[bitw] << (in_bit - bitw - 1));
                            }
                            // Maxpool
                            if (out > maxTemp) { maxTemp = out; }
                        }
                    }
                    // Binarize
                    //maxTemp = (bnPrec)maxTemp >= *thresh;
                    // Shift based on current kernel slice
                    //maxTemp = maxTemp << (pckWdt - 1 - ks);
                    // First time writing to a given word, make sure to clear it
                    // Write out
                    float out_temp = maxTemp / (float)(1 << (in_bit));// pow(2, in_bit);
                    pOut[y * (wdth - kwdt + 1) * knum / pckWdt + x * knum / pckWdt + k] = (float)*gamma++ * (((bnPrec)out_temp - *mean++) / (*var++)) + *beta++;
                }
            }
        }
    }

}

#ifdef NEON
/**
 * @details Dense binarized Convolutional (CN) layer with output binarization - XY outer loop, batch norm, padding, pooling, NEON
 * Outer loop: XY, Pad: yes, Pool: yes BatchNorm: yes, SIMD: NEON (128)  
 * 
 * @param[in] pAct - pointer to the packed activation vector (row-column-depth)
 * @param[in] pKrn - pointer to the packed weight matrix (kernel-row-column-depth flattened)
 * @param[in] dpth - input depth 
 * @param[in] wdth - input width 
 * @param[in] hght - input height
 * @param[in] kdpt - kernel depth 
 * @param[in] kwdt - kernel width 
 * @param[in] khgt - kernel height
 * @param[in] knum - number of kernels 
 * @param[out] pOut - pointer to the packed output vector (row-column-depth)
 * @param[in] pad  - padding size
 * @param[in] pool - pooling size
 * @param[in] thresh - pointer to batch normalization threshold 
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void CnBnPdPlXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pKrn, const uint16_t dpth, const uint16_t wdth, const uint16_t hght, const uint16_t kdpt, const uint16_t kwdt, const uint16_t khgt, const uint16_t knum, pckDtype * __restrict pOut, const uint8_t pad, const uint8_t pool, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // Temporary variables
   pckDtype xnorTmp = 0;
   int32_t  outTemp = 0;
   pckDtype pckTemp = 0;
   uint8_t  yCoeff  = wdth*dpth/pckWdt;
   uint8_t  xCoeff  = dpth/pckWdt;
   uint8_t  kCoeff  = khgt*kwdt*kdpt/pckWdt;
   uint8_t  kyCoeff = kwdt*kdpt/pckWdt;
   uint8_t  kxCoeff = kdpt/pckWdt;
   // Starting indices for padding
   uint8_t  xStart, yStart = 0;
   uint8_t  xxStart, yyStart = 0;
   // Ending indices for padding
   uint8_t  xEnd, yEnd = 0;
   uint8_t  xxEnd, yyEnd = 0;
   // XY count for padding adjustment
   uint8_t  gyCount = 0;
   // For maxpooling
   int32_t  maxTemp = 0;
   pckDtype  *signs = sign;
   bnDtype   *threshLoc = thresh;
   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;

   // Y dim
   for (uint16_t y = 0; y < (hght-khgt+2*pad)/pool +1; y++) {
      // Account for padding - skip padded values
      if (y < pad) { yStart = pad-y; } else { yStart = 0; yyStart = 0; }
      if (y*pool > hght-khgt+pad-pool) { yEnd = y*pool -(hght-khgt+pad-pool) + 1; } else { yEnd = khgt; yyEnd = khgt; }
      // X dim
      for (uint16_t x = 0; x < (wdth-kwdt+2*pad)/pool +1; x++) {
         // Account for padding - skip padded values
         if (x < pad) { xStart = pad-x; } else { xStart = 0; xxStart = 0; }
         if (x*pool > wdth-kwdt+pad-pool) { xEnd = x*pool - (wdth-kwdt+pad-pool) + 1; } else { xEnd = kwdt; xxEnd = kwdt; }
         // Restart kernel bn pointer
         threshLoc = thresh;
         signs = sign;
         // Outer loop - kernels
         for (uint16_t k = 0; k<knum/pckWdt; k++) {
            // Packed slices
            pckTemp = 0;
            for (uint16_t ks = 0; ks<pckWdt; ks++) {
               maxTemp = -(khgt*kwdt*kdpt);
               for (uint16_t yy = 0; yy < pool; yy++) {
                  if (yStart != 0) { yyStart = yStart-yy; };
                  if (y*pool+yEnd >= hght) { yyEnd = yEnd-yy; };
                  for (uint16_t xx = 0; xx < pool; xx++) {
                     if (xStart != 0) { xxStart = xStart-xx; };
                     if (x*pool+xEnd >= wdth) { xxEnd = xEnd-xx; };
                     xyCount = 0;
                     vecOut[0] = 0;
                     vecOut[1] = 0;
                     // K-Y dim
                     for (uint16_t ky = yyStart; ky < yyEnd; ky++) {
                        // K-X dim
                        for (uint16_t kx = xxStart; kx < xxEnd; kx++) {
                              xyCount++;
                              // Z dim
                              for (uint16_t z = 0; z < dpth/128; z++) {
                                 // Load values
                                 vecAct = vld1q_s32(pAct + (y*pool+yy+ky-pad)*yCoeff + (x*pool+xx+kx-pad)*xCoeff + z*128/pckWdt);
                                 vecWgt = vld1q_s32(pKrn + (k*pckWdt+ks)*kCoeff +  ky*kyCoeff + kx*kxCoeff+ z*128/pckWdt);
                                 // XNOR
                                 vecAct = veorq_s32(vecAct, vecWgt);
                                 vecAct = vmvnq_s32(vecAct);
                                 // popcount
                                 // popcount only works on 8-bit vectors, so needs some casting
                                 // Not a problem here because those are binary vectors not values
                                 vecAct = vreinterpretq_s32_s8(vcntq_s8(vreinterpretq_s8_s32(vecAct)));
                                 // Now we need to do addition
                                 // 16x8b reduce to 8x16b
                                 vecAct = vreinterpretq_s32_s16(vpaddlq_s8(vreinterpretq_s8_s32(vecAct)));
                                 // 8x16b to 4x32b
                                 vecAct = vpaddlq_s16(vreinterpretq_s16_s32(vecAct));
                                 // 4x32b to a two values
                                 vecOut += vpaddlq_s32(vecAct);
                              } // Z dim
                        } // K-X dim
                     } // K-Y dim
                     // Extract the output
                     outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
                     // Adjust the output value
                     outTemp = outTemp - (xyCount*kdpt - outTemp);
                     // Maxpool
                     if (outTemp > maxTemp) { maxTemp = outTemp;}
                  }
               }
               // Batch normalize/ binarize
               maxTemp = (bnPrec) maxTemp >= *threshLoc++;
               // Shift based on current kernel slice
               pckTemp |= maxTemp << (pckWdt-1-ks);
            }
            pckTemp = ~(pckTemp ^ *signs++);
            pOut[y*(((wdth-kwdt+2*pad)/pool)+1)*knum/pckWdt + x*knum/pckWdt + k] = pckTemp;
         }
      }
   }

}
#endif


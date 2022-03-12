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
 * \file      3pxnet_fc.c
 * \brief     3PXNet fully-connected layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#include "3pxnet_fc.h"

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, NEON support)
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, BN is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 * @return 0 - Success, 1 - Failure
 */
uint8_t Fc3pxnWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype* __restrict thresh, pckDtype* __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit){

   // Batch Norm present (thresh != NULL)
   if (thresh) {
      // Output or input not multiple of pack width - not supported atm
      if (numIn % pckWdt != 0 || numOut % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (numIn % (4*pckWdt) == 0) {
         FcBn3pxnNeonQ(pAct, pWgt, pInd, numIn, numOut, pOut, thresh, sign);
         return 0;
      }
      // Check for dual SIMD support
      else if (numIn % (2*pckWdt) == 0) {
         FcBn3pxnNeonQ(pAct, pWgt, pInd, numIn, numOut, pOut, thresh, sign);
         return 0;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         FcBn3pxnPtr(pAct, pWgt, pInd, numIn, numOut, pOut, thresh, sign, offset, in_bit, out_bit);
         return 0;
      }
   }
   // No Batch Norm (bnDtype == NULL)
   else {
      // Output or input not multiple of pack width - not supported atm
      if (numIn % pckWdt != 0 || numOut % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (numIn % (4*pckWdt) == 0) {
         Fc3pxnNeonQ(pAct, pWgt, pInd, numIn, numOut, pOut);
         return 0;
      }
      // Check for dual SIMD support
      else if (numIn % (2*pckWdt) == 0) {
         Fc3pxnNeon(pAct, pWgt, pInd, numIn, numOut, pOut);
         return 0;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         Fc3pxnPtr(pAct, pWgt, pInd, numIn, numOut, pOut, in_bit, out_bit);
         return 0;
      }
   }
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - general wrapper.
 * Selects the appropriate implementation (batch normalization, NEON support)
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] mean - batch norm mean (per output) (if NULL, batch norm is skipped)
 * @param[in] var - sqrt(var - epsilon) - corrected batch norm variance (per output)
 * @param[in] gamma - batch norm gamma (per output)
 * @param[in] beta - batch norm beta (per output)
 * @return 0 - Success, 1 - Failure
 */
uint8_t Fc3pxnNoBinWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta, uint8_t in_bit, uint8_t out_bit) {

   // Batch Norm present (mean != NULL)
   if (mean) {
      // Input not multiple of pack width - not supported atm
      if (numIn % pckWdt != 0 ) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (numIn % (4*pckWdt) == 0) {
         FcBn3pxnNeonQNoBin(pAct, pWgt, pInd, numIn, numOut, pOut, mean, var, gamma, beta);
         return 0;
      }
      // Check for dual SIMD support
      else if (numIn % (2*pckWdt) == 0) {
         FcBn3pxnNeonNoBin(pAct, pWgt, pInd, numIn, numOut, pOut, mean, var, gamma, beta);
         return 0;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         FcBn3pxnPtrNoBin(pAct, pWgt, pInd, numIn, numOut, pOut, mean, var, gamma, beta, in_bit, out_bit);
         return 0;
      }
   }
   // No Batch Norm (bnDtype == NULL)
   else {
      // Output or input not multiple of pack width - not supported atm
      if (numIn % pckWdt != 0) {
         return 1;
      }
      // NEON is currently implemented for 32-bit packs
#ifdef NEON
#ifdef PCK32
      // Check for quad SIMD support
      else if (numIn % (4*pckWdt) == 0) {
         Fc3pxnNeonQNoBin(pAct, pWgt, pInd, numIn, numOut, pOut);
         return 0;
      }
      // Check for dual SIMD support
      else if (numIn % (2*pckWdt) == 0) {
         Fc3pxnNeonNoBin(pAct, pWgt, pInd, numIn, numOut, pOut);
         return 0;
      }
#endif
#endif
      // Roll back to default implementation
      else {
         Fc3pxnPtrNoBin(pAct, pWgt, pInd, numIn, numOut, pOut, in_bit, out_bit);
         return 0;
      }
   }
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - pointer version
 * Indexing: pointer, batch norm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnPtr(pckDtype* __restrict pAct, pckDtype* __restrict pWgt, uint8_t* __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype* __restrict pOut, uint8_t in_bit, uint8_t out_bit){
   // Input counter
   uint16_t inCnt = numIn;
   // Output counter
   uint16_t outCnt = numOut;
   // temporary output value (before binarization)
   uint32_t xnorTmp[in_bit];
   int32_t  outTemp[in_bit];
   // Packing/shifting  index
   uint8_t packIdx = pckWdt-1;
   // For holding batches of output values 
   pckDtype pckTemp[out_bit];
   memset(pckTemp, 0, sizeof(int32_t) * out_bit);
   int out = 0;
   while (outCnt) {
       out = 0;
       memset(outTemp, 0, sizeof(outTemp));
      inCnt = numIn;
      while (inCnt) {
          // XNOR multiplication
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              xnorTmp[bitw] = ~((*(pckDtype*)(pAct + (*pInd)*in_bit+bitw)) ^ *pWgt);
              xnorTmp[bitw] = popcount(xnorTmp[bitw]);
              // Accumulation
              outTemp[bitw] += xnorTmp[bitw];
          }
          pInd++;
          pWgt++;
         // Decrement input counter
         inCnt -= pckWdt;
      }
      // Adjust value
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
      }
      for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
        int temp = out >= 0;
        pckTemp[bitw] |= (temp << packIdx);
        out = (temp == 0 ? out + (1 << (out_bit - bitw - 1)) : out - (1 << (out_bit - bitw - 1)));
      }
      // Full output block - write out
      if (packIdx == 0) {
          for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
              *pOut++ = pckTemp[bitw];
          }
          packIdx = pckWdt - 1;
         memset(pckTemp, 0, sizeof(int32_t) * out_bit);
      }
      else {
         packIdx--;
      }
      outCnt--;
   }
	
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version
 * Indexing: array, batch norm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, uint8_t in_bit, uint8_t out_bit) {

   // temporary output value (before binarization)
    uint32_t xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    // For holding batches of output values 
    pckDtype pckTemp[out_bit];
   // Weight index counter
   uint32_t wgtCnt = 0;
   int out = 0;

   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
       memset(pckTemp, 0, sizeof(pckTemp));
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
          out = 0;
          memset(outTemp, 0, sizeof(outTemp));
         for (uint16_t inCnt = 0; inCnt < numIn/pckWdt; inCnt++) {
             for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                 // XNOR multiplication
                 xnorTemp[bitw] = ~(pAct[pInd[wgtCnt] * in_bit + bitw] ^ pWgt[wgtCnt]);
                 // popcount/Accumulate
                 outTemp[bitw] += popcount(xnorTemp[bitw]);                 
             }
             wgtCnt++;
         }
         for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
             // Adjust the output value
             outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
             // Get the int full precision value 
             out += (outTemp[bitw] << (in_bit - bitw - 1));
         }
         // Binarize
         for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
             int temp = out > 0;
             // Shift 
             pckTemp[bitw] |= (temp << (pckWdt - packIdx - 1));
             out = (temp == 0 ? out + (1 << (out_bit - bitw - 1)) : out - (1 << (out_bit - bitw - 1)));
         }
      }
      for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
          *pOut++ = pckTemp[bitw];
      }
   }
}

#ifdef NEON

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 * Indexing: array, batch norm: no, SIMD: NEON (64)
 * numIn has to be a multiple of 64
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut) {

   // For holding inputs and weights
   int32x2_t vecAct, vecWgt;
   int64x1_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[2];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
      pckTemp = 0;
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
         vecOut = 0;
         for (uint16_t inCnt = 0; inCnt < numIn/(64); inCnt++) {
            // Load values
            actTemp[0] = *(pckDtype*)(pAct+*pInd++);
            actTemp[1] = *(pckDtype*)(pAct+*pInd++);
            vecAct = vld1_s32(actTemp);
            vecWgt = vld1_s32(pWgt);
            // XNOR
            vecAct = veor_s32(vecAct, vecWgt);
            vecAct = vmvn_s32(vecAct);
            // popcount
            // popcount only works on 8-bit vectors, so needs some casting
            // Not a problem here because those are binary vectors not values
            vecAct = vreinterpret_s32_s8(vcnt_s8(vreinterpret_s8_s32(vecAct)));
            // Now we need to do addition
            // 8x8b reduce to 4x16b
            vecAct = vreinterpret_s32_s16(vpaddl_s8(vreinterpret_s8_s32(vecAct)));
            // 4x16b to 2x32b
            vecAct = vpaddl_s16(vreinterpret_s16_s32(vecAct));
            // 2x32b to a single value
            vecOut += vpaddl_s32(vecAct);
            pWgt += 2;
         }
         // Extract the output
         outTemp = (int32_t) vget_lane_s64(vecOut, 0);
         // Adjust the value
         outTemp = outTemp - (numIn- outTemp);
         // Binarize
         outTemp = outTemp >= 0;
         // Shift
         outTemp = outTemp << (pckWdt-1-packIdx);
         // Pack
         pckTemp |= outTemp;
      }
      // Write output block
      pOut[outCnt] = pckTemp;
   }

}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 * Indexing: array, batch norm: no, SIMD: NEON (128)
 * numIn has to be a multiple of 128
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut) {

   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
      pckTemp = 0;
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
         vecOut[0] = 0;
         vecOut[1] = 0;
         for (uint16_t inCnt = 0; inCnt < numIn/(128); inCnt++) {
            // Load values
            actTemp[0] = *(pckDtype*)(pAct+*pInd++);
            actTemp[1] = *(pckDtype*)(pAct+*pInd++);
            actTemp[2] = *(pckDtype*)(pAct+*pInd++);
            actTemp[3] = *(pckDtype*)(pAct+*pInd++);
            vecAct = vld1q_s32(actTemp);
            vecWgt = vld1q_s32(pWgt);
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
            pWgt += 4;
         }
         // Extract the output
         outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
         // Adjust the value
         outTemp = outTemp - (numIn- outTemp);
         // Binarize
         outTemp = outTemp >= 0;
         // Shift
         outTemp = outTemp << (pckWdt-1-packIdx);
         // Pack
         pckTemp |= outTemp;
      }
      // Write output block
      pOut[outCnt] = pckTemp;
   }

}
#endif /* NEON */

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - pointer version
 * Indexing: pointer, batch norm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, uint8_t in_bit, uint8_t out_bit) {
   // Input counter
   uint16_t inCnt = numIn;
   // Output counter
   uint16_t outCnt = numOut;
   // temporary output value (before binarization)
   uint32_t xnorTmp[in_bit];
   int32_t  outTemp[in_bit];
   int out = 0;
   memset(xnorTmp, 0, sizeof(xnorTmp));
   
   while (outCnt) {
       memset(outTemp, 0, sizeof(outTemp));
      inCnt = numIn;
      while (inCnt) {
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              xnorTmp[bitw] = ~((*(pckDtype*)(pAct + (*pInd) * in_bit + bitw)) ^ *pWgt);
              xnorTmp[bitw] = popcount(xnorTmp[bitw]);
              // Accumulation
              outTemp[bitw] += xnorTmp[bitw];
          }
          pInd++;
          pWgt++;
         // Decrement input counter
         inCnt -= pckWdt;
      }
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
      }
      // Write output block
      *pOut++ = (float)out;
      outCnt--;
   }
	
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version
 * Indexing: array, batch norm: no, SIMD: none
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, uint8_t in_bit, uint8_t out_bit) {

   // temporary output value (before binarization)
    uint32_t xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    // Weight index counter
    uint32_t wgtCnt = 0;
    int16_t out = 0;

   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
       memset(outTemp, 0, sizeof(outTemp));
       out = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/pckWdt; inCnt++) {
         // XNOR multiplication
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              // XNOR multiplication
              xnorTemp[bitw] = ~(pAct[pInd[wgtCnt]*in_bit+bitw] ^ pWgt[wgtCnt]);
              // popcount//Accummulate
              outTemp[bitw] += popcount(xnorTemp[bitw]);
          }
         wgtCnt++;
      }
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
         
      }
      *pOut++ = (float)out;
   }

}

#ifdef NEON

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 * Indexing: array, batch norm: no, SIMD: NEON (64)
 * numIn has to be a multiple of 64
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut) {

   // For holding inputs and weights
   int32x2_t vecAct, vecWgt;
   int64x1_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[2];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
      vecOut = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/(64); inCnt++) {
         // Load values
         actTemp[0] = *(pckDtype*)(pAct+*pInd++);
         actTemp[1] = *(pckDtype*)(pAct+*pInd++);
         vecAct = vld1_s32(actTemp);
         vecWgt = vld1_s32(pWgt);
         // XNOR
         vecAct = veor_s32(vecAct, vecWgt);
         vecAct = vmvn_s32(vecAct);
         // popcount
         // popcount only works on 8-bit vectors, so needs some casting
         // Not a problem here because those are binary vectors not values
         vecAct = vreinterpret_s32_s8(vcnt_s8(vreinterpret_s8_s32(vecAct)));
         // Now we need to do addition
         // 8x8b reduce to 4x16b
         vecAct = vreinterpret_s32_s16(vpaddl_s8(vreinterpret_s8_s32(vecAct)));
         // 4x16b to 2x32b
         vecAct = vpaddl_s16(vreinterpret_s16_s32(vecAct));
         // 2x32b to a single value
         vecOut += vpaddl_s32(vecAct);
         pWgt += 2;
      }
      // Extract the output
      outTemp = (int32_t) vget_lane_s64(vecOut, 0);
      // Adjust the value
      outTemp = outTemp - (numIn- outTemp);
      // Write output block
      pOut[outCnt] = (float) outTemp;
   }

}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 * Indexing: array, batch norm: no, SIMD: NEON (128)
 * numIn has to be a multiple of 128 
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 */
void Fc3pxnNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut) {

   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];
   // Output value
   int32_t   outTemp;

   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
      vecOut[0] = 0;
      vecOut[1] = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/(128); inCnt++) {
         // Load values
         actTemp[0] = *(pckDtype*)(pAct+*pInd++);
         actTemp[1] = *(pckDtype*)(pAct+*pInd++);
         actTemp[2] = *(pckDtype*)(pAct+*pInd++);
         actTemp[3] = *(pckDtype*)(pAct+*pInd++);
         vecAct = vld1q_s32(actTemp);
         vecWgt = vld1q_s32(pWgt);
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
         // 4x32b to two values 
         vecOut += vpaddlq_s32(vecAct);
         pWgt += 4;
      }
      // Extract the output
      outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
      // Adjust the value
      outTemp = outTemp - (numIn- outTemp);
      // Write output block
      pOut[outCnt] = (float) outTemp;
   }

}

#endif /* NEON */

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - pointer version, batch norm
 * Indexing: pointer, batch norm: yes, SIMD: none      
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void FcBn3pxnPtr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, 
    const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype* __restrict thresh, pckDtype* __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {

   // Input counter
   uint16_t inCnt = numIn;
   // Output counter
   uint16_t outCnt = numOut;
   // temporary output value (before binarization)
   uint32_t xnorTmp[in_bit];
   int32_t  outTemp[in_bit];
   int32_t pckTemp[out_bit];
   int out = 0;
   // Packing/shifting  index
   uint8_t packIdx = pckWdt-1;
   memset(pckTemp, 0, sizeof(int32_t) * out_bit);
   
   while (outCnt) {
       memset(outTemp, 0, in_bit * sizeof(int32_t));
       out = 0;
      inCnt = numIn;
      while (inCnt) {
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              xnorTmp[bitw] = ~((*(pckDtype*)(pAct + (*pInd) * in_bit + bitw)) ^ *pWgt);
              xnorTmp[bitw] = popcount(xnorTmp[bitw]);
              // Accumulation
              outTemp[bitw] += xnorTmp[bitw];
          }
          pInd++;
          pWgt++;
          // Decrement input counter
          inCnt -= pckWdt;
      }
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
      }
      int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
      int temp = 0;
      for (int i = 0; i != in_bit; i++) {
          temp |= (1 << i);
      }
      temp = temp & out;
      out_temp += temp;
      for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
          int temp = out_temp >= *thresh;
          // Shift 
          pckTemp[bitw] |= (temp << (packIdx));
          //out_temp = (temp ^ (1 & ((*signs) >> (pckWdt - ks - 1))) ? out_temp + *offsets * (1 << (out_bit - bitw - 1)) : out_temp - *offsets * (1 << (out_bit - bitw - 1)));
          out_temp = (temp ^ (1 & ((*sign) >> (packIdx))) ? out_temp + ((*offset) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offset) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
      }
      thresh++;
      offset++;
      // Full output block - write out
      if (packIdx == 0) {
          for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
              *pOut++ = ~(pckTemp[bitw] ^ (*sign));
          }
          sign++;
          memset(pckTemp, 0, sizeof(int32_t) * out_bit);
         packIdx = pckWdt-1;
      }
      else {
         packIdx--;
      }
      outCnt--;
   }
	
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 * Indexing: array, batch norm: yes, SIMD: none      
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void FcBn3pxnArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit) {

   // temporary output value (before binarization)
    uint32_t xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
    // For holding batches of output values 
    pckDtype pckTemp[out_bit];
   // Weight index counter
   uint32_t wgtCnt = 0;
   int out = 0;
   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
       memset(pckTemp, 0, sizeof(pckTemp));
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
          memset(outTemp, 0, sizeof(outTemp));
          out = 0;
         for (uint16_t inCnt = 0; inCnt < numIn/pckWdt; inCnt++) {
             for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
                 // XNOR multiplication
                 xnorTemp[bitw] = ~(pAct[pInd[wgtCnt] * in_bit + bitw] ^ pWgt[wgtCnt]);
                 // popcount//Accummulate
                 outTemp[bitw] += popcount(xnorTemp[bitw]);
             }
            wgtCnt++;
         }
         for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
             // Adjust the output value
             outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
             // Get the int full precision value 
             out += (outTemp[bitw] << (in_bit - bitw - 1));
         }
         int out_temp = out >> (in_bit) << (16);/// pow(2, in_bit);
         int temp = 0;
         for (int i = 0; i != in_bit; i++) {
             temp |= (1 << i);
         }
         temp = temp & out;
         out_temp += temp;
         for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
             int temp = out_temp > *thresh;
             // Shift 
             pckTemp[bitw] |= (temp << (packIdx));
             out_temp = (temp ^ (1 & ((*sign) >> (packIdx))) ? out_temp + ((*offset) >> (bitw + 1))/* * pow(2, -bitw - 1)*/ : out_temp - ((*offset) >> (bitw + 1))/* * pow(2, -bitw - 1)*/);
         }
         thresh++;
         offset++;
      }
      for (uint8_t bitw = 0; bitw != out_bit; bitw++) {
          *pOut++ = ~(pckTemp[bitw] ^ sign[outCnt]);
      }
   }
}

#ifdef NEON

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 * Indexing: array, batch norm: yes, SIMD: NEON (64)
 * numIn has to be a multiple of 64
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void FcBn3pxnNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // For holding inputs and weights
   int32x2_t vecAct, vecWgt;
   int64x1_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[2];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
      pckTemp = 0;
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
         vecOut = 0;
         for (uint16_t inCnt = 0; inCnt < numIn/(64); inCnt++) {
            // Load values
            actTemp[0] = *(pckDtype*)(pAct+*pInd++);
            actTemp[1] = *(pckDtype*)(pAct+*pInd++);
            vecAct = vld1_s32(actTemp);
            vecWgt = vld1_s32(pWgt);
            // XNOR
            vecAct = veor_s32(vecAct, vecWgt);
            vecAct = vmvn_s32(vecAct);
            // popcount
            // popcount only works on 8-bit vectors, so needs some casting
            // Not a problem here because those are binary vectors not values
            vecAct = vreinterpret_s32_s8(vcnt_s8(vreinterpret_s8_s32(vecAct)));
            // Now we need to do addition
            // 8x8b reduce to 4x16b
            vecAct = vreinterpret_s32_s16(vpaddl_s8(vreinterpret_s8_s32(vecAct)));
            // 4x16b to 2x32b
            vecAct = vpaddl_s16(vreinterpret_s16_s32(vecAct));
            // 2x32b to a single value
            vecOut += vpaddl_s32(vecAct);
            pWgt += 2;
         }
         // Extract the output
         outTemp = (int32_t) vget_lane_s64(vecOut, 0);
         // Adjust the value
         outTemp = outTemp - (numIn- outTemp);
         // Batch normalize/ binarize
         outTemp = (bnPrec) outTemp >= *thresh++;
         // Shift
         outTemp = outTemp << (pckWdt-1-packIdx);
         // Pack
         pckTemp |= outTemp;
      }
      // Write output block
      pckTemp = ~(pckTemp ^ *sign++);
      pOut[outCnt] = pckTemp;
   }

}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 * Indexing: array, batch norm: yes, SIMD: NEON (128)
 * numIn has to be a multiple of 128
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] thresh - pointer to batch normalization threshold (if NULL, Bn is skipped)
 * @param[in] sign - pointer to the packed batch normalization signs
 */
void FcBn3pxnNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign) {

   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut/pckWdt; outCnt++) {
      pckTemp = 0;
      for (uint8_t packIdx = 0; packIdx < pckWdt; packIdx++) {
         vecOut[0] = 0;
         vecOut[1] = 0;
         for (uint16_t inCnt = 0; inCnt < numIn/(128); inCnt++) {
            // Load values
            actTemp[0] = *(pckDtype*)(pAct+*pInd++);
            actTemp[1] = *(pckDtype*)(pAct+*pInd++);
            actTemp[2] = *(pckDtype*)(pAct+*pInd++);
            actTemp[3] = *(pckDtype*)(pAct+*pInd++);
            vecAct = vld1q_s32(actTemp);
            vecWgt = vld1q_s32(pWgt);
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
            pWgt += 4;
         }
         // Extract the output
         outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
         // Adjust the value
         outTemp = outTemp - (numIn- outTemp);
         // Batch normalize/ binarize
         outTemp = (bnPrec) outTemp >= *thresh++;
         // Shift
         outTemp = outTemp << (pckWdt-1-packIdx);
         // Pack
         pckTemp |= outTemp;
      }
      // Write output block
      pckTemp = ~(pckTemp ^ *sign++);
      pOut[outCnt] = pckTemp;
   }

}
#endif /* NEON */

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - pointer version, batch norm
 * Indexing: pointer, batch norm: yes, SIMD: none      
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] mean - batch norm mean (per output) 
 * @param[in] var - sqrt(var - epsilon) - corrected batch norm variance (per output)
 * @param[in] gamma - batch norm gamma (per output)
 * @param[in] beta - batch norm beta (per output)
 */
void FcBn3pxnPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta, uint8_t in_bit, uint8_t out_bit) {

   // Input counter
   uint16_t inCnt = numIn;
   // Output counter
   uint16_t outCnt = numOut;
   // temporary output value (before binarization)
   int16_t out = 0;
   // temporary output value (before binarization)
   uint32_t xnorTemp[in_bit];
   int32_t  outTemp[in_bit];
   memset(xnorTemp, 0, sizeof(xnorTemp));
   while (outCnt) {
       memset(outTemp, 0, sizeof(outTemp));
       out = 0;
      inCnt = numIn;
      while (inCnt) {
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              // XNOR multiplication
              xnorTemp[bitw] = ~((*(pckDtype*)(pAct + (*pInd) * in_bit + bitw)) ^ *pWgt);
              // popcount//Accummulate
              outTemp[bitw] += popcount(xnorTemp[bitw]);
          }
          pWgt++;
         // Decrement input counter
         inCnt -= pckWdt;
      }
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
      }
      // Batch normalize
      // Write output out
      *pOut++ = *gamma++ * (((bnPrec)out - *mean++) / (*var++)) + *beta++;
      outCnt--;
   }
	
}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm
 * Indexing: array, batch norm: yes, SIMD: none      
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] mean - batch norm mean (per output) 
 * @param[in] var - sqrt(var - epsilon) - corrected batch norm variance (per output)
 * @param[in] gamma - batch norm gamma (per output)
 * @param[in] beta - batch norm beta (per output)
 */
void FcBn3pxnArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta,uint8_t in_bit, uint8_t out_bit) {

   // temporary output value (before binarization)
    uint32_t xnorTemp[in_bit];
    int32_t  outTemp[in_bit];
   // Weight index counter
   uint32_t wgtCnt = 0;
   int out = 0;
   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
       memset(outTemp, 0, sizeof(outTemp));
       out = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/pckWdt; inCnt++) {
          for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
              // XNOR multiplication
              xnorTemp[bitw] = ~(pAct[pInd[wgtCnt]* in_bit + bitw] ^ pWgt[wgtCnt]);
              // popcount//Accummulate
              outTemp[bitw] += popcount(xnorTemp[bitw]);
          }
         wgtCnt++;
      }
      // Batch normalize
      for (uint8_t bitw = 0; bitw != in_bit; bitw++) {
          // Adjust the output value
          outTemp[bitw] = outTemp[bitw] - (numIn - outTemp[bitw]);
          // Get the int full precision value 
          out += (outTemp[bitw] << (in_bit - bitw - 1));
      }
      *pOut++ = (float)*gamma++ * (((bnPrec)out - *mean++) / (*var++)) + *beta++;
   }
}

#ifdef NEON

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 * Indexing: array, batch norm: yes, SIMD: NEON (64) 
 * numIn has to be a multiple of 64
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] mean - batch norm mean (per output) 
 * @param[in] var - sqrt(var - epsilon) - corrected batch norm variance (per output)
 * @param[in] gamma - batch norm gamma (per output)
 * @param[in] beta - batch norm beta (per output)
 */
void FcBn3pxnNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta) {

   // For holding inputs and weights
   int32x2_t vecAct, vecWgt;
   int64x1_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[2];
   // Output value
   int32_t   outTemp;
   // For holding batches of output values 
   pckDtype pckTemp = 0;

   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
      vecOut = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/(64); inCnt++) {
         // Load values
         actTemp[0] = *(pckDtype*)(pAct+*pInd++);
         actTemp[1] = *(pckDtype*)(pAct+*pInd++);
         vecAct = vld1_s32(actTemp);
         vecWgt = vld1_s32(pWgt);
         // XNOR
         vecAct = veor_s32(vecAct, vecWgt);
         vecAct = vmvn_s32(vecAct);
         // popcount
         // popcount only works on 8-bit vectors, so needs some casting
         // Not a problem here because those are binary vectors not values
         vecAct = vreinterpret_s32_s8(vcnt_s8(vreinterpret_s8_s32(vecAct)));
         // Now we need to do addition
         // 8x8b reduce to 4x16b
         vecAct = vreinterpret_s32_s16(vpaddl_s8(vreinterpret_s8_s32(vecAct)));
         // 4x16b to 2x32b
         vecAct = vpaddl_s16(vreinterpret_s16_s32(vecAct));
         // 2x32b to a single value
         vecOut += vpaddl_s32(vecAct);
         pWgt += 2;
      }
      // Extract the output
      outTemp = (int32_t) vget_lane_s64(vecOut, 0);
      // Adjust the value
      outTemp = outTemp - (numIn- outTemp);
      // Batch normalize
      // Write output block
      pOut[outCnt] = (float) *gamma++ * (((bnPrec) outTemp - *mean++)/(*var++)) + *beta++;
   }

}

/**
 * @details 3PXNet binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 * Indexing: array, batch norm: yes, SIMD: NEON (128)
 * numIn has to be a multiple of 128 
 * 
 * @param[in] pAct - pointer to the packed activation vector
 * @param[in] pWgt - pointer to the packed weight matrix (row-major flattened)
 * @param[in] pInd - pointer to the index matrix (row-major flattened)
 * @param[in] numIn - length of the input vector
 * @param[in] numOut - length of the output vector
 * @param[out] pOut - pointer to the packed output vector
 * @param[in] mean - batch norm mean (per output) 
 * @param[in] var - sqrt(var - epsilon) - corrected batch norm variance (per output)
 * @param[in] gamma - batch norm gamma (per output)
 * @param[in] beta - batch norm beta (per output)
 */
void FcBn3pxnNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, uint8_t * __restrict pInd, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta) {

   // For holding inputs and weights
   int32x4_t vecAct, vecWgt;
   int64x2_t vecOut ;
   // Holding temporary inputs
   pckDtype  actTemp[4];
   // Output value
   int32_t   outTemp;

   for (uint16_t outCnt = 0; outCnt < numOut; outCnt++) {
      vecOut[0] = 0;
      vecOut[1] = 0;
      for (uint16_t inCnt = 0; inCnt < numIn/(128); inCnt++) {
         // Load values
         actTemp[0] = *(pckDtype*)(pAct+*pInd++);
         actTemp[1] = *(pckDtype*)(pAct+*pInd++);
         actTemp[2] = *(pckDtype*)(pAct+*pInd++);
         actTemp[3] = *(pckDtype*)(pAct+*pInd++);
         vecAct = vld1q_s32(actTemp);
         vecWgt = vld1q_s32(pWgt);
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
         // 4x32b to two values 
         vecOut += vpaddlq_s32(vecAct);
         pWgt += 4;
      }
      // Extract the output
      outTemp = (int32_t) vgetq_lane_s64(vecOut, 0) + vgetq_lane_s64(vecOut, 1);
      // Adjust the value
      outTemp = outTemp - (numIn- outTemp);
      // Batch normalize
      // Write output block
      pOut[outCnt] = (float) *gamma++ * (((bnPrec) outTemp - *mean++)/(*var++)) + *beta++;
   }
}

#endif /* NEON */

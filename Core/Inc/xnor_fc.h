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
 * \file      xnor_dense_fc.h
 * \brief     Dense binarized (XNOR) fully-connected layer implementations
 * \author    Wojciech Romaszkan 
 * \author    NanoCAD Laboratory, University of California Los Angeles
 * \copyright MIT License
 */

#ifndef XNOR_DENSE_FC_H
#define XNOR_DENSE_FC_H

#ifdef __cplusplus
extern "C" {
#endif
    
#include "datatypes.h"
#ifdef NEON
#include "arm_neon.h"
#endif /* NEON */
#include <math.h>
#include <stdint.h>
#include "utils.h"

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - general wrapper.
 */
uint8_t FcXnorWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - general wrapper.
 */
uint8_t FcXnorNoBinWrap(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - pointer version
 */
void FcXnorPtr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version
 */
void FcXnorArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, uint8_t in_bit, uint8_t out_bit);

#ifdef NEON

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 */
void FcXnorNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version, NEON support
 */
void FcXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut);
#endif /* NEON */

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - pointer version, batch norm
 */
void FcXnorPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version, batch norm
 */
void FcXnorArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, uint8_t in_bit, uint8_t out_bit);
#ifdef NEON

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version, batch norm, NEON support
 */
void FcXnorNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);

/**
 * @brief  Dense binarized Fully Connected (FC) layer with output binarization - array version, batch norm, NEON support
 */
void FcXnorNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut);
#endif /* NEON */

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - pointer version 
 */
void FcBnXnorPtr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version 
 */
void FcBnXnorArr(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, pckDtype * __restrict thresh, pckDtype * __restrict sign, pckDtype* __restrict offset, uint8_t in_bit, uint8_t out_bit);

#ifdef NEON

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 */
void FcBnXnorNeon(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version, NEON support
 */
void FcBnXnorNeonQ(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, pckDtype * __restrict pOut, bnDtype * __restrict thresh, pckDtype * __restrict sign);
#endif /* NEON */

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - pointer version, batch norm
 */
void FcBnXnorPtrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta, uint8_t in_bit, uint8_t out_bit);

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version, batch norm
 */
void FcBnXnorArrNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta, uint8_t in_bit, uint8_t out_bit);

#ifdef NEON

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 */
void FcBnXnorNeonNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);

/**
 * @brief  Dense binarized Fully Connected (FC) layer without output binarization - array version, batch norm, NEON support
 */
void FcBnXnorNeonQNoBin(pckDtype * __restrict pAct, pckDtype * __restrict pWgt, const uint16_t numIn, const uint16_t numOut, float * __restrict pOut, bnDtype * __restrict mean, bnDtype * __restrict var, bnDtype * __restrict gamma, bnDtype * __restrict beta);
#endif /* NEON */


#ifdef __cplusplus
}
#endif

#endif /* XNOR_DENSE_FC_H */


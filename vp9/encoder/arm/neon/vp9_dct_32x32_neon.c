/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include "./vp9_rtcd.h"
#include "./vpx_config.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"

// Store given value to ouptut buffer at given 'row'
// The location to store is calucluated based on 'pass'
// See memory layout for more information
#define STORE_TO_OUTPUT(val, row, origin, pass, temp) {         \
    if(pass) {                                                  \
      temp = origin + (row % 8) * out_strd + (row - row % 8);   \
    } else {                                                    \
      temp = origin + row * out_strd;                           \
    }                                                           \
    vst1q_s16(temp, val);                                       \
}

// Computes butterfly. Cosine/Sine coefficients is symmetric/same
// Coefficients of both wings are cospi_16_64
static inline void butterfly_sym(int16x8_t in_1, int16x8_t in_2,
                                 int16x8_t *out_1, int16x8_t *out_2) {
  int32x4_t mul_1_lw = vmull_n_s16(vget_low_s16(in_1), (int16_t) cospi_16_64);
  int32x4_t mul_1_hi = vmull_n_s16(vget_high_s16(in_1), (int16_t) cospi_16_64);
  int32x4_t mul_2_lw = vmull_n_s16(vget_low_s16(in_2), (int16_t) cospi_16_64);
  int32x4_t mul_2_hi = vmull_n_s16(vget_high_s16(in_2), (int16_t) cospi_16_64);

  int32x4_t add_1_lw = vaddq_s32(mul_1_lw, mul_2_lw);
  int32x4_t add_1_hi = vaddq_s32(mul_1_hi, mul_2_hi);
  int32x4_t sub_2_lw = vsubq_s32(mul_1_lw, mul_2_lw);
  int32x4_t sub_2_hi = vsubq_s32(mul_1_hi, mul_2_hi);

  int16x4_t shrn_1_lw = vrshrn_n_s32(add_1_lw, DCT_CONST_BITS);
  int16x4_t shrn_1_hi = vrshrn_n_s32(add_1_hi, DCT_CONST_BITS);
  int16x4_t shrn_2_lw = vrshrn_n_s32(sub_2_lw, DCT_CONST_BITS);
  int16x4_t shrn_2_hi = vrshrn_n_s32(sub_2_hi, DCT_CONST_BITS);

  *out_1 = vcombine_s16(shrn_1_lw, shrn_1_hi);
  *out_2 = vcombine_s16(shrn_2_lw, shrn_2_hi);
}

// Computes butterfly. Cosine/Sine coefficients are unique
static inline void butterfly_std(int16x8_t in_1, int16x8_t in_2,
                                 int16x8_t *out_add, int16x8_t *out_sub,
                                 tran_high_t coeff1, tran_high_t coeff2) {
  {
    int32x4_t mul_1_lw = vmull_n_s16(vget_low_s16(in_1), (int16_t) coeff1);
    int32x4_t mul_1_hi = vmull_n_s16(vget_high_s16(in_1), (int16_t) coeff1);
    int32x4_t mul_2_lw = vmull_n_s16(vget_low_s16(in_2), (int16_t) coeff2);
    int32x4_t mul_2_hi = vmull_n_s16(vget_high_s16(in_2), (int16_t) coeff2);

    int32x4_t add_lw = vaddq_s32(mul_1_lw, mul_2_lw);
    int32x4_t add_hi = vaddq_s32(mul_1_hi, mul_2_hi);

    int16x4_t shrn_lw = vrshrn_n_s32(add_lw, DCT_CONST_BITS);
    int16x4_t shrn_hi = vrshrn_n_s32(add_hi, DCT_CONST_BITS);

    *out_add = vcombine_s16(shrn_lw, shrn_hi);
  }
  {
    int32x4_t mul_1_lw = vmull_n_s16(vget_low_s16(in_1), (int16_t) coeff2);
    int32x4_t mul_1_hi = vmull_n_s16(vget_high_s16(in_1), (int16_t) coeff2);
    int32x4_t mul_2_lw = vmull_n_s16(vget_low_s16(in_2), (int16_t) coeff1);
    int32x4_t mul_2_hi = vmull_n_s16(vget_high_s16(in_2), (int16_t) coeff1);

    int32x4_t sub_lw = vsubq_s32(mul_1_lw, mul_2_lw);
    int32x4_t sub_hi = vsubq_s32(mul_1_hi, mul_2_hi);

    int16x4_t shrn_lw = vrshrn_n_s32(sub_lw, DCT_CONST_BITS);
    int16x4_t shrn_hi = vrshrn_n_s32(sub_hi, DCT_CONST_BITS);

    *out_sub = vcombine_s16(shrn_lw, shrn_hi);
  }
}

// Compute half round shift of a block of 8x8
// Implements the function of
//tran_high_t half_round_shift(tran_high_t input)
static inline void half_round_shift_neon(int16x8_t *in_0, int16x8_t *in_1,
                                         int16x8_t *in_2, int16x8_t *in_3,
                                         int16x8_t *in_4, int16x8_t *in_5,
                                         int16x8_t *in_6, int16x8_t *in_7) {
  uint16x8_t sgn_0, sgn_1, sgn_2, sgn_3, sgn_4, sgn_5, sgn_6, sgn_7;
  int16x8_t one;
  int bit_depth, half_shft;

  bit_depth = 16 - 1;
  half_shft = 2;

  sgn_0 = vshrq_n_u16(vreinterpretq_u16_s16(*in_0), bit_depth);
  sgn_1 = vshrq_n_u16(vreinterpretq_u16_s16(*in_1), bit_depth);
  sgn_2 = vshrq_n_u16(vreinterpretq_u16_s16(*in_2), bit_depth);
  sgn_3 = vshrq_n_u16(vreinterpretq_u16_s16(*in_3), bit_depth);
  sgn_4 = vshrq_n_u16(vreinterpretq_u16_s16(*in_4), bit_depth);
  sgn_5 = vshrq_n_u16(vreinterpretq_u16_s16(*in_5), bit_depth);
  sgn_6 = vshrq_n_u16(vreinterpretq_u16_s16(*in_6), bit_depth);
  sgn_7 = vshrq_n_u16(vreinterpretq_u16_s16(*in_7), bit_depth);

  *in_0 = vaddq_s16(*in_0, vreinterpretq_s16_u16(sgn_0));
  *in_1 = vaddq_s16(*in_1, vreinterpretq_s16_u16(sgn_1));
  *in_2 = vaddq_s16(*in_2, vreinterpretq_s16_u16(sgn_2));
  *in_3 = vaddq_s16(*in_3, vreinterpretq_s16_u16(sgn_3));
  *in_4 = vaddq_s16(*in_4, vreinterpretq_s16_u16(sgn_4));
  *in_5 = vaddq_s16(*in_5, vreinterpretq_s16_u16(sgn_5));
  *in_6 = vaddq_s16(*in_6, vreinterpretq_s16_u16(sgn_6));
  *in_7 = vaddq_s16(*in_7, vreinterpretq_s16_u16(sgn_7));

  one = vdupq_n_s16(1);

  *in_0 = vaddq_s16(*in_0, one);
  *in_1 = vaddq_s16(*in_1, one);
  *in_2 = vaddq_s16(*in_2, one);
  *in_3 = vaddq_s16(*in_3, one);
  *in_4 = vaddq_s16(*in_4, one);
  *in_5 = vaddq_s16(*in_5, one);
  *in_6 = vaddq_s16(*in_6, one);
  *in_7 = vaddq_s16(*in_7, one);

  *in_0 = vshrq_n_s16(*in_0, half_shft);
  *in_1 = vshrq_n_s16(*in_1, half_shft);
  *in_2 = vshrq_n_s16(*in_2, half_shft);
  *in_3 = vshrq_n_s16(*in_3, half_shft);
  *in_4 = vshrq_n_s16(*in_4, half_shft);
  *in_5 = vshrq_n_s16(*in_5, half_shft);
  *in_6 = vshrq_n_s16(*in_6, half_shft);
  *in_7 = vshrq_n_s16(*in_7, half_shft);
}

// Compute half round shift of a 8x8 block
// Implements the function of
//output[j * 32 + i] = (temp_out[j] + 1 + (temp_out[j] > 0)) >> 2;
static inline void fdct_preproc_pass2(int16x8_t *in_0, int16x8_t *in_1,
                                      int16x8_t *in_2, int16x8_t *in_3,
                                      int16x8_t *in_4, int16x8_t *in_5,
                                      int16x8_t *in_6, int16x8_t *in_7) {
  uint16x8_t sgn_0, sgn_1, sgn_2, sgn_3, sgn_4, sgn_5, sgn_6, sgn_7;
  int bit_depth = 16 - 1;
  int half_shft = 2;

  sgn_0 = vshrq_n_u16(vreinterpretq_u16_s16(*in_0), bit_depth);
  sgn_1 = vshrq_n_u16(vreinterpretq_u16_s16(*in_1), bit_depth);
  sgn_2 = vshrq_n_u16(vreinterpretq_u16_s16(*in_2), bit_depth);
  sgn_3 = vshrq_n_u16(vreinterpretq_u16_s16(*in_3), bit_depth);
  sgn_4 = vshrq_n_u16(vreinterpretq_u16_s16(*in_4), bit_depth);
  sgn_5 = vshrq_n_u16(vreinterpretq_u16_s16(*in_5), bit_depth);
  sgn_6 = vshrq_n_u16(vreinterpretq_u16_s16(*in_6), bit_depth);
  sgn_7 = vshrq_n_u16(vreinterpretq_u16_s16(*in_7), bit_depth);

  *in_0 = vsubq_s16(*in_0, vreinterpretq_s16_u16(sgn_0));
  *in_1 = vsubq_s16(*in_1, vreinterpretq_s16_u16(sgn_1));
  *in_2 = vsubq_s16(*in_2, vreinterpretq_s16_u16(sgn_2));
  *in_3 = vsubq_s16(*in_3, vreinterpretq_s16_u16(sgn_3));
  *in_4 = vsubq_s16(*in_4, vreinterpretq_s16_u16(sgn_4));
  *in_5 = vsubq_s16(*in_5, vreinterpretq_s16_u16(sgn_5));
  *in_6 = vsubq_s16(*in_6, vreinterpretq_s16_u16(sgn_6));
  *in_7 = vsubq_s16(*in_7, vreinterpretq_s16_u16(sgn_7));

  *in_0 = vrshrq_n_s16(*in_0, half_shft);
  *in_1 = vrshrq_n_s16(*in_1, half_shft);
  *in_2 = vrshrq_n_s16(*in_2, half_shft);
  *in_3 = vrshrq_n_s16(*in_3, half_shft);
  *in_4 = vrshrq_n_s16(*in_4, half_shft);
  *in_5 = vrshrq_n_s16(*in_5, half_shft);
  *in_6 = vrshrq_n_s16(*in_6, half_shft);
  *in_7 = vrshrq_n_s16(*in_7, half_shft);
}

// Transpose an 8x8 matrix
// Input rows and output colums 0-7 must be in
// v_x0, v_x1, v_x2, v_x3, v_x4, v_x4, v_x5, v_x6, v_x7 respectively
static inline void transpose_8x8(int16x8_t *v_x0, int16x8_t *v_x1,
                                 int16x8_t *v_x2, int16x8_t *v_x3,
                                 int16x8_t *v_x4, int16x8_t *v_x5,
                                 int16x8_t *v_x6, int16x8_t *v_x7) {
  const int16x8x2_t r01_s16 = vtrnq_s16(*v_x0, *v_x1);
  const int16x8x2_t r23_s16 = vtrnq_s16(*v_x2, *v_x3);
  const int16x8x2_t r45_s16 = vtrnq_s16(*v_x4, *v_x5);
  const int16x8x2_t r67_s16 = vtrnq_s16(*v_x6, *v_x7);

  const int32x4x2_t r01_s32 = vtrnq_s32(vreinterpretq_s32_s16(r01_s16.val[0]),
                                        vreinterpretq_s32_s16(r23_s16.val[0]));
  const int32x4x2_t r23_s32 = vtrnq_s32(vreinterpretq_s32_s16(r01_s16.val[1]),
                                        vreinterpretq_s32_s16(r23_s16.val[1]));
  const int32x4x2_t r45_s32 = vtrnq_s32(vreinterpretq_s32_s16(r45_s16.val[0]),
                                        vreinterpretq_s32_s16(r67_s16.val[0]));
  const int32x4x2_t r67_s32 = vtrnq_s32(vreinterpretq_s32_s16(r45_s16.val[1]),
                                        vreinterpretq_s32_s16(r67_s16.val[1]));

  *v_x0 = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r01_s32.val[0]),
                   vget_low_s32(r45_s32.val[0])));
  *v_x2 = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r01_s32.val[1]),
                   vget_low_s32(r45_s32.val[1])));
  *v_x1 = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r23_s32.val[0]),
                   vget_low_s32(r67_s32.val[0])));
  *v_x3 = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r23_s32.val[1]),
                   vget_low_s32(r67_s32.val[1])));
  *v_x4 = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r01_s32.val[0]),
                   vget_high_s32(r45_s32.val[0])));
  *v_x6 = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r01_s32.val[1]),
                   vget_high_s32(r45_s32.val[1])));
  *v_x5 = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r23_s32.val[0]),
                   vget_high_s32(r67_s32.val[0])));
  *v_x7 = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r23_s32.val[1]),
                   vget_high_s32(r67_s32.val[1])));
}

// Function to compute Forward DCT 32x32 with RD
// -----------------------------------------------------------------------------
// Memory layout
// -----------------------------------------------------------------------------
// We use 3 buffers, input, output and intermediate buffer
//
// Input buffer
//   Input buffer will be a 32x32 buffer with a stride
//   Input buffer is used only in first pass, values are just read from it.
//
// Output buffer
//   Output buffer will be a 32x32 buffer
//   Output buffer will be used to write outputs in both the passes,
//       hence in the second pass it will be used as the input buffer too.
//   The structure of reading and writing into output buffer are different
//       in different passes.
//   In first pass, one 32x8 band is processed at a time. The macros for
//       storing will use the row numbers directly to load.
//   In second pass, one 8x32 band is processed at a time. The input is
//       organized in columns and hence it will have to transposed before
//       processing. For the ease of transpose, the 8x32 band is further
//       divided into four 8x8 sub-bands. Hence
//       - sub-band 0 will correspond to output[0]  to output[7]
//       - sub-band 1 will correspond to output[8]  to output[15]
//       - sub-band 2 will correspond to output[16] to output[23]
//       - sub-band 3 will correspond to output[24] to output[31]
//
// intermediate buffer
//   intermediate buffer is the scratch buffer, with size of 32x8
//
// -----------------------------------------------------------------------------
// Function design
// -----------------------------------------------------------------------------
// This 2-D transform is done by performing a 1-D transform in two passes
// The 1-D transform is always performed on columns
//  - In the first pass the data is used as is.
//  - In the second pass the input is transposed first and then used for
//    transforms
//  - The final output is transposed again
//
// Each pass is composed of 4 loops, with each iteration processing 32x8 band
// In each iteration the processing is done in blocks A, B, C and D.
//   Block A corresponds to row 0-3
//   Block B corresponds to row 4-7
//   Block C corresponds to row 8-15
//   Block D corresponds to row 16-31

void vp9_fdct32x32_rd_neon(const tran_low_t *input, tran_low_t *output,
                           int strd) {
  // Set input as actual input for pass 0
  const tran_low_t *in = input - 8;
  int in_strd = strd;
  // Intermediate memory
  tran_low_t interm[32 * 8];
  int interm_strd = 8;
  // Set output as output mem for pass 0
  tran_low_t *out = output - 8;
  // Stride of output to move between rows
  int out_strd = 32;
  // Pass of transform
  int pass;
  // To round or not to round in RD
  int round = 0;
  // Loop counter for Sub bands
  int band;

  for (pass = 0; pass < 2; pass++) {
    for (band = 0; band < 4; band++) {
      // Load two 8x8 sub bands. The matrices must be complementary
      // if coeff i is in a matrix 31-i should be in other.
      // In the second pass, we need to transpose the coefficents and
      // adjust the magnitude as below
      //output[j * 32 + i] = (temp_out[j] + 1 + (temp_out[j] > 0)) >> 2;
      if (pass == 0) {
        int row;
        // Adjust input pointer
        // Move to next 32x8 block
        in += 8;
        out += 8;

        // Do the Stage 1 for fdct pass 0
        for (row = 0; row < 16; row += 4) {
          // load data from input row 0-7
          int16x8_t input_0 = vshlq_n_s16(vld1q_s16(&in[(row + 0) * in_strd]),
                                          2);
          int16x8_t input_1 = vshlq_n_s16(vld1q_s16(&in[(row + 1) * in_strd]),
                                          2);
          int16x8_t input_2 = vshlq_n_s16(vld1q_s16(&in[(row + 2) * in_strd]),
                                          2);
          int16x8_t input_3 = vshlq_n_s16(vld1q_s16(&in[(row + 3) * in_strd]),
                                          2);

          // load data from input row 31-24
          int16x8_t input_31 = vshlq_n_s16(
              vld1q_s16(&in[(31 - (row + 0)) * in_strd]), 2);
          int16x8_t input_30 = vshlq_n_s16(
              vld1q_s16(&in[(31 - (row + 1)) * in_strd]), 2);
          int16x8_t input_29 = vshlq_n_s16(
              vld1q_s16(&in[(31 - (row + 2)) * in_strd]), 2);
          int16x8_t input_28 = vshlq_n_s16(
              vld1q_s16(&in[(31 - (row + 3)) * in_strd]), 2);

          // -------------------------------------------------------------------
          // STAGE 1, BLOCK A, B, C and D : 0-31
          // -------------------------------------------------------------------
          // 1) First stage computation of 32x8/8x32 block
          // 2) storing of this 32x8 coeffs into intermediate buffer

          int16x8_t stg1_0 = vaddq_s16(input_0, input_31);
          int16x8_t stg1_1 = vaddq_s16(input_1, input_30);
          int16x8_t stg1_2 = vaddq_s16(input_2, input_29);
          int16x8_t stg1_3 = vaddq_s16(input_3, input_28);

          int16x8_t stg1_31 = vsubq_s16(input_0, input_31);
          int16x8_t stg1_30 = vsubq_s16(input_1, input_30);
          int16x8_t stg1_29 = vsubq_s16(input_2, input_29);
          int16x8_t stg1_28 = vsubq_s16(input_3, input_28);

          vst1q_s16(&interm[(row + 0) * interm_strd], stg1_0);
          vst1q_s16(&interm[(row + 1) * interm_strd], stg1_1);
          vst1q_s16(&interm[(row + 2) * interm_strd], stg1_2);
          vst1q_s16(&interm[(row + 3) * interm_strd], stg1_3);

          vst1q_s16(&interm[(31 - row - 0) * interm_strd], stg1_31);
          vst1q_s16(&interm[(31 - row - 1) * interm_strd], stg1_30);
          vst1q_s16(&interm[(31 - row - 2) * interm_strd], stg1_29);
          vst1q_s16(&interm[(31 - row - 3) * interm_strd], stg1_28);
        }
      } else {
        int col;
        // In here we have to process 2 8x8 block at once, since tranpose
        // operates at 8x8 block level
        // Adjust input and output pointer to move to next 32x8 block
        in += 8 * in_strd;
        out += 8 * out_strd;

        // Do the Stage 1 for fdct pass 0
        for (col = 0; col < 16; col += 8) {

          // load data from input row 0-7
          int16x8_t input_0 = vld1q_s16(&in[0 * in_strd + col]);
          int16x8_t input_1 = vld1q_s16(&in[1 * in_strd + col]);
          int16x8_t input_2 = vld1q_s16(&in[2 * in_strd + col]);
          int16x8_t input_3 = vld1q_s16(&in[3 * in_strd + col]);
          int16x8_t input_4 = vld1q_s16(&in[4 * in_strd + col]);
          int16x8_t input_5 = vld1q_s16(&in[5 * in_strd + col]);
          int16x8_t input_6 = vld1q_s16(&in[6 * in_strd + col]);
          int16x8_t input_7 = vld1q_s16(&in[7 * in_strd + col]);

          // load data from input row 31-24
          int16x8_t input_31 = vld1q_s16(&in[7 * in_strd + 24 - col]);
          int16x8_t input_30 = vld1q_s16(&in[6 * in_strd + 24 - col]);
          int16x8_t input_29 = vld1q_s16(&in[5 * in_strd + 24 - col]);
          int16x8_t input_28 = vld1q_s16(&in[4 * in_strd + 24 - col]);
          int16x8_t input_27 = vld1q_s16(&in[3 * in_strd + 24 - col]);
          int16x8_t input_26 = vld1q_s16(&in[2 * in_strd + 24 - col]);
          int16x8_t input_25 = vld1q_s16(&in[1 * in_strd + 24 - col]);
          int16x8_t input_24 = vld1q_s16(&in[0 * in_strd + 24 - col]);

          transpose_8x8(&input_0, &input_1, &input_2, &input_3, &input_4,
                        &input_5, &input_6, &input_7);

          fdct_preproc_pass2(&input_0, &input_1, &input_2, &input_3, &input_4,
                             &input_5, &input_6, &input_7);

          transpose_8x8(&input_24, &input_25, &input_26, &input_27, &input_28,
                        &input_29, &input_30, &input_31);

          fdct_preproc_pass2(&input_24, &input_25, &input_26, &input_27,
                             &input_28, &input_29, &input_30, &input_31);

          // -------------------------------------------------------------------
          // STAGE 1, BLOCK A, B, C and D : 0-31
          // -------------------------------------------------------------------
          // 1) Transposition of 32x8 block if its pass 2
          // 2) First stage computation of 32x8/8x32 block
          // 3) storing of this 32x8 coeffs into intermediate buffer
          {
            int16x8_t stg1_0 = vaddq_s16(input_0, input_31);
            int16x8_t stg1_1 = vaddq_s16(input_1, input_30);
            int16x8_t stg1_2 = vaddq_s16(input_2, input_29);
            int16x8_t stg1_3 = vaddq_s16(input_3, input_28);
            int16x8_t stg1_31 = vsubq_s16(input_0, input_31);
            int16x8_t stg1_30 = vsubq_s16(input_1, input_30);
            int16x8_t stg1_29 = vsubq_s16(input_2, input_29);
            int16x8_t stg1_28 = vsubq_s16(input_3, input_28);

            int16x8_t stg1_4 = vaddq_s16(input_4, input_27);
            int16x8_t stg1_5 = vaddq_s16(input_5, input_26);
            int16x8_t stg1_6 = vaddq_s16(input_6, input_25);
            int16x8_t stg1_7 = vaddq_s16(input_7, input_24);
            int16x8_t stg1_27 = vsubq_s16(input_4, input_27);
            int16x8_t stg1_26 = vsubq_s16(input_5, input_26);
            int16x8_t stg1_25 = vsubq_s16(input_6, input_25);
            int16x8_t stg1_24 = vsubq_s16(input_7, input_24);

            vst1q_s16(&interm[(col + 0) * interm_strd], stg1_0);
            vst1q_s16(&interm[(col + 1) * interm_strd], stg1_1);
            vst1q_s16(&interm[(col + 2) * interm_strd], stg1_2);
            vst1q_s16(&interm[(col + 3) * interm_strd], stg1_3);
            vst1q_s16(&interm[(31 - col - 0) * interm_strd], stg1_31);
            vst1q_s16(&interm[(31 - col - 1) * interm_strd], stg1_30);
            vst1q_s16(&interm[(31 - col - 2) * interm_strd], stg1_29);
            vst1q_s16(&interm[(31 - col - 3) * interm_strd], stg1_28);

            vst1q_s16(&interm[(col + 4) * interm_strd], stg1_4);
            vst1q_s16(&interm[(col + 5) * interm_strd], stg1_5);
            vst1q_s16(&interm[(col + 6) * interm_strd], stg1_6);
            vst1q_s16(&interm[(col + 7) * interm_strd], stg1_7);
            vst1q_s16(&interm[(31 - col - 4) * interm_strd], stg1_27);
            vst1q_s16(&interm[(31 - col - 5) * interm_strd], stg1_26);
            vst1q_s16(&interm[(31 - col - 6) * interm_strd], stg1_25);
            vst1q_s16(&interm[(31 - col - 7) * interm_strd], stg1_24);
          }
        }
      }
      // Process block A and B and C, STAGE 2 to STAGE 6
      {
        // ---------------------------------------------------------------------
        // STAGE 2, BLOCK A, B, C : 0-15
        // ---------------------------------------------------------------------
        // 1) Load coeffs 1-16 in i, 16-i format
        // 2) Store the subtracted coeffs in butterfly(corresponding to BLOCK C)
        // 3) Don't store the added coeffs, keep them in registers so that wecan
        //    process block A and B together without memory access

        // variables that transfer data between Stage 2 and Stage 3
        int16x8_t stg2_0, stg2_1, stg2_2, stg2_3;
        int16x8_t stg2_4, stg2_5, stg2_6, stg2_7;
        {
          {
            //load data from input row 0-4
            int16x8_t input_0 = vld1q_s16(&interm[0 * interm_strd]);
            int16x8_t input_1 = vld1q_s16(&interm[1 * interm_strd]);
            int16x8_t input_2 = vld1q_s16(&interm[2 * interm_strd]);
            int16x8_t input_3 = vld1q_s16(&interm[3 * interm_strd]);

            //load data from input row 16-12
            int16x8_t input_16 = vld1q_s16(&interm[(16 - 1) * interm_strd]);
            int16x8_t input_15 = vld1q_s16(&interm[(16 - 2) * interm_strd]);
            int16x8_t input_14 = vld1q_s16(&interm[(16 - 3) * interm_strd]);
            int16x8_t input_13 = vld1q_s16(&interm[(16 - 4) * interm_strd]);

            stg2_0 = vaddq_s16(input_0, input_16);
            stg2_1 = vaddq_s16(input_1, input_15);
            stg2_2 = vaddq_s16(input_2, input_14);
            stg2_3 = vaddq_s16(input_3, input_13);
            {
              int16x8_t stg2_16 = vsubq_s16(input_0, input_16);
              int16x8_t stg2_15 = vsubq_s16(input_1, input_15);
              int16x8_t stg2_14 = vsubq_s16(input_2, input_14);
              int16x8_t stg2_13 = vsubq_s16(input_3, input_13);

              vst1q_s16(&interm[(16 - 1) * interm_strd], stg2_16);
              vst1q_s16(&interm[(16 - 2) * interm_strd], stg2_15);
              vst1q_s16(&interm[(16 - 3) * interm_strd], stg2_14);
              vst1q_s16(&interm[(16 - 4) * interm_strd], stg2_13);
            }
          }
          {
            // load data from input row 4-7
            int16x8_t input_4 = vld1q_s16(&interm[4 * interm_strd]);
            int16x8_t input_5 = vld1q_s16(&interm[5 * interm_strd]);
            int16x8_t input_6 = vld1q_s16(&interm[6 * interm_strd]);
            int16x8_t input_7 = vld1q_s16(&interm[7 * interm_strd]);

            // load data from input row 8-11
            int16x8_t input_11 = vld1q_s16(&interm[(16 - 5) * interm_strd]);
            int16x8_t input_10 = vld1q_s16(&interm[(16 - 6) * interm_strd]);
            int16x8_t input_9 = vld1q_s16(&interm[(16 - 7) * interm_strd]);
            int16x8_t input_8 = vld1q_s16(&interm[(16 - 8) * interm_strd]);

            stg2_4 = vaddq_s16(input_4, input_11);
            stg2_5 = vaddq_s16(input_5, input_10);
            stg2_6 = vaddq_s16(input_6, input_9);
            stg2_7 = vaddq_s16(input_7, input_8);
            {
              int16x8_t stg2_11 = vsubq_s16(input_4, input_11);
              int16x8_t stg2_10 = vsubq_s16(input_5, input_10);
              int16x8_t stg2_9 = vsubq_s16(input_6, input_9);
              int16x8_t stg2_8 = vsubq_s16(input_7, input_8);

              vst1q_s16(&interm[(16 - 5) * interm_strd], stg2_11);
              vst1q_s16(&interm[(16 - 6) * interm_strd], stg2_10);
              vst1q_s16(&interm[(16 - 7) * interm_strd], stg2_9);
              vst1q_s16(&interm[(16 - 8) * interm_strd], stg2_8);
            }
          }
        }
        if (round) {
          half_round_shift_neon(&stg2_0, &stg2_1, &stg2_2, &stg2_3,
                                &stg2_4, &stg2_5, &stg2_6, &stg2_7);
        }
        {
          tran_low_t * temp;
          // -------------------------------------------------------------------
          // STAGE 3 BLOCK A, B : 0-7
          // -------------------------------------------------------------------
          int16x8_t stg3_0 = vaddq_s16(stg2_0, stg2_7);
          int16x8_t stg3_1 = vaddq_s16(stg2_1, stg2_6);
          int16x8_t stg3_2 = vaddq_s16(stg2_2, stg2_5);
          int16x8_t stg3_3 = vaddq_s16(stg2_3, stg2_4);

          int16x8_t stg3_7 = vsubq_s16(stg2_0, stg2_7);
          int16x8_t stg3_6 = vsubq_s16(stg2_1, stg2_6);
          int16x8_t stg3_5 = vsubq_s16(stg2_2, stg2_5);
          int16x8_t stg3_4 = vsubq_s16(stg2_3, stg2_4);

          // -------------------------------------------------------------------
          // STAGE 4 BLOCK A, B : 0-7
          // -------------------------------------------------------------------
          int16x8_t stg4_0 = vaddq_s16(stg3_0, stg3_3);
          int16x8_t stg4_1 = vaddq_s16(stg3_1, stg3_2);
          int16x8_t stg4_2 = vsubq_s16(stg3_1, stg3_2);
          int16x8_t stg4_3 = vsubq_s16(stg3_0, stg3_3);

          int16x8_t stg4_4 = stg3_4;
          int16x8_t stg4_5, stg4_6;
          int16x8_t stg4_7 = stg3_7;
          butterfly_sym(stg3_6, stg3_5, &stg4_6, &stg4_5);
          {
            // -----------------------------------------------------------------
            // STAGE 5 BLOCK A, B : 0-7
            // -----------------------------------------------------------------
            int16x8_t stg5_0, stg5_1, stg5_2, stg5_3;
            int16x8_t stg5_4 = vaddq_s16(stg4_4, stg4_5);
            int16x8_t stg5_5 = vsubq_s16(stg4_4, stg4_5);
            int16x8_t stg5_6 = vsubq_s16(stg4_7, stg4_6);
            int16x8_t stg5_7 = vaddq_s16(stg4_7, stg4_6);
            butterfly_sym(stg4_0, stg4_1, &stg5_0, &stg5_1);
            butterfly_std(stg4_3, stg4_2, &stg5_2, &stg5_3,
                          cospi_8_64, cospi_24_64);

            // -----------------------------------------------------------------
            // STORE OUTPUT BLOCK A : 0-3
            // -----------------------------------------------------------------
            STORE_TO_OUTPUT(stg5_0, 0, out, pass, temp);
            STORE_TO_OUTPUT(stg5_1, 16, out, pass, temp);
            STORE_TO_OUTPUT(stg5_2, 8, out, pass, temp);
            STORE_TO_OUTPUT(stg5_3, 24, out, pass, temp);
            {
              // ---------------------------------------------------------------
              // STAGE 6 BLOCK B : 4-7
              // ---------------------------------------------------------------
              int16x8_t stg6_4, stg6_5, stg6_6, stg6_7;
              butterfly_std(stg5_7, stg5_4, &stg6_4, &stg6_7,
                            cospi_4_64, cospi_28_64);
              butterfly_std(stg5_6, stg5_5, &stg6_5, &stg6_6,
                            cospi_20_64, cospi_12_64);

              // ---------------------------------------------------------------
              // STORE OUTPUT BLOCK B : 4-7
              // ---------------------------------------------------------------
              STORE_TO_OUTPUT(stg6_4, 4, out, pass, temp);
              STORE_TO_OUTPUT(stg6_5, 20, out, pass, temp);
              STORE_TO_OUTPUT(stg6_6, 12, out, pass, temp);
              STORE_TO_OUTPUT(stg6_7, 28, out, pass, temp);
            }
          }
        }
      }
      // -----------------------------------------------------------------------
      // STAGE 3,4,5,6 BLOCK C : 8-15
      // -----------------------------------------------------------------------
      {
        tran_low_t * locl;
        int16x8_t stg2_8 = vld1q_s16(&interm[8 * interm_strd]);
        int16x8_t stg2_9 = vld1q_s16(&interm[9 * interm_strd]);
        int16x8_t stg2_10 = vld1q_s16(&interm[10 * interm_strd]);
        int16x8_t stg2_11 = vld1q_s16(&interm[11 * interm_strd]);
        int16x8_t stg2_12 = vld1q_s16(&interm[12 * interm_strd]);
        int16x8_t stg2_13 = vld1q_s16(&interm[13 * interm_strd]);
        int16x8_t stg2_14 = vld1q_s16(&interm[14 * interm_strd]);
        int16x8_t stg2_15 = vld1q_s16(&interm[15 * interm_strd]);

        if (round) {
          half_round_shift_neon(&stg2_8, &stg2_9, &stg2_10, &stg2_11,
                                &stg2_12, &stg2_13, &stg2_14, &stg2_15);
        }
        {
          // -------------------------------------------------------------------
          // STAGE 3 BLOCK C : 8-15
          // -------------------------------------------------------------------
          int16x8_t stg3_8 = stg2_8;
          int16x8_t stg3_9 = stg2_9;
          int16x8_t stg3_14 = stg2_14;
          int16x8_t stg3_15 = stg2_15;
          int16x8_t stg3_10, stg3_11, stg3_12, stg3_13;
          butterfly_sym(stg2_13, stg2_10, &stg3_13, &stg3_10);
          butterfly_sym(stg2_12, stg2_11, &stg3_12, &stg3_11);
          {
            // -----------------------------------------------------------------
            // STAGE 4 BLOCK C : 8-15
            // -----------------------------------------------------------------
            int16x8_t stg4_8 = vaddq_s16(stg3_8, stg3_11);
            int16x8_t stg4_9 = vaddq_s16(stg3_9, stg3_10);
            int16x8_t stg4_10 = vsubq_s16(stg3_9, stg3_10);
            int16x8_t stg4_11 = vsubq_s16(stg3_8, stg3_11);
            int16x8_t stg4_12 = vsubq_s16(stg3_15, stg3_12);
            int16x8_t stg4_13 = vsubq_s16(stg3_14, stg3_13);
            int16x8_t stg4_14 = vaddq_s16(stg3_14, stg3_13);
            int16x8_t stg4_15 = vaddq_s16(stg3_15, stg3_12);

            // -----------------------------------------------------------------
            // STAGE 5 BLOCK C : 8-15
            // -----------------------------------------------------------------
            int16x8_t stg5_8 = stg4_8;
            int16x8_t stg5_11 = stg4_11;
            int16x8_t stg5_12 = stg4_12;
            int16x8_t stg5_15 = stg4_15;
            int16x8_t stg5_9, stg5_10, stg5_13, stg5_14;
            butterfly_std(stg4_14, stg4_9, &stg5_14, &stg5_9,
                          cospi_8_64, cospi_24_64);
            butterfly_std(stg4_10, stg4_13, &stg5_10, &stg5_13,
                          -cospi_24_64, -cospi_8_64);
            {
              // ---------------------------------------------------------------
              // STAGE 6 BLOCK C : 8-15
              // ---------------------------------------------------------------
              int16x8_t stg6_8 = vaddq_s16(stg5_8, stg5_9);
              int16x8_t stg6_9 = vsubq_s16(stg5_8, stg5_9);
              int16x8_t stg6_10 = vsubq_s16(stg5_11, stg5_10);
              int16x8_t stg6_11 = vaddq_s16(stg5_11, stg5_10);
              int16x8_t stg6_12 = vaddq_s16(stg5_12, stg5_13);
              int16x8_t stg6_13 = vsubq_s16(stg5_12, stg5_13);
              int16x8_t stg6_14 = vsubq_s16(stg5_15, stg5_14);
              int16x8_t stg6_15 = vaddq_s16(stg5_15, stg5_14);

              // ---------------------------------------------------------------
              // STAGE 7 BLOCK C : 8-15
              // ---------------------------------------------------------------
              int16x8_t stg7_8, stg7_9, stg7_10, stg7_11;
              int16x8_t stg7_12, stg7_13, stg7_14, stg7_15;
              butterfly_std(stg6_15, stg6_8, &stg7_8, &stg7_15,
                            cospi_2_64, cospi_30_64);
              butterfly_std(stg6_14, stg6_9, &stg7_9, &stg7_14,
                            cospi_18_64, cospi_14_64);
              butterfly_std(stg6_13, stg6_10, &stg7_10, &stg7_13,
                            cospi_10_64, cospi_22_64);
              butterfly_std(stg6_12, stg6_11, &stg7_11, &stg7_12,
                            cospi_26_64, cospi_6_64);

              // ---------------------------------------------------------------
              // STORE OUTPUT BLOCK C : 8-15
              // ---------------------------------------------------------------
              STORE_TO_OUTPUT(stg7_8, 2, out, pass, locl);
              STORE_TO_OUTPUT(stg7_9, 18, out, pass, locl);
              STORE_TO_OUTPUT(stg7_10, 10, out, pass, locl);
              STORE_TO_OUTPUT(stg7_11, 26, out, pass, locl);
              STORE_TO_OUTPUT(stg7_12, 6, out, pass, locl);
              STORE_TO_OUTPUT(stg7_13, 22, out, pass, locl);
              STORE_TO_OUTPUT(stg7_14, 14, out, pass, locl);
              STORE_TO_OUTPUT(stg7_15, 30, out, pass, locl);
            }
          }
        }
      }
      // -----------------------------------------------------------------------
      // STAGE 2,3,4,5,6,7,8 BLOCK D: 16-31
      // -----------------------------------------------------------------------
      {
        tran_low_t * locl;
        // Stage 2
        int16x8_t stg1_20 = vld1q_s16(&interm[20 * interm_strd]);
        int16x8_t stg1_21 = vld1q_s16(&interm[21 * interm_strd]);
        int16x8_t stg1_22 = vld1q_s16(&interm[22 * interm_strd]);
        int16x8_t stg1_23 = vld1q_s16(&interm[23 * interm_strd]);
        int16x8_t stg1_24 = vld1q_s16(&interm[24 * interm_strd]);
        int16x8_t stg1_25 = vld1q_s16(&interm[25 * interm_strd]);
        int16x8_t stg1_26 = vld1q_s16(&interm[26 * interm_strd]);
        int16x8_t stg1_27 = vld1q_s16(&interm[27 * interm_strd]);

        // ---------------------------------------------------------------------
        // STAGE 2 BLOCK D: 16-31
        // ---------------------------------------------------------------------
        int16x8_t stg2_20, stg2_21, stg2_22, stg2_23;
        int16x8_t stg2_24, stg2_25, stg2_26, stg2_27;
        butterfly_sym(stg1_27, stg1_20, &stg2_27, &stg2_20);
        butterfly_sym(stg1_26, stg1_21, &stg2_26, &stg2_21);
        butterfly_sym(stg1_25, stg1_22, &stg2_25, &stg2_22);
        butterfly_sym(stg1_24, stg1_23, &stg2_24, &stg2_23);

        if (round) {
          half_round_shift_neon(&stg2_20, &stg2_21, &stg2_22, &stg2_23,
                                &stg2_24, &stg2_25, &stg2_26, &stg2_27);
        }
        {
          int16x8_t stg2_16 = vld1q_s16(&interm[16 * interm_strd]);
          int16x8_t stg2_17 = vld1q_s16(&interm[17 * interm_strd]);
          int16x8_t stg2_18 = vld1q_s16(&interm[18 * interm_strd]);
          int16x8_t stg2_19 = vld1q_s16(&interm[19 * interm_strd]);
          int16x8_t stg2_28 = vld1q_s16(&interm[28 * interm_strd]);
          int16x8_t stg2_29 = vld1q_s16(&interm[29 * interm_strd]);
          int16x8_t stg2_30 = vld1q_s16(&interm[30 * interm_strd]);
          int16x8_t stg2_31 = vld1q_s16(&interm[31 * interm_strd]);

          if (round) {
            half_round_shift_neon(&stg2_16, &stg2_17, &stg2_18, &stg2_19,
                                  &stg2_28, &stg2_29, &stg2_30, &stg2_31);
          }
          {
            // -----------------------------------------------------------------
            // STAGE 3 BLOCK D: 16-31
            // -----------------------------------------------------------------
            int16x8_t stg3_16 = vaddq_s16(stg2_16, stg2_23);
            int16x8_t stg3_17 = vaddq_s16(stg2_17, stg2_22);
            int16x8_t stg3_18 = vaddq_s16(stg2_18, stg2_21);
            int16x8_t stg3_19 = vaddq_s16(stg2_19, stg2_20);
            int16x8_t stg3_20 = vsubq_s16(stg2_19, stg2_20);
            int16x8_t stg3_21 = vsubq_s16(stg2_18, stg2_21);
            int16x8_t stg3_22 = vsubq_s16(stg2_17, stg2_22);
            int16x8_t stg3_23 = vsubq_s16(stg2_16, stg2_23);

            int16x8_t stg3_24 = vsubq_s16(stg2_31, stg2_24);
            int16x8_t stg3_25 = vsubq_s16(stg2_30, stg2_25);
            int16x8_t stg3_26 = vsubq_s16(stg2_29, stg2_26);
            int16x8_t stg3_27 = vsubq_s16(stg2_28, stg2_27);
            int16x8_t stg3_28 = vaddq_s16(stg2_28, stg2_27);
            int16x8_t stg3_29 = vaddq_s16(stg2_29, stg2_26);
            int16x8_t stg3_30 = vaddq_s16(stg2_30, stg2_25);
            int16x8_t stg3_31 = vaddq_s16(stg2_31, stg2_24);

            // -----------------------------------------------------------------
            // STAGE 4 BLOCK D: 16-31
            // -----------------------------------------------------------------
            int16x8_t stg4_16 = stg3_16;
            int16x8_t stg4_17 = stg3_17;
            int16x8_t stg4_22 = stg3_22;
            int16x8_t stg4_23 = stg3_23;
            int16x8_t stg4_24 = stg3_24;
            int16x8_t stg4_25 = stg3_25;
            int16x8_t stg4_30 = stg3_30;
            int16x8_t stg4_31 = stg3_31;
            int16x8_t stg4_18, stg4_19, stg4_20, stg4_21;
            int16x8_t stg4_26, stg4_27, stg4_28, stg4_29;
            butterfly_std(stg3_29, stg3_18, &stg4_29, &stg4_18,
                          cospi_8_64, cospi_24_64);
            butterfly_std(stg3_28, stg3_19, &stg4_28, &stg4_19,
                          cospi_8_64, cospi_24_64);
            butterfly_std(stg3_20, stg3_27, &stg4_20, &stg4_27,
                          -cospi_24_64, -cospi_8_64);
            butterfly_std(stg3_21, stg3_26, &stg4_21, &stg4_26,
                          -cospi_24_64, -cospi_8_64);
            {
              // ---------------------------------------------------------------
              // STAGE 5 BLOCK D PART 1: 16-19 28-31
              // ---------------------------------------------------------------
              int16x8_t stg5_16 = vaddq_s16(stg4_16, stg4_19);
              int16x8_t stg5_17 = vaddq_s16(stg4_17, stg4_18);
              int16x8_t stg5_18 = vsubq_s16(stg4_17, stg4_18);
              int16x8_t stg5_19 = vsubq_s16(stg4_16, stg4_19);

              int16x8_t stg5_28 = vsubq_s16(stg4_31, stg4_28);
              int16x8_t stg5_29 = vsubq_s16(stg4_30, stg4_29);
              int16x8_t stg5_30 = vaddq_s16(stg4_30, stg4_29);
              int16x8_t stg5_31 = vaddq_s16(stg4_31, stg4_28);

              // ---------------------------------------------------------------
              // STAGE 6 BLOCK D PART 1: 16-19 28-31
              // ---------------------------------------------------------------
              int16x8_t stg6_16 = stg5_16;
              int16x8_t stg6_19 = stg5_19;
              int16x8_t stg6_28 = stg5_28;
              int16x8_t stg6_31 = stg5_31;
              int16x8_t stg6_17, stg6_18, stg6_29, stg6_30;
              butterfly_std(stg5_30, stg5_17, &stg6_30, &stg6_17,
                            cospi_4_64, cospi_28_64);
              butterfly_std(stg5_18, stg5_29, &stg6_18, &stg6_29,
                            -cospi_28_64, -cospi_4_64);
              {
                // -------------------------------------------------------------
                // STAGE 7 BLOCK D PART 1: 16-19 28-31
                // -------------------------------------------------------------
                int16x8_t stg7_16 = vaddq_s16(stg6_16, stg6_17);
                int16x8_t stg7_17 = vsubq_s16(stg6_16, stg6_17);

                int16x8_t stg7_18 = vsubq_s16(stg6_19, stg6_18);
                int16x8_t stg7_19 = vaddq_s16(stg6_19, stg6_18);

                int16x8_t stg7_28 = vaddq_s16(stg6_28, stg6_29);
                int16x8_t stg7_29 = vsubq_s16(stg6_28, stg6_29);

                int16x8_t stg7_30 = vsubq_s16(stg6_31, stg6_30);
                int16x8_t stg7_31 = vaddq_s16(stg6_31, stg6_30);

                // -------------------------------------------------------------
                // STAGE 8 BLOCK D PART 1: 16-19 28-31
                // -------------------------------------------------------------
                int16x8_t stg8_16, stg8_17, stg8_18, stg8_19;
                int16x8_t stg8_28, stg8_29, stg8_30, stg8_31;

                butterfly_std(stg7_31, stg7_16, &stg8_16, &stg8_31,
                              cospi_1_64, cospi_31_64);
                butterfly_std(stg7_30, stg7_17, &stg8_17, &stg8_30,
                              cospi_17_64, cospi_15_64);
                butterfly_std(stg7_29, stg7_18, &stg8_18, &stg8_29,
                              cospi_9_64, cospi_23_64);
                butterfly_std(stg7_28, stg7_19, &stg8_19, &stg8_28,
                              cospi_25_64, cospi_7_64);
                {
                  // -----------------------------------------------------------
                  // STAGE 5 BLOCK D PART 2 : 20-27
                  // -----------------------------------------------------------
                  int16x8_t stg5_20 = vsubq_s16(stg4_23, stg4_20);
                  int16x8_t stg5_21 = vsubq_s16(stg4_22, stg4_21);
                  int16x8_t stg5_22 = vaddq_s16(stg4_22, stg4_21);
                  int16x8_t stg5_23 = vaddq_s16(stg4_23, stg4_20);

                  int16x8_t stg5_24 = vaddq_s16(stg4_24, stg4_27);
                  int16x8_t stg5_25 = vaddq_s16(stg4_25, stg4_26);
                  int16x8_t stg5_26 = vsubq_s16(stg4_25, stg4_26);
                  int16x8_t stg5_27 = vsubq_s16(stg4_24, stg4_27);

                  // -----------------------------------------------------------
                  // STAGE 6 BLOCK D PART 2 : 20-27
                  // -----------------------------------------------------------
                  int16x8_t stg6_20 = stg5_20;
                  int16x8_t stg6_23 = stg5_23;
                  int16x8_t stg6_24 = stg5_24;
                  int16x8_t stg6_27 = stg5_27;
                  int16x8_t stg6_21, stg6_22, stg6_25, stg6_26;
                  butterfly_std(stg5_26, stg5_21, &stg6_26, &stg6_21,
                                cospi_20_64, cospi_12_64);
                  butterfly_std(stg5_22, stg5_25, &stg6_22, &stg6_25,
                                -cospi_12_64, -cospi_20_64);
                  {
                    // ---------------------------------------------------------
                    // STAGE 7 BLOCK D PART 2 : 20-27
                    // ---------------------------------------------------------
                    int16x8_t stg7_20 = vaddq_s16(stg6_20, stg6_21);
                    int16x8_t stg7_21 = vsubq_s16(stg6_20, stg6_21);

                    int16x8_t stg7_22 = vsubq_s16(stg6_23, stg6_22);
                    int16x8_t stg7_23 = vaddq_s16(stg6_23, stg6_22);

                    int16x8_t stg7_24 = vaddq_s16(stg6_24, stg6_25);
                    int16x8_t stg7_25 = vsubq_s16(stg6_24, stg6_25);

                    int16x8_t stg7_26 = vsubq_s16(stg6_27, stg6_26);
                    int16x8_t stg7_27 = vaddq_s16(stg6_27, stg6_26);

                    // ---------------------------------------------------------
                    // STAGE 8 BLOCK D PART 2 : 20-27
                    // ---------------------------------------------------------
                    int16x8_t stg8_20, stg8_21, stg8_22, stg8_23;
                    int16x8_t stg8_24, stg8_25, stg8_26, stg8_27;

                    butterfly_std(stg7_27, stg7_20, &stg8_20, &stg8_27,
                                  cospi_5_64, cospi_27_64);
                    butterfly_std(stg7_26, stg7_21, &stg8_21, &stg8_26,
                                  cospi_21_64, cospi_11_64);
                    butterfly_std(stg7_25, stg7_22, &stg8_22, &stg8_25,
                                  cospi_13_64, cospi_19_64);
                    butterfly_std(stg7_24, stg7_23, &stg8_23, &stg8_24,
                                  cospi_29_64, cospi_3_64);

                    // ---------------------------------------------------------
                    // STORE OUTPUT BLOCK D PART 1 : 16-23
                    // ---------------------------------------------------------
                    STORE_TO_OUTPUT(stg8_16, 1, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_17, 17, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_18, 9, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_19, 25, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_20, 5, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_21, 21, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_22, 13, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_23, 29, out, pass, locl);

                    // ---------------------------------------------------------
                    // STORE OUTPUT BLOCK D PART 2 : 24-31
                    // ---------------------------------------------------------
                    STORE_TO_OUTPUT(stg8_24, 3, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_25, 19, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_26, 11, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_27, 27, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_28, 7, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_29, 23, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_30, 15, out, pass, locl);
                    STORE_TO_OUTPUT(stg8_31, 31, out, pass, locl);
                  }
                }
              }
            }
          }
        }
      }
    }
    // Go to next pass
    // Set the round mode 1
    round = 1;
    // restore the output pointer to actual output
    out_strd = 32;
    out = output - 8 * out_strd;
    // Set the input pointer to intermediate memory
    in_strd = out_strd;
    in = output - 8 * in_strd;
  }

  // transpose whole 32x32
  {
    int col, row;

    for (col = 0; col < 32; col += 8)
      for (row = 0; row < 32; row += 8) {
        // Adjust input and output pointer
        // We have to move to next 32x8 block
        tran_low_t *sblk = output + col + row * 32;
        int sblk_strd = 32;

        // load data from input row 0-7
        int16x8_t input_0 = vld1q_s16(&sblk[0 * sblk_strd]);
        int16x8_t input_1 = vld1q_s16(&sblk[1 * sblk_strd]);
        int16x8_t input_2 = vld1q_s16(&sblk[2 * sblk_strd]);
        int16x8_t input_3 = vld1q_s16(&sblk[3 * sblk_strd]);
        int16x8_t input_4 = vld1q_s16(&sblk[4 * sblk_strd]);
        int16x8_t input_5 = vld1q_s16(&sblk[5 * sblk_strd]);
        int16x8_t input_6 = vld1q_s16(&sblk[6 * sblk_strd]);
        int16x8_t input_7 = vld1q_s16(&sblk[7 * sblk_strd]);

        transpose_8x8(&input_0, &input_1, &input_2, &input_3, &input_4,
                      &input_5, &input_6, &input_7);

        // Store data to rows 0-7
        vst1q_s16(&sblk[0 * sblk_strd], input_0);
        vst1q_s16(&sblk[1 * sblk_strd], input_1);
        vst1q_s16(&sblk[2 * sblk_strd], input_2);
        vst1q_s16(&sblk[3 * sblk_strd], input_3);
        vst1q_s16(&sblk[4 * sblk_strd], input_4);
        vst1q_s16(&sblk[5 * sblk_strd], input_5);
        vst1q_s16(&sblk[6 * sblk_strd], input_6);
        vst1q_s16(&sblk[7 * sblk_strd], input_7);
      }
  }
}


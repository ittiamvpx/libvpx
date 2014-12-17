/*
 *  Copyright (c) 2010 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include <arm_neon.h>
#include <assert.h>
#include <math.h>

#include "./vpx_config.h"
#include "./vp9_rtcd.h"

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_idct.h"

// -----------------------------------------------------------------------------
// Load 16 rows
// -----------------------------------------------------------------------------
static inline void load_input(int16x8_t* v_ip_row, const int16_t *ptr,
                              int stride) {
  v_ip_row[0] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[1] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[2] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[3] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[4] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[5] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[6] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[7] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[8] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[9] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[10] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[11] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[12] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[13] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[14] = vld1q_s16(ptr);
  ptr += stride;
  v_ip_row[15] = vld1q_s16(ptr);
  ptr += stride;
}

// -----------------------------------------------------------------------------
// Store 16 rows
// -----------------------------------------------------------------------------
static inline void store_output(int16x8_t* v_op_row, int16_t *ptr, int stride) {
  vst1q_s16(ptr, v_op_row[0]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[1]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[2]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[3]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[4]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[5]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[6]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[7]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[8]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[9]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[10]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[11]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[12]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[13]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[14]);
  ptr += stride;
  vst1q_s16(ptr, v_op_row[15]);
  ptr += stride;
}  // END OF store_output

// -----------------------------------------------------------------------------
// Multiply all inputs by 4
// -----------------------------------------------------------------------------
static inline void multiply_by_4(int16x8_t* v_input) {
  v_input[0] = vshlq_n_s16(v_input[0], 2);
  v_input[1] = vshlq_n_s16(v_input[1], 2);
  v_input[2] = vshlq_n_s16(v_input[2], 2);
  v_input[3] = vshlq_n_s16(v_input[3], 2);
  v_input[4] = vshlq_n_s16(v_input[4], 2);
  v_input[5] = vshlq_n_s16(v_input[5], 2);
  v_input[6] = vshlq_n_s16(v_input[6], 2);
  v_input[7] = vshlq_n_s16(v_input[7], 2);
  v_input[8] = vshlq_n_s16(v_input[8], 2);
  v_input[9] = vshlq_n_s16(v_input[9], 2);
  v_input[10] = vshlq_n_s16(v_input[10], 2);
  v_input[11] = vshlq_n_s16(v_input[11], 2);
  v_input[12] = vshlq_n_s16(v_input[12], 2);
  v_input[13] = vshlq_n_s16(v_input[13], 2);
  v_input[14] = vshlq_n_s16(v_input[14], 2);
  v_input[15] = vshlq_n_s16(v_input[15], 2);
}  // END OF multiply_by_4

// -----------------------------------------------------------------------------
// Calculate the input for the DCT transform
//  - Input and ouput variables cannot be  same.
// -----------------------------------------------------------------------------
static inline void calc_input_for_dct(int16x8_t* v_output, int16x8_t* v_input,
                                      int pass) {
  v_output[0] = vaddq_s16(v_input[0], v_input[15]);
  v_output[1] = vaddq_s16(v_input[1], v_input[14]);
  v_output[2] = vaddq_s16(v_input[2], v_input[13]);
  v_output[3] = vaddq_s16(v_input[3], v_input[12]);
  v_output[4] = vaddq_s16(v_input[4], v_input[11]);
  v_output[5] = vaddq_s16(v_input[5], v_input[10]);
  v_output[6] = vaddq_s16(v_input[6], v_input[9]);
  v_output[7] = vaddq_s16(v_input[7], v_input[8]);

  v_output[8] = vsubq_s16(v_input[7], v_input[8]);
  v_output[9] = vsubq_s16(v_input[6], v_input[9]);
  v_output[10] = vsubq_s16(v_input[5], v_input[10]);
  v_output[11] = vsubq_s16(v_input[4], v_input[11]);
  v_output[12] = vsubq_s16(v_input[3], v_input[12]);
  v_output[13] = vsubq_s16(v_input[2], v_input[13]);
  v_output[14] = vsubq_s16(v_input[1], v_input[14]);
  v_output[15] = vsubq_s16(v_input[0], v_input[15]);
  if (pass == 0) {
    multiply_by_4(v_output);
  }
}
// END OF calc_input_for_dct

// -----------------------------------------------------------------------------
// Rounds and shifts value in input registers(for pass=1)
// -----------------------------------------------------------------------------
static inline void round_shift(int16x8_t* v_output, int16x8_t* v_input,
                               int tx_type) {
  {
    const int16x8_t v_ones = vdupq_n_s16(1);
    v_input[0] = vaddq_s16(v_input[0], v_ones);
    v_input[1] = vaddq_s16(v_input[1], v_ones);
    v_input[2] = vaddq_s16(v_input[2], v_ones);
    v_input[3] = vaddq_s16(v_input[3], v_ones);
    v_input[4] = vaddq_s16(v_input[4], v_ones);
    v_input[5] = vaddq_s16(v_input[5], v_ones);
    v_input[6] = vaddq_s16(v_input[6], v_ones);
    v_input[7] = vaddq_s16(v_input[7], v_ones);
    v_input[8] = vaddq_s16(v_input[8], v_ones);
    v_input[9] = vaddq_s16(v_input[9], v_ones);
    v_input[10] = vaddq_s16(v_input[10], v_ones);
    v_input[11] = vaddq_s16(v_input[11], v_ones);
    v_input[12] = vaddq_s16(v_input[12], v_ones);
    v_input[13] = vaddq_s16(v_input[13], v_ones);
    v_input[14] = vaddq_s16(v_input[14], v_ones);
    v_input[15] = vaddq_s16(v_input[15], v_ones);
  }
  if (tx_type != DCT_DCT) {
    uint16x8_t v_sign;
    v_sign = vshrq_n_u16((uint16x8_t) v_input[0], 15);
    v_input[0] = vaddq_s16(v_input[0], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[1], 15);
    v_input[1] = vaddq_s16(v_input[1], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[2], 15);
    v_input[2] = vaddq_s16(v_input[2], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[3], 15);
    v_input[3] = vaddq_s16(v_input[3], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[4], 15);
    v_input[4] = vaddq_s16(v_input[4], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[5], 15);
    v_input[5] = vaddq_s16(v_input[5], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[6], 15);
    v_input[6] = vaddq_s16(v_input[6], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[7], 15);
    v_input[7] = vaddq_s16(v_input[7], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[8], 15);
    v_input[8] = vaddq_s16(v_input[8], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[9], 15);
    v_input[9] = vaddq_s16(v_input[9], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[10], 15);
    v_input[10] = vaddq_s16(v_input[10], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[11], 15);
    v_input[11] = vaddq_s16(v_input[11], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[12], 15);
    v_input[12] = vaddq_s16(v_input[12], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[13], 15);
    v_input[13] = vaddq_s16(v_input[13], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[14], 15);
    v_input[14] = vaddq_s16(v_input[14], (int16x8_t) v_sign);
    v_sign = vshrq_n_u16((uint16x8_t) v_input[15], 15);
    v_input[15] = vaddq_s16(v_input[15], (int16x8_t) v_sign);
  }
  v_output[0] = vshrq_n_s16(v_input[0], 2);
  v_output[1] = vshrq_n_s16(v_input[1], 2);
  v_output[2] = vshrq_n_s16(v_input[2], 2);
  v_output[3] = vshrq_n_s16(v_input[3], 2);
  v_output[4] = vshrq_n_s16(v_input[4], 2);
  v_output[5] = vshrq_n_s16(v_input[5], 2);
  v_output[6] = vshrq_n_s16(v_input[6], 2);
  v_output[7] = vshrq_n_s16(v_input[7], 2);
  v_output[8] = vshrq_n_s16(v_input[8], 2);
  v_output[9] = vshrq_n_s16(v_input[9], 2);
  v_output[10] = vshrq_n_s16(v_input[10], 2);
  v_output[11] = vshrq_n_s16(v_input[11], 2);
  v_output[12] = vshrq_n_s16(v_input[12], 2);
  v_output[13] = vshrq_n_s16(v_input[13], 2);
  v_output[14] = vshrq_n_s16(v_input[14], 2);
  v_output[15] = vshrq_n_s16(v_input[15], 2);
}
// END of round_shift

// -----------------------------------------------------------------------------
static inline void do_butterfly_no_coeffs(int16x8_t ip1, int16x8_t ip2,
                                          int16x8_t* sum, int16x8_t* diff) {
  *diff = vsubq_s16(ip1, ip2);
  *sum = vaddq_s16(ip1, ip2);
}
// END of do_butterfly_no_coeffs

// -----------------------------------------------------------------------------
static inline void do_butterfly_symmetric_coeffs(int16x8_t ip1, int16x8_t ip2,
                                                 tran_high_t constant,
                                                 int16x8_t* op1,
                                                 int16x8_t* op2) {
  const int16x4_t v_constant = vdup_n_s16(constant);
  const int16x8_t v_sum = vaddq_s16(ip1, ip2);
  const int16x8_t v_diff = vsubq_s16(ip1, ip2);
  const int32x4_t v_mul_temp_1 = vmull_s16(vget_high_s16(v_sum), v_constant);
  const int32x4_t v_mul_temp_2 = vmull_s16(vget_low_s16(v_sum), v_constant);
  const int32x4_t v_mul_temp_3 = vmull_s16(vget_high_s16(v_diff), v_constant);
  const int32x4_t v_mul_temp_4 = vmull_s16(vget_low_s16(v_diff), v_constant);
  const int16x4_t v_temp1 = vqrshrn_n_s32(v_mul_temp_1, 14);
  const int16x4_t v_temp2 = vqrshrn_n_s32(v_mul_temp_2, 14);
  const int16x4_t v_temp3 = vqrshrn_n_s32(v_mul_temp_3, 14);
  const int16x4_t v_temp4 = vqrshrn_n_s32(v_mul_temp_4, 14);
  *op1 = vcombine_s16(v_temp2, v_temp1);
  *op2 = vcombine_s16(v_temp4, v_temp3);
}
// END OF do_butterfly_symmetric_coeffs

// -----------------------------------------------------------------------------
// BUTTERFLY for DCT
// -----------------------------------------------------------------------------
static inline void do_butterfly_dct(int16x4_t ip1, int16x4_t ip2,
                                    int16x4_t ip3, int16x4_t ip4,
                                    tran_high_t first_constant,
                                    tran_high_t second_constant,
                                    int16x8_t* op1, int16x8_t* op2) {
  const int16x4_t v_constant_1 = vdup_n_s16(first_constant);
  const int16x4_t v_constant_2 = vdup_n_s16(second_constant);
  const int32x4_t v_mul_temp_1 = vmull_s16(ip1, v_constant_1);
  const int32x4_t v_mul_temp_2 = vmull_s16(ip2, v_constant_1);
  const int32x4_t v_mul_temp_3 = vmull_s16(ip3, v_constant_2);
  const int32x4_t v_mul_temp_4 = vmull_s16(ip4, v_constant_2);
  const int32x4_t v_mul_temp_5 = vmull_s16(ip1, v_constant_2);
  const int32x4_t v_mul_temp_6 = vmull_s16(ip2, v_constant_2);
  const int32x4_t v_mul_temp_7 = vmull_s16(ip3, v_constant_1);
  const int32x4_t v_mul_temp_8 = vmull_s16(ip4, v_constant_1);
  const int32x4_t v_diff_1 = vsubq_s32(v_mul_temp_1, v_mul_temp_3);
  const int32x4_t v_diff_2 = vsubq_s32(v_mul_temp_2, v_mul_temp_4);
  const int32x4_t v_sum_1 = vaddq_s32(v_mul_temp_5, v_mul_temp_7);
  const int32x4_t v_sum_2 = vaddq_s32(v_mul_temp_6, v_mul_temp_8);
  const int16x4_t v_temp1 = vqrshrn_n_s32(v_diff_1, 14);
  const int16x4_t v_temp2 = vqrshrn_n_s32(v_diff_2, 14);
  const int16x4_t v_temp3 = vqrshrn_n_s32(v_sum_1, 14);
  const int16x4_t v_temp4 = vqrshrn_n_s32(v_sum_2, 14);
  *op1 = vcombine_s16(v_temp1, v_temp2);
  *op2 = vcombine_s16(v_temp3, v_temp4);
}
// END OF do_butterfly_dct

// -----------------------------------------------------------------------------
// BUTTERFLY for DST
// -----------------------------------------------------------------------------
static inline void do_butterfly_dst(int16x4_t ip1, int16x4_t ip2,
                                    int16x4_t ip3, int16x4_t ip4,
                                    tran_high_t first_constant,
                                    tran_high_t second_constant,
                                    int32x4_t* op1, int32x4_t* op2,
                                    int32x4_t* op3, int32x4_t* op4) {
  const int16x4_t v_constant_1 = vdup_n_s16(first_constant);
  const int16x4_t v_constant_2 = vdup_n_s16(second_constant);
  const int32x4_t v_mul_temp_1 = vmull_s16(ip1, v_constant_1);
  const int32x4_t v_mul_temp_2 = vmull_s16(ip2, v_constant_1);
  const int32x4_t v_mul_temp_3 = vmull_s16(ip3, v_constant_2);
  const int32x4_t v_mul_temp_4 = vmull_s16(ip4, v_constant_2);
  const int32x4_t v_mul_temp_5 = vmull_s16(ip1, v_constant_2);
  const int32x4_t v_mul_temp_6 = vmull_s16(ip2, v_constant_2);
  const int32x4_t v_mul_temp_7 = vmull_s16(ip3, v_constant_1);
  const int32x4_t v_mul_temp_8 = vmull_s16(ip4, v_constant_1);
  *op3 = vsubq_s32(v_mul_temp_1, v_mul_temp_3);
  *op4 = vsubq_s32(v_mul_temp_2, v_mul_temp_4);
  *op1 = vaddq_s32(v_mul_temp_5, v_mul_temp_7);
  *op2 = vaddq_s32(v_mul_temp_6, v_mul_temp_8);
}
// END OF do_butterfly_dst

// -----------------------------------------------------------------------------
// BUTTERFLY and ROUND SHIFT for DST
// -----------------------------------------------------------------------------
static inline void do_butterfly_without_coeffs_and_roundshift(int32x4_t ip1,
                                                              int32x4_t ip2,
                                                              int32x4_t ip3,
                                                              int32x4_t ip4,
                                                              int16x8_t* op1,
                                                              int16x8_t* op2) {
  const int32x4_t v_sum_1 = vaddq_s32(ip1, ip3);
  const int32x4_t v_sum_2 = vaddq_s32(ip2, ip4);
  const int32x4_t v_diff_1 = vsubq_s32(ip1, ip3);
  const int32x4_t v_diff_2 = vsubq_s32(ip2, ip4);
  const int16x4_t v_temp1 = vqrshrn_n_s32(v_sum_1, 14);
  const int16x4_t v_temp2 = vqrshrn_n_s32(v_sum_2, 14);
  const int16x4_t v_temp3 = vqrshrn_n_s32(v_diff_1, 14);
  const int16x4_t v_temp4 = vqrshrn_n_s32(v_diff_2, 14);
  *op1 = vcombine_s16(v_temp1, v_temp2);
  *op2 = vcombine_s16(v_temp3, v_temp4);
}
// END OF do_butterfly_without_coeffs_and_roundshift

// -----------------------------------------------------------------------------
// Transpose a 8x8 16bit data matrix.
// -----------------------------------------------------------------------------
static inline void transpose_8x8(int16x8_t* v_output, int16x8_t* v_input) {
  const int16x8x2_t r01_s16 = vtrnq_s16(v_input[0], v_input[1]);
  const int16x8x2_t r23_s16 = vtrnq_s16(v_input[2], v_input[3]);
  const int16x8x2_t r45_s16 = vtrnq_s16(v_input[4], v_input[5]);
  const int16x8x2_t r67_s16 = vtrnq_s16(v_input[6], v_input[7]);

  const int32x4x2_t r01_s32 = vtrnq_s32(vreinterpretq_s32_s16(r01_s16.val[0]),
                                        vreinterpretq_s32_s16(r23_s16.val[0]));
  const int32x4x2_t r23_s32 = vtrnq_s32(vreinterpretq_s32_s16(r01_s16.val[1]),
                                        vreinterpretq_s32_s16(r23_s16.val[1]));
  const int32x4x2_t r45_s32 = vtrnq_s32(vreinterpretq_s32_s16(r45_s16.val[0]),
                                        vreinterpretq_s32_s16(r67_s16.val[0]));
  const int32x4x2_t r67_s32 = vtrnq_s32(vreinterpretq_s32_s16(r45_s16.val[1]),
                                        vreinterpretq_s32_s16(r67_s16.val[1]));
  v_output[0] = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r01_s32.val[0]),
                   vget_low_s32(r45_s32.val[0])));
  v_output[2] = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r01_s32.val[1]),
                   vget_low_s32(r45_s32.val[1])));
  v_output[1] = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r23_s32.val[0]),
                   vget_low_s32(r67_s32.val[0])));
  v_output[3] = vreinterpretq_s16_s32(
      vcombine_s32(vget_low_s32(r23_s32.val[1]),
                   vget_low_s32(r67_s32.val[1])));
  v_output[4] = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r01_s32.val[0]),
                   vget_high_s32(r45_s32.val[0])));
  v_output[6] = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r01_s32.val[1]),
                   vget_high_s32(r45_s32.val[1])));
  v_output[5] = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r23_s32.val[0]),
                   vget_high_s32(r67_s32.val[0])));
  v_output[7] = vreinterpretq_s16_s32(
      vcombine_s32(vget_high_s32(r23_s32.val[1]),
                   vget_high_s32(r67_s32.val[1])));
}

// -----------------------------------------------------------------------------
// dct_single_pass
//   - calculates DCT transform for a single pass(16 rows, 8 cols )
// -----------------------------------------------------------------------------
static void dct_single_pass(int16x8_t* v_input, int16x8_t* v_output) {
  int16x8_t v_s[8];
  int16x8_t v_x[4];
  int16x8_t v_step[8];

  // stage 1
  do_butterfly_no_coeffs(v_input[0], v_input[7], &v_s[0], &v_s[7]);
  do_butterfly_no_coeffs(v_input[1], v_input[6], &v_s[1], &v_s[6]);
  do_butterfly_no_coeffs(v_input[2], v_input[5], &v_s[2], &v_s[5]);
  do_butterfly_no_coeffs(v_input[3], v_input[4], &v_s[3], &v_s[4]);
  // fdct4(step, step);
  do_butterfly_no_coeffs(v_s[0], v_s[3], &v_x[0], &v_x[3]);
  do_butterfly_no_coeffs(v_s[1], v_s[2], &v_x[1], &v_x[2]);

  // t0 = (x0 + x1) * cospi_16_64;
  // t1 = (x0 - x1) * cospi_16_64;
  // out[0] = fdct_round_shift(t0);
  // out[8] = fdct_round_shift(t1);
  do_butterfly_symmetric_coeffs(v_x[0], v_x[1],
                                cospi_16_64,
                                &v_output[0], &v_output[8]);
  // t2 = x3 * cospi_8_64  + x2 * cospi_24_64;
  // t3 = x3 * cospi_24_64 - x2 * cospi_8_64;
  // out[4] = fdct_round_shift(t2);
  // out[12] = fdct_round_shift(t3);
  do_butterfly_dct(vget_low_s16(v_x[3]), vget_high_s16(v_x[3]),
                   vget_low_s16(v_x[2]), vget_high_s16(v_x[2]),
                   cospi_24_64, cospi_8_64,
                   &v_output[12], &v_output[4]);

  //  Stage 2

  // t0 = (s6 - s5) * cospi_16_64;
  // t1 = (s6 + s5) * cospi_16_64;
  // t2 = fdct_round_shift(t0);
  // t3 = fdct_round_shift(t1);
  do_butterfly_symmetric_coeffs(v_s[6], v_s[5],
                                cospi_16_64,
                                &v_s[5], &v_s[6]);

  //  Stage 3
  do_butterfly_no_coeffs(v_s[4], v_s[6], &v_x[0], &v_x[1]);
  do_butterfly_no_coeffs(v_s[7], v_s[5], &v_x[3], &v_x[2]);

  // Stage 4

  // t0 = x0 * cospi_28_64 + x3 *   cospi_4_64;
  // t3 = x3 * cospi_28_64 + x0 *  -cospi_4_64;
  // out[2] = fdct_round_shift(t0);
  // out[14] = fdct_round_shift(t3);
  do_butterfly_dct(vget_low_s16(v_x[3]), vget_high_s16(v_x[3]),
                   vget_low_s16(v_x[0]), vget_high_s16(v_x[0]),
                   cospi_28_64, cospi_4_64,
                   &v_output[14], &v_output[2]);

  // t1 = x1 * cospi_12_64 + x2 *  cospi_20_64;
  // t2 = x2 * cospi_12_64 + x1 * -cospi_20_64;
  // out[6] = fdct_round_shift(t2);
  // out[10] = fdct_round_shift(t1);
  do_butterfly_dct(vget_low_s16(v_x[2]), vget_high_s16(v_x[2]),
                   vget_low_s16(v_x[1]), vget_high_s16(v_x[1]),
                   cospi_12_64,  cospi_20_64,
                   &v_output[6], &v_output[10]);

  // step 2

  // temp1 = (step1[5] - step1[2]) * cospi_16_64;
  // temp2 = (step1[5] + step1[2]) * cospi_16_64;
  // step2[2] = fdct_round_shift(temp1);
  // step2[5] = fdct_round_shift(temp2);
  do_butterfly_symmetric_coeffs(v_input[13], v_input[10],
                                cospi_16_64,
                                &v_s[5], &v_s[2]);

  // temp2 = (step1[4] - step1[3]) * cospi_16_64;
  // temp1 = (step1[4] + step1[3]) * cospi_16_64;
  // step2[3] = fdct_round_shift(temp2);
  // step2[4] = fdct_round_shift(temp1);
  do_butterfly_symmetric_coeffs(v_input[12], v_input[11],
                                cospi_16_64,
                                &v_s[4], &v_s[3]);

  // step 3
  do_butterfly_no_coeffs(v_input[8], v_s[3], &v_s[0], &v_x[1]);
  do_butterfly_no_coeffs(v_input[9], v_s[2], &v_s[1], &v_x[0]);
  do_butterfly_no_coeffs(v_input[14], v_s[5], &v_s[6], &v_x[3]);
  do_butterfly_no_coeffs(v_input[15], v_s[4], &v_s[7], &v_x[2]);

  // step 4
  // temp1 = step3[1] *  -cospi_8_64 + step3[6] * cospi_24_64;
  // temp2 = step3[1] * cospi_24_64 + step3[6] *  cospi_8_64
  // step2[1] = fdct_round_shift(temp1);
  // step2[6] = fdct_round_shift(temp2);
  do_butterfly_dct(vget_low_s16(v_s[6]), vget_high_s16(v_s[6]),
                   vget_low_s16(v_s[1]), vget_high_s16(v_s[1]),
                   cospi_24_64, cospi_8_64,
                   &v_s[1], &v_s[6]);

  // temp2 = step3[2] * cospi_24_64 + step3[5] *  cospi_8_64;
  // temp1 = step3[2] * cospi_8_64 - step3[5] * cospi_24_64;
  // step2[2] = fdct_round_shift(temp2);
  // step2[5] = fdct_round_shift(temp1);
  do_butterfly_dct(vget_low_s16(v_x[0]), vget_high_s16(v_x[0]),
                   vget_low_s16(v_x[3]), vget_high_s16(v_x[3]),
                   cospi_8_64, cospi_24_64,
                   &v_s[5], &v_s[2]);

  // step 5
  do_butterfly_no_coeffs(v_s[0], v_s[1], &v_step[0], &v_step[1]);
  do_butterfly_no_coeffs(v_x[1], v_s[2], &v_step[2], &v_step[3]);
  do_butterfly_no_coeffs(v_x[2], v_s[5], &v_step[5], &v_step[4]);
  do_butterfly_no_coeffs(v_s[7], v_s[6], &v_step[7], &v_step[6]);

  // step 6

  // temp1 = step1[1] * -cospi_18_64 + step1[6] * cospi_14_64
  // out[7] = fdct_round_shift(temp1);
  // temp2 = step1[1] * cospi_14_64 + step1[6] * cospi_18_64;
  // out[9] = fdct_round_shift(temp2);
  do_butterfly_dct(vget_low_s16(v_step[6]), vget_high_s16(v_step[6]),
                   vget_low_s16(v_step[1]), vget_high_s16(v_step[1]),
                   cospi_14_64, cospi_18_64,
                   &v_output[7], &v_output[9]);

  // temp2 = step1[0] *  -cospi_2_64 + step1[7] * cospi_30_64;
  // out[15] = fdct_round_shift(temp2);
  // temp1 = step1[0] * cospi_30_64 + step1[7] *  cospi_2_64;
  // out[1] = fdct_round_shift(temp1);
  do_butterfly_dct(vget_low_s16(v_step[7]), vget_high_s16(v_step[7]),
                   vget_low_s16(v_step[0]), vget_high_s16(v_step[0]),
                   cospi_30_64, cospi_2_64,
                   &v_output[15], &v_output[1]);

  // temp1 = step1[3] * -cospi_26_64 + step1[4] *  cospi_6_64;
  // out[3] = fdct_round_shift(temp1);
  // temp2 = step1[3] *  cospi_6_64 + step1[4] * cospi_26_64;
  // out[13] = fdct_round_shift(temp2)
  do_butterfly_dct(vget_low_s16(v_step[4]), vget_high_s16(v_step[4]),
                   vget_low_s16(v_step[3]), vget_high_s16(v_step[3]),
                   cospi_6_64, cospi_26_64,
                   &v_output[3], &v_output[13]);

  // temp2 = step1[2] * -cospi_10_64 + step1[5] * cospi_22_64;
  // out[11] = fdct_round_shift(temp2);
  // temp1 = step1[2] * cospi_22_64 + step1[5] * cospi_10_64;
  // out[5] = fdct_round_shift(temp1);
  do_butterfly_dct(vget_low_s16(v_step[5]), vget_high_s16(v_step[5]),
                   vget_low_s16(v_step[2]), vget_high_s16(v_step[2]),
                   cospi_22_64, cospi_10_64,
                   &v_output[11], &v_output[5]);
}

// --------------------------------------------------------------------------
// dst_single_pass
//   - calculates DST transform for a single pass(16 rows, 8 cols )
// --------------------------------------------------------------------------
static void dst_single_pass(int16x8_t* v_input, int16x8_t* v_output) {
  int32x4_t v_s[16];
  int16x8_t v_x[16];
  // stage 1 for x0,x1,x2,x3,x8,x9,x10,x11

  // s0 = x0 * cospi_1_64  + x1 * cospi_31_64;
  // s1 = x0 * cospi_31_64 - x1 * cospi_1_64;
  do_butterfly_dst(vget_low_s16(v_input[15]), vget_high_s16(v_input[15]),
                   vget_low_s16(v_input[0]), vget_high_s16(v_input[0]),
                   cospi_31_64, cospi_1_64,
                   &v_s[0], &v_s[4], &v_s[1], &v_s[5]);

  // s8 = x8 * cospi_17_64 + x9 * cospi_15_64;
  // s9 = x8 * cospi_15_64 - x9 * cospi_17_64;
  do_butterfly_dst(vget_low_s16(v_input[7]), vget_high_s16(v_input[7]),
                   vget_low_s16(v_input[8]), vget_high_s16(v_input[8]),
                   cospi_15_64, cospi_17_64,
                   &v_s[8], &v_s[12], &v_s[9], &v_s[13]);

  // x9 = fdct_round_shift(s1 - s9) ; x1 = fdct_round_shift(s1 + s9);
  do_butterfly_without_coeffs_and_roundshift(v_s[1], v_s[5], v_s[9], v_s[13],
                                             &v_x[1], &v_x[9]);

  // x8 = fdct_round_shift(s0 - s8), ;x0 = fdct_round_shift(s0 + s8);
  do_butterfly_without_coeffs_and_roundshift(v_s[0], v_s[4], v_s[8], v_s[12],
                                             &v_x[0], &v_x[8]);

  // s2 = x2 * cospi_5_64  + x3 * cospi_27_64;
  // s3 = x2 * cospi_27_64 - x3 * cospi_5_64;
  do_butterfly_dst(vget_low_s16(v_input[13]), vget_high_s16(v_input[13]),
                   vget_low_s16(v_input[2]), vget_high_s16(v_input[2]),
                   cospi_27_64, cospi_5_64,
                   &v_s[2], &v_s[6], &v_s[3], &v_s[7]);

  // s10 = x10 * cospi_21_64 + x11 * cospi_11_64;
  // s11 = x10 * cospi_11_64 - x11 * cospi_21_64;
  do_butterfly_dst(vget_low_s16(v_input[5]), vget_high_s16(v_input[5]),
                   vget_low_s16(v_input[10]), vget_high_s16(v_input[10]),
                   cospi_11_64, cospi_21_64,
                   &v_s[10], &v_s[14], &v_s[11], &v_s[15]);

  // x3 = fdct_round_shift(s3 + s11);,x11 = fdct_round_shift(s3 - s11);
  do_butterfly_without_coeffs_and_roundshift(v_s[3], v_s[7], v_s[11], v_s[15],
                                             &v_x[3], &v_x[11]);

  // x2 = fdct_round_shift(s2 + s10);,  x10 = fdct_round_shift(s2 - s10);
  do_butterfly_without_coeffs_and_roundshift(v_s[2], v_s[6], v_s[10], v_s[14],
                                             &v_x[2], &v_x[10]);

  // s4 = x4 * cospi_9_64  + x5 * cospi_23_64;
  // s5 = x4 * cospi_23_64 - x5 * cospi_9_64;
  do_butterfly_dst(vget_low_s16(v_input[11]), vget_high_s16(v_input[11]),
                   vget_low_s16(v_input[4]), vget_high_s16(v_input[4]),
                   cospi_23_64, cospi_9_64,
                   &v_s[4], &v_s[8], &v_s[5], &v_s[9]);

  // s12 = x12 * cospi_25_64 + x13 * cospi_7_64;
  // s13 = x12 * cospi_7_64  - x13 * cospi_25_64;
  do_butterfly_dst(vget_low_s16(v_input[3]), vget_high_s16(v_input[3]),
                   vget_low_s16(v_input[12]), vget_high_s16(v_input[12]),
                   cospi_7_64, cospi_25_64,
                   &v_s[12], &v_s[14], &v_s[13], &v_s[15]);

  //  x4 = fdct_round_shift(s4 + s12);  x12 = fdct_round_shift(s4 - s12);
  do_butterfly_without_coeffs_and_roundshift(v_s[4], v_s[8], v_s[12], v_s[14],
                                             &v_x[4], &v_x[12]);

  // x5 = fdct_round_shift(s5 + s13), x13 = fdct_round_shift(s5 - s13);
  do_butterfly_without_coeffs_and_roundshift(v_s[5], v_s[9], v_s[13], v_s[15],
                                             &v_x[5], &v_x[13]);

  // s6 = x6 * cospi_13_64 + x7 * cospi_19_64;
  // s7 = x6 * cospi_19_64 - x7 * cospi_13_64;
  do_butterfly_dst(vget_low_s16(v_input[9]), vget_high_s16(v_input[9]),
                   vget_low_s16(v_input[6]), vget_high_s16(v_input[6]),
                   cospi_19_64, cospi_13_64,
                   &v_s[6], &v_s[10], &v_s[7], &v_s[11]);

  // s14 = x14 * cospi_29_64 + x15 * cospi_3_64;
  // s15 = x14 * cospi_3_64  - x15 * cospi_29_64;
  do_butterfly_dst(vget_low_s16(v_input[1]), vget_high_s16(v_input[1]),
                   vget_low_s16(v_input[14]), vget_high_s16(v_input[14]),
                   cospi_3_64, cospi_29_64,
                   &v_s[14], &v_s[12], &v_s[15], &v_s[13]);

  // x6 = fdct_round_shift(s6 + s14), x14 = fdct_round_shift(s6 - s14);
  do_butterfly_without_coeffs_and_roundshift(v_s[6], v_s[10], v_s[14], v_s[12],
                                             &v_x[6], &v_x[14]);

  // x7 = fdct_round_shift(s7 + s15),  x15 = fdct_round_shift(s7 - s15);
  do_butterfly_without_coeffs_and_roundshift(v_s[7], v_s[11], v_s[15], v_s[13],
                                             &v_x[7], &v_x[15]);

  // stage 2 for x8,x9,x10,x11, x12,x13,x14,x15

  // s8 =    x8 * cospi_4_64   + x9 * cospi_28_64;
  // s9 =    x8 * cospi_28_64  - x9 * cospi_4_64;
  do_butterfly_dst(vget_low_s16(v_x[8]), vget_high_s16(v_x[8]),
                   vget_low_s16(v_x[9]), vget_high_s16(v_x[9]),
                   cospi_28_64, cospi_4_64,
                   &v_s[8], &v_s[10], &v_s[9], &v_s[11]);

  // s12 = - x12 * cospi_28_64 + x13 * cospi_4_64;
  // s13 =   x12 * cospi_4_64  + x13 * cospi_28_64;
  do_butterfly_dst(vget_low_s16(v_x[13]), vget_high_s16(v_x[13]),
                   vget_low_s16(v_x[12]), vget_high_s16(v_x[12]),
                   cospi_4_64, cospi_28_64,
                   &v_s[13], &v_s[15], &v_s[12], &v_s[14]);

  // x8 = fdct_round_shift(s8 + s12);   x12 = fdct_round_shift(s8 - s12);
  do_butterfly_without_coeffs_and_roundshift(v_s[8], v_s[10], v_s[12], v_s[14],
                                             &v_x[8], &v_x[12]);

  // x9 = fdct_round_shift(s9 + s13);, x13 = fdct_round_shift(s9 - s13);
  do_butterfly_without_coeffs_and_roundshift(v_s[9], v_s[11], v_s[13], v_s[15],
                                             &v_x[9], &v_x[13]);

  // s10 =   x10 * cospi_20_64 + x11 * cospi_12_64;
  // s11 =   x10 * cospi_12_64 - x11 * cospi_20_64;
  do_butterfly_dst(vget_low_s16(v_x[10]), vget_high_s16(v_x[10]),
                   vget_low_s16(v_x[11]), vget_high_s16(v_x[11]),
                   cospi_12_64, cospi_20_64,
                   &v_s[10], &v_s[12], &v_s[11], &v_s[13]);

  // s14 = - x14 * cospi_12_64 + x15 * cospi_20_64;
  // s15 =   x14 * cospi_20_64 + x15 * cospi_12_64;
  do_butterfly_dst(vget_low_s16(v_x[15]), vget_high_s16(v_x[15]),
                   vget_low_s16(v_x[14]), vget_high_s16(v_x[14]),
                   cospi_20_64, cospi_12_64,
                   &v_s[15], &v_s[9], &v_s[14], &v_s[8]);

  // x10 = fdct_round_shift(s10 + s14);  x14 = fdct_round_shift(s10 - s14);
  do_butterfly_without_coeffs_and_roundshift(v_s[10], v_s[12], v_s[14], v_s[8],
                                             &v_x[10], &v_x[14]);

  // x11 = fdct_round_shift(s11 + s15);  x15 = fdct_round_shift(s11 - s15);
  do_butterfly_without_coeffs_and_roundshift(v_s[11], v_s[13], v_s[15], v_s[9],
                                             &v_x[11], &v_x[15]);

  // stage 3 for x8,x9,x10,x11, x12,x13,x14,x15

  // s12 = x12 * cospi_8_64  + x13 * cospi_24_64;
  // s13 = x12 * cospi_24_64 - x13 * cospi_8_64;
  do_butterfly_dst(vget_low_s16(v_x[12]), vget_high_s16(v_x[12]),
                   vget_low_s16(v_x[13]), vget_high_s16(v_x[13]),
                   cospi_24_64, cospi_8_64,
                   &v_s[12], &v_s[8], &v_s[13], &v_s[9]);

  // s14 = - x14 * cospi_24_64 + x15 * cospi_8_64;
  // s15 =   x14 * cospi_8_64  + x15 * cospi_24_64
  do_butterfly_dst(vget_low_s16(v_x[15]), vget_high_s16(v_x[15]),
                   vget_low_s16(v_x[14]), vget_high_s16(v_x[14]),
                   cospi_8_64, cospi_24_64,
                   &v_s[15], &v_s[11], &v_s[14], &v_s[10]);

  // x12 = fdct_round_shift(s12 + s14);  x14 = fdct_round_shift(s12 - s14);
  do_butterfly_without_coeffs_and_roundshift(v_s[12], v_s[8], v_s[14], v_s[10],
                                             &v_output[2], &v_x[14]);

  // x13 = fdct_round_shift(s13 + s15),  x15 = fdct_round_shift(s13 - s15);
  do_butterfly_without_coeffs_and_roundshift(v_s[13], v_s[9], v_s[15], v_s[11],
                                             &v_x[13], &v_x[15]);
  v_output[13] = vnegq_s16(v_x[13]);

  // x8 = s8 + s10;
  // x9 = s9 + s11;
  // x10 = s8 - s10;
  // x11 = s9 - s11;
  do_butterfly_no_coeffs(v_x[8], v_x[10], &v_x[12], &v_x[13]);
  v_output[1] = vnegq_s16(v_x[12]);
  do_butterfly_no_coeffs(v_x[9], v_x[11], &v_output[14], &v_x[12]);
  // stage 4 for x8,x9,x10,x11, x12,x13,x14,x15

  // s10 = cospi_16_64 * (x10 + x11);
  // s11 = cospi_16_64 * (- x10 + x11);
  // x10 = fdct_round_shift(s10);
  // x11 = fdct_round_shift(s11);
  do_butterfly_symmetric_coeffs(v_x[12], v_x[13], cospi_16_64,
                                &v_output[6],&v_output[9]);

  // s14 = (- cospi_16_64) * (x14 + x15);
  // s15 = cospi_16_64 * (x14 - x15);
  // x14 = fdct_round_shift(s14);
  // x15 = fdct_round_shift(s15);
  do_butterfly_symmetric_coeffs(v_x[15], v_x[14], -cospi_16_64,
                                &v_output[5], &v_output[10]);

  // stage 2 for x0 -x7
  do_butterfly_no_coeffs(v_x[0], v_x[4], &v_x[0], &v_x[8]);
  do_butterfly_no_coeffs(v_x[1], v_x[5], &v_x[1], &v_x[9]);
  do_butterfly_no_coeffs(v_x[2], v_x[6], &v_x[2], &v_x[10]);
  do_butterfly_no_coeffs(v_x[3], v_x[7], &v_x[3], &v_x[11]);

  // stage 3 for x0 - x7

  // s4 = x4 * cospi_8_64  + x5 * cospi_24_64;
  // s5 = x4 * cospi_24_64 - x5 * cospi_8_64;
  do_butterfly_dst(vget_low_s16(v_x[8]), vget_high_s16(v_x[8]),
                   vget_low_s16(v_x[9]), vget_high_s16(v_x[9]),
                   cospi_24_64, cospi_8_64,
                   &v_s[4], &v_s[8], &v_s[5], &v_s[9]);

  // s6 = - x6 * cospi_24_64 + x7 * cospi_8_64;
  // s7 =   x6 * cospi_8_64  + x7 * cospi_24_64;
  do_butterfly_dst(vget_low_s16(v_x[11]), vget_high_s16(v_x[11]),
                   vget_low_s16(v_x[10]), vget_high_s16(v_x[10]),
                   cospi_8_64, cospi_24_64,
                   &v_s[7], &v_s[11], &v_s[6], &v_s[10]);

  // x4 = fdct_round_shift(s4 + s6),x6 = fdct_round_shift(s4 - s6);
  do_butterfly_without_coeffs_and_roundshift(v_s[4], v_s[8], v_s[6], v_s[10],
                                             &v_x[4], &v_x[6]);

  // x5 = fdct_round_shift(s5 + s7) x7 = fdct_round_shift(s5 - s7);
  do_butterfly_without_coeffs_and_roundshift(v_s[5], v_s[9], v_s[7], v_s[11],
                                             &v_output[12], &v_x[7]);
  v_output[3] = vnegq_s16(v_x[4]);
  // x0 = s0 + s2;
  // x1 = s1 + s3;
  // x2 = s0 - s2;
  // x3 = s1 - s3;
  do_butterfly_no_coeffs(v_x[0], v_x[2], &v_output[0], &v_x[12]);
  do_butterfly_no_coeffs(v_x[1], v_x[3], &v_x[1], &v_x[13]);
  v_output[15] = vnegq_s16(v_x[1]);

  // stage 4 for x0 -x7

  // s6 = cospi_16_64 * (x6 + x7);
  // s7 = cospi_16_64 * (- x6 + x7);
  // x6 = fdct_round_shift(s6);
  // x7 = fdct_round_shift(s7);
  do_butterfly_symmetric_coeffs(v_x[7], v_x[6], cospi_16_64,
                                &v_output[4], &v_output[11]);

  // s2 = (- cospi_16_64) * (x2 + x3);
  // s3 = cospi_16_64 * (x2 - x3);
  // x2 = fdct_round_shift(s2);
  // x3 = fdct_round_shift(s3)
  do_butterfly_symmetric_coeffs(v_x[13], v_x[12], -cospi_16_64,
                                &v_output[7],  &v_output[8]);
}

//------------------------------------------------------------------------------
// BLOCK A = rows 1 to 8 cols 1 to 8
// BLOCK B = rows 9 to 16 cols 1 to 8
// BLOCK C = rows 1 to 8 cols 9 to 16
// BLOCK D = rows 9 to 16 cols 9 to 16
// dct_single_pass/dst_single_pass process 16 rows and 8 columns in a pass
// Output buffer is also used as intermediate buffer
// Code flow:
//  - Call dct_single_pass/dst_single_pass for BLOCK A and B
//  - Call dct_single_pass/dst_single_pass for BLOCK C and D
//  - Transpose the intermediate outputs of BLOCK B and D
//  - Call dct_single_pass/dst_single_pass for BLOCK B and D
//  - Transpose the result and store the final output for BLOCK B and D
//  - Transpose the intermediate outputs of BLOCK A and C
//  - Call dct_single_pass/dst_single_pass for BLOCK A and C
//  - Transpose the result and store the final output for BLOCK A and C
//------------------------------------------------------------------------------
void vp9_fdct16x16_neon(const int16_t *input, tran_low_t *output, int stride) {

  int16x8_t v_input_vals[16];
  int16x8_t v_temp_buffer_1[16];
  int16x8_t v_temp_buffer_2[16];
  int16x8_t v_temp_buffer_3[16];
  const int16_t *ip = input;
  int16_t *op = output;
  // load 16 rows and 8 columns(1-8)
  load_input(v_temp_buffer_1, ip, stride);

  // calculate input for pass = 0
  calc_input_for_dct(v_input_vals, v_temp_buffer_1, 0);
  // DCT SINGLE PASS
  dct_single_pass(v_input_vals, v_temp_buffer_1);
  ip = input + 8;

  // load 16 rows and 8 columns(9 - 16)
  load_input(v_temp_buffer_2, ip, stride);
  calc_input_for_dct(v_input_vals, v_temp_buffer_2, 0);
  // DCT SINGLE PASS for 16 rows and 8 columns(9 - 16)
  dct_single_pass(v_input_vals, v_temp_buffer_2);

  transpose_8x8(&v_temp_buffer_3[8], &v_temp_buffer_2[8]);
  transpose_8x8(&v_temp_buffer_3[0], &v_temp_buffer_1[8]);
  // second pass for rows(1 - 16) and cols (1 -  8)
  round_shift(v_temp_buffer_3, v_temp_buffer_3, DCT_DCT);
  calc_input_for_dct(v_input_vals, v_temp_buffer_3, 1);
  dct_single_pass(v_input_vals, v_temp_buffer_3);
  transpose_8x8(&v_temp_buffer_2[8], &v_temp_buffer_3[8]);
  transpose_8x8(&v_temp_buffer_1[8], &v_temp_buffer_3[0]);

  // second pass for rows(1 - 16) and cols (9 -  16)
  transpose_8x8(&v_temp_buffer_3[8], &v_temp_buffer_2[0]);
  transpose_8x8(&v_temp_buffer_3[0], &v_temp_buffer_1[0]);
  round_shift(v_temp_buffer_3, v_temp_buffer_3, DCT_DCT);
  calc_input_for_dct(v_input_vals, v_temp_buffer_3, 1);
  dct_single_pass(v_input_vals, v_temp_buffer_3);
  transpose_8x8(&v_temp_buffer_2[0], &v_temp_buffer_3[8]);
  transpose_8x8(&v_temp_buffer_1[0], &v_temp_buffer_3[0]);

  store_output(v_temp_buffer_1, op, 16);
  op = output + 8;
  store_output(v_temp_buffer_2, op, 16);
}

void vp9_fht16x16_neon(const int16_t *input, tran_low_t *output, int stride,
                       int tx_type) {
  if (tx_type == DCT_DCT) {
    vp9_fdct16x16_neon(input, output, stride);
  } else {
    int16x8_t v_input_vals[16];
    int16x8_t v_temp_buffer_1[16];
    int16x8_t v_temp_buffer_2[16];
    int16x8_t v_temp_buffer_3[16];
    const int16_t *ip = input;
    int16_t *op = output;
    // load 16 rows and 8 columns(1-8)
    load_input(v_temp_buffer_1, ip, stride);

    if (DCT_ADST != tx_type) {
      multiply_by_4(v_temp_buffer_1);
      dst_single_pass(v_temp_buffer_1, v_temp_buffer_1);
    } else {
      // calculate input for pass = 0
      calc_input_for_dct(v_input_vals, v_temp_buffer_1, 0);
      // DCT SINGLE PASS
      dct_single_pass(v_input_vals, v_temp_buffer_1);
    }

    ip = input + 8;
    // load 16 rows and 8 columns(9 - 16)
    load_input(v_temp_buffer_2, ip, stride);
    if (DCT_ADST != tx_type) {
      multiply_by_4(v_temp_buffer_2);
      dst_single_pass(v_temp_buffer_2, v_temp_buffer_2);
    } else {
      // calculate input for pass = 0
      calc_input_for_dct(v_input_vals, v_temp_buffer_2, 0);
      // DCT SINGLE PASS
      dct_single_pass(v_input_vals, v_temp_buffer_2);
    }

    transpose_8x8(&v_temp_buffer_3[8], &v_temp_buffer_2[8]);
    transpose_8x8(&v_temp_buffer_3[0], &v_temp_buffer_1[8]);
    // second pass for rows(1 - 16) and cols (1 -  8)
    round_shift(v_temp_buffer_3, v_temp_buffer_3, tx_type);
    if (ADST_ADST == tx_type || DCT_ADST == tx_type) {
      dst_single_pass(v_temp_buffer_3, v_temp_buffer_3);
    } else {
      calc_input_for_dct(v_input_vals, v_temp_buffer_3, 1);
      dct_single_pass(v_input_vals, v_temp_buffer_3);
    }
    transpose_8x8(&v_temp_buffer_2[8], &v_temp_buffer_3[8]);
    transpose_8x8(&v_temp_buffer_1[8], &v_temp_buffer_3[0]);

    // second pass for rows(1 - 16) and cols (9 -  16)
    transpose_8x8(&v_temp_buffer_3[8], &v_temp_buffer_2[0]);
    transpose_8x8(&v_temp_buffer_3[0], &v_temp_buffer_1[0]);

    round_shift(v_temp_buffer_3, v_temp_buffer_3, tx_type);
    if (ADST_ADST == tx_type || DCT_ADST == tx_type) {
      dst_single_pass(v_temp_buffer_3, v_temp_buffer_3);
    } else {
      calc_input_for_dct(v_input_vals, v_temp_buffer_3, 1);
      dct_single_pass(v_input_vals, v_temp_buffer_3);
    }
    transpose_8x8(&v_temp_buffer_2[0], &v_temp_buffer_3[8]);
    transpose_8x8(&v_temp_buffer_1[0], &v_temp_buffer_3[0]);

    store_output(v_temp_buffer_1, op, 16);
    op = output + 8;
    store_output(v_temp_buffer_2, op, 16);
  }
}

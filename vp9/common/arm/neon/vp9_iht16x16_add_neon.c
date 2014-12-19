/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */
#include <arm_neon.h>

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_common.h"

// -----------------------------------------------------------------------------
// Store 16 rows
// -----------------------------------------------------------------------------
static inline void store_output(uint8x8_t* v_op_row, uint8_t*ptr, int stride) {
  vst1_u8(ptr, v_op_row[0]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[1]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[2]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[3]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[4]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[5]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[6]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[7]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[8]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[9]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[10]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[11]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[12]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[13]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[14]);
  ptr += stride;
  vst1_u8(ptr, v_op_row[15]);
  ptr += stride;
}  // END OF store_output

// -----------------------------------------------------------------------------
// Right shift all inputs by 6
// -----------------------------------------------------------------------------
static inline void right_shift_by_6(int16x8_t* v_input) {
  v_input[0] = vrshrq_n_s16(v_input[0], 6);
  v_input[1] = vrshrq_n_s16(v_input[1], 6);
  v_input[2] = vrshrq_n_s16(v_input[2], 6);
  v_input[3] = vrshrq_n_s16(v_input[3], 6);
  v_input[4] = vrshrq_n_s16(v_input[4], 6);
  v_input[5] = vrshrq_n_s16(v_input[5], 6);
  v_input[6] = vrshrq_n_s16(v_input[6], 6);
  v_input[7] = vrshrq_n_s16(v_input[7], 6);
  v_input[8] = vrshrq_n_s16(v_input[8], 6);
  v_input[9] = vrshrq_n_s16(v_input[9], 6);
  v_input[10] = vrshrq_n_s16(v_input[10], 6);
  v_input[11] = vrshrq_n_s16(v_input[11], 6);
  v_input[12] = vrshrq_n_s16(v_input[12], 6);
  v_input[13] = vrshrq_n_s16(v_input[13], 6);
  v_input[14] = vrshrq_n_s16(v_input[14], 6);
  v_input[15] = vrshrq_n_s16(v_input[15], 6);
}  // END OF right_shift_by_6


// -----------------------------------------------------------------------------
static inline void do_butterfly_no_coeffs(int16x8_t ip1, int16x8_t ip2,
                                          int16x8_t* sum, int16x8_t* diff) {
  *diff = vsubq_s16(ip1, ip2);
  *sum = vaddq_s16(ip1, ip2);
}

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

// -----------------------------------------------------------------------------
// BUTTERFLY for DST
// -----------------------------------------------------------------------------
static inline void do_butterfly_dst(int16x8_t ip1, int16x8_t ip2,
                                    tran_high_t first_constant,
                                    tran_high_t second_constant,
                                    int32x4_t* op1, int32x4_t* op2,
                                    int32x4_t* op3, int32x4_t* op4) {
  const int16x4_t v_constant_1 = vdup_n_s16(first_constant);
  const int16x4_t v_constant_2 = vdup_n_s16(second_constant);
  const int32x4_t v_mul_temp_1 = vmull_s16(vget_low_s16(ip1), v_constant_1);
  const int32x4_t v_mul_temp_2 = vmull_s16(vget_high_s16(ip1), v_constant_1);
  const int32x4_t v_mul_temp_3 = vmull_s16(vget_low_s16(ip2), v_constant_2);
  const int32x4_t v_mul_temp_4 = vmull_s16(vget_high_s16(ip2), v_constant_2);
  const int32x4_t v_mul_temp_5 = vmull_s16(vget_low_s16(ip1), v_constant_2);
  const int32x4_t v_mul_temp_6 = vmull_s16(vget_high_s16(ip1), v_constant_2);
  const int32x4_t v_mul_temp_7 = vmull_s16(vget_low_s16(ip2), v_constant_1);
  const int32x4_t v_mul_temp_8 = vmull_s16(vget_high_s16(ip2), v_constant_1);
  *op3 = vsubq_s32(v_mul_temp_1, v_mul_temp_3);
  *op4 = vsubq_s32(v_mul_temp_2, v_mul_temp_4);
  *op1 = vaddq_s32(v_mul_temp_5, v_mul_temp_7);
  *op2 = vaddq_s32(v_mul_temp_6, v_mul_temp_8);
}

// -----------------------------------------------------------------------------
// BUTTERFLY and ROUND SHIFT for DST
// -----------------------------------------------------------------------------
static inline void do_dual_butterfly_no_coeffs(int32x4_t ip1,
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


//---------------------------------------------------------------------------
// Performs ADST for a 8x16 buffer
// This function should be called twice for full 16x16 transform
//
// vp9_iadst16x16_256_add_neon_single_pass(const int16_t *src,
//                                        int16_t *temp_buffer,
//                                        int  do_adding,
//                                        void *dest,
//                                        int dest_stride)//
//
//
// The output pointer *dest points to
// int16_t* when processed for row transform
// uint8_t* when processed for column transform

void vp9_iadst16x16_256_add_neon_single_pass(const int16_t *src,
                                             int16_t *temp_buffer,
                                             int do_adding, void *dest,
                                             int dest_stride) {

  int32x4_t v_s[16];
  int16x8_t v_input[16];
  uint8x8_t v_output[16];
  uint16x8_t v_out_temp;
  int16x8_t v_x[16];
  const int16_t *ptr = src;
  (void) temp_buffer;

  v_input[0]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[8]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[1]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[9]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[2]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[10] = vld1q_s16(ptr);
  ptr += 8;
  v_input[3]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[11] = vld1q_s16(ptr);
  ptr += 8;
  v_input[4]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[12] = vld1q_s16(ptr);
  ptr += 8;
  v_input[5]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[13] = vld1q_s16(ptr);
  ptr += 8;
  v_input[6]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[14] = vld1q_s16(ptr);
  ptr += 8;
  v_input[7]  = vld1q_s16(ptr);
  ptr += 8;
  v_input[15]  = vld1q_s16(ptr);

  transpose_8x8(&v_input[0], &v_input[0]);
  transpose_8x8(&v_input[8], &v_input[8]);

  // stage 1 for x0,x1,x2,x3,x8,x9,x10 and x11

  // s0 = x0 * cospi_1_64  + x1 * cospi_31_64;
  // s1 = x0 * cospi_31_64 - x1 * cospi_1_64;
  do_butterfly_dst(v_input[15], v_input[0], cospi_31_64, cospi_1_64,
                   &v_s[0], &v_s[4], &v_s[1], &v_s[5]);

  // s8 = x8 * cospi_17_64 + x9 * cospi_15_64;
  // s9 = x8 * cospi_15_64 - x9 * cospi_17_64;
  do_butterfly_dst(v_input[7], v_input[8], cospi_15_64, cospi_17_64,
                   &v_s[8], &v_s[12], &v_s[9], &v_s[13]);

  // x0 = dct_const_round_shift(s0 + s8);
  // x8 = dct_const_round_shift(s0 - s8);
  do_dual_butterfly_no_coeffs(v_s[0], v_s[4], v_s[8], v_s[12], &v_x[0],
                              &v_x[8]);

  // x1 = dct_const_round_shift(s1 + s9);
  // x9 = dct_const_round_shift(s1 - s9);
  do_dual_butterfly_no_coeffs(v_s[1], v_s[5], v_s[9], v_s[13],
                              &v_x[1], &v_x[9]);

  // s2 = x2 * cospi_5_64  + x3 * cospi_27_64;
  // s3 = x2 * cospi_27_64 - x3 * cospi_5_64;
  do_butterfly_dst(v_input[13], v_input[2], cospi_27_64, cospi_5_64,
                   &v_s[2], &v_s[6], &v_s[3], &v_s[7]);

  // s10 = x10 * cospi_21_64 + x11 * cospi_11_64;
  // s11 = x10 * cospi_11_64 - x11 * cospi_21_64;
  do_butterfly_dst(v_input[5], v_input[10], cospi_11_64, cospi_21_64,
                   &v_s[10], &v_s[14], &v_s[11], &v_s[15]);

  // x10 = dct_const_round_shift(s2 - s10);
  // x2  = dct_const_round_shift(s2 + s10);
  do_dual_butterfly_no_coeffs(v_s[2], v_s[6], v_s[10], v_s[14],
                              &v_x[2], &v_x[10]);

  // x11 = dct_const_round_shift(s3 - s11);
  // x3  = dct_const_round_shift(s3 + s11);
  do_dual_butterfly_no_coeffs(v_s[3], v_s[7], v_s[11], v_s[15],
                              &v_x[3], &v_x[11]);

  // stage 1 for x4,x5,x6,x7,x12,x13,x14 and x15

  // s4 = x4 * cospi_9_64  + x5 * cospi_23_64;
  // s5 = x4 * cospi_23_64 - x5 * cospi_9_64;
  do_butterfly_dst(v_input[11], v_input[4], cospi_23_64, cospi_9_64,
                   &v_s[4], &v_s[8], &v_s[5], &v_s[9]);

  // s12 = x12 * cospi_25_64 + x13 * cospi_7_64;
  // s13 = x12 * cospi_7_64  - x13 * cospi_25_64;
  do_butterfly_dst(v_input[3], v_input[12], cospi_7_64, cospi_25_64,
                   &v_s[12], &v_s[14], &v_s[13], &v_s[15]);

  // x4  = dct_const_round_shift(s4 + s12);
  // x12 = dct_const_round_shift(s4 - s12);
  do_dual_butterfly_no_coeffs(v_s[4], v_s[8], v_s[12], v_s[14],
                              &v_x[4], &v_x[12]);

  // x5  = dct_const_round_shift(s5 + s13);
  // x13 = dct_const_round_shift(s5 - s13);
  do_dual_butterfly_no_coeffs(v_s[5], v_s[9], v_s[13], v_s[15],
                              &v_x[5], &v_x[13]);

  // s6 = x6 * cospi_13_64 + x7 * cospi_19_64;
  // s7 = x6 * cospi_19_64 - x7 * cospi_13_64;
  do_butterfly_dst(v_input[9], v_input[6], cospi_19_64, cospi_13_64,
                   &v_s[6], &v_s[10], &v_s[7], &v_s[11]);

  // s14 = x14 * cospi_29_64 + x15 * cospi_3_64;
  // s15 = x14 * cospi_3_64  - x15 * cospi_29_64;
  do_butterfly_dst(v_input[1], v_input[14], cospi_3_64, cospi_29_64,
                   &v_s[14], &v_s[12], &v_s[15], &v_s[13]);

  // x6  = dct_const_round_shift(s6 + s14);
  // x14 = dct_const_round_shift(s6 - s14);
  do_dual_butterfly_no_coeffs(v_s[6], v_s[10], v_s[14], v_s[12],
                              &v_x[6], &v_x[14]);

  // x7  = dct_const_round_shift(s7 + s15);
  // x15 = dct_const_round_shift(s7 - s15);
  do_dual_butterfly_no_coeffs(v_s[7], v_s[11], v_s[15], v_s[13],
                              &v_x[7], &v_x[15]);

  // stage 2 for x8,x9,x10,x11,x12,x13,x14 and x15

  // s8 =  x8 * cospi_4_64   + x9 * cospi_28_64;
  // s9 =  x8 * cospi_28_64  - x9 * cospi_4_64;
  do_butterfly_dst(v_x[8], v_x[9], cospi_28_64, cospi_4_64,
                   &v_s[8], &v_s[10], &v_s[9], &v_s[11]);

  // s12 = -x12 * cospi_28_64 + x13 * cospi_4_64;
  // s13 =  x12 * cospi_4_64  + x13 * cospi_28_64;
  do_butterfly_dst(v_x[13], v_x[12], cospi_4_64, cospi_28_64,
                   &v_s[13], &v_s[15], &v_s[12], &v_s[14]);
  // x8  = dct_const_round_shift(s8 + s12);
  // x12 = dct_const_round_shift(s8 - s12);
  do_dual_butterfly_no_coeffs(v_s[8], v_s[10], v_s[12], v_s[14],
                              &v_x[8], &v_x[12]);

  // x9  = dct_const_round_shift(s9 + s13);
  // x13 = dct_const_round_shift(s9 - s13);
  do_dual_butterfly_no_coeffs(v_s[9], v_s[11], v_s[13], v_s[15],
                              &v_x[9], &v_x[13]);

  // s10 =   x10 * cospi_20_64 + x11 * cospi_12_64;
  // s11 =   x10 * cospi_12_64 - x11 * cospi_20_64;
  do_butterfly_dst(v_x[10], v_x[11], cospi_12_64, cospi_20_64,
                   &v_s[10], &v_s[8], &v_s[11], &v_s[9]);

  // s14 = -x14 * cospi_12_64 + x15 * cospi_20_64;
  // s15 =  x14 * cospi_20_64 + x15 * cospi_12_64;
  do_butterfly_dst(v_x[15], v_x[14], cospi_20_64, cospi_12_64,
                   &v_s[15], &v_s[13], &v_s[14], &v_s[12]);

  // x10 = dct_const_round_shift(s10 + s14);
  // x14 = dct_const_round_shift(s10 - s14);
  do_dual_butterfly_no_coeffs(v_s[10], v_s[8], v_s[14], v_s[12],
                              &v_x[10], &v_x[14]);

  // x11 = dct_const_round_shift(s11 + s15);
  // x15 = dct_const_round_shift(s11 - s15);
  do_dual_butterfly_no_coeffs(v_s[11], v_s[9], v_s[15], v_s[13],
                              &v_x[11], &v_x[15]);

  // stage 3 for x8,x9,x10,x11,x12,x13,x14 and x15

  // s12 = x12 * cospi_8_64  + x13 * cospi_24_64;
  // s13 = x12 * cospi_24_64 - x13 * cospi_8_64;
  do_butterfly_dst(v_x[12], v_x[13], cospi_24_64, cospi_8_64,
                   &v_s[12], &v_s[8], &v_s[13], &v_s[9]);

  // s14 = -x14 * cospi_24_64 + x15 * cospi_8_64;
  // s15 =  x14 * cospi_8_64  + x15 * cospi_24_64;
  do_butterfly_dst(v_x[15], v_x[14], cospi_8_64, cospi_24_64,
                   &v_s[15], &v_s[11], &v_s[14], &v_s[10]);

  // x12 = dct_const_round_shift(s12 + s14);
  // x14 = dct_const_round_shift(s12 - s14);
  do_dual_butterfly_no_coeffs(v_s[12], v_s[8], v_s[14], v_s[10],
                              &v_x[12], &v_x[14]);

  // x13 = dct_const_round_shift(s13 + s15);
  // x15 = dct_const_round_shift(s13 - s15);
  do_dual_butterfly_no_coeffs(v_s[13], v_s[9], v_s[15], v_s[11],
                              &v_x[13], &v_x[15]);

  // x8 = s8 + s10;
  // x9 = s9 + s11;
  // x10 = s8 - s10;
  // x11 = s9 - s11;
  do_butterfly_no_coeffs(v_x[8], v_x[10], &v_x[8], &v_x[10]);
  do_butterfly_no_coeffs(v_x[9], v_x[11], &v_x[9], &v_x[11]);

  // stage 4 for x8,x9,x10,x11,x12,x13,x14 and x15

  // s10 = cospi_16_64 * ( x10 + x11);
  // s11 = cospi_16_64 * (-x10 + x11);
  // x10 = dct_const_round_shift(s10);
  // x11 = dct_const_round_shift(s11);
  do_butterfly_symmetric_coeffs(v_x[11], v_x[10], cospi_16_64,
                                &v_x[10], &v_x[11]);

  // s14 = -cospi_16_64 * (x14 + x15);
  // s15 =  cospi_16_64 * (x14 - x15);
  // x14 = dct_const_round_shift(s14);
  // x15 = dct_const_round_shift(s15);
  do_butterfly_symmetric_coeffs(v_x[15], v_x[14], -cospi_16_64,
                                &v_x[14], &v_x[15]);

  //  stage 2 for x0,x1,x2,x3,x4,x5,x6 and x7

  // x0 = s0 + s4;
  // x1 = s1 + s5;
  // x2 = s2 + s6;
  // x3 = s3 + s7;
  // x4 = s0 - s4;
  // x5 = s1 - s5;
  // x6 = s2 - s6;
  // x7 = s3 - s7;

  do_butterfly_no_coeffs(v_x[0], v_x[4], &v_x[0], &v_x[4]);
  do_butterfly_no_coeffs(v_x[1], v_x[5], &v_x[1], &v_x[5]);
  do_butterfly_no_coeffs(v_x[2], v_x[6], &v_x[2], &v_x[6]);
  do_butterfly_no_coeffs(v_x[3], v_x[7], &v_x[3], &v_x[7]);

  // stage 3 for x0,x1,x2,x3,x4,x5,x6 and x7

  // s4 = x4 * cospi_8_64  + x5 * cospi_24_64;
  // s5 = x4 * cospi_24_64 - x5 * cospi_8_64;
  do_butterfly_dst(v_x[4], v_x[5], cospi_24_64, cospi_8_64,
                   &v_s[4], &v_s[8], &v_s[5], &v_s[9]);

  // s6 = -x6 * cospi_24_64 + x7 * cospi_8_64;
  // s7 =  x6 * cospi_8_64  + x7 * cospi_24_64;
  do_butterfly_dst(v_x[7], v_x[6], cospi_8_64, cospi_24_64,
                   &v_s[7], &v_s[11], &v_s[6], &v_s[10]);

  // x4 = dct_const_round_shift(s4 + s6);
  // x6 = dct_const_round_shift(s4 - s6);
  do_dual_butterfly_no_coeffs(v_s[4], v_s[8], v_s[6], v_s[10],
                              &v_x[4], &v_x[6]);

  // x5 = dct_const_round_shift(s5 + s7);
  // x7 = dct_const_round_shift(s5 - s7);
  do_dual_butterfly_no_coeffs(v_s[5], v_s[9], v_s[7], v_s[11],
                              &v_x[5], &v_x[7]);

  // x0 = s0 + s2;
  // x1 = s1 + s3;
  // x2 = s0 - s2;
  // x3 = s1 - s3;
  do_butterfly_no_coeffs(v_x[0], v_x[2], &v_x[0], &v_x[2]);
  do_butterfly_no_coeffs(v_x[1], v_x[3], &v_x[1], &v_x[3]);
  // stage 4 for for x0,x1,x2,x3,x4,x5,x6 and x7

  // s6 = cospi_16_64 * (x6 + x7);
  // s7 = cospi_16_64 * (- x6 + x7);
  // x6 = dct_const_round_shift(s6);
  // x7 = dct_const_round_shift(s7);
  do_butterfly_symmetric_coeffs(v_x[7], v_x[6], cospi_16_64, &v_x[6], &v_x[7]);

  // s2 = -cospi_16_64 * (x2 + x3);
  // s3 =  cospi_16_64 * (x2 - x3);
  // x2 = dct_const_round_shift(s2);
  // x3 = dct_const_round_shift(s3);
  do_butterfly_symmetric_coeffs(v_x[3], v_x[2], -cospi_16_64, &v_x[2], &v_x[3]);

  v_x[8] = vnegq_s16(v_x[8]);
  v_x[4] = vnegq_s16(v_x[4]);
  v_x[13] = vnegq_s16(v_x[13]);
  v_x[1] = vnegq_s16(v_x[1]);
  if (do_adding) {
    uint8_t* ptr = (uint8_t*) dest;
    v_output[0] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[1] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[2] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[3] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[4] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[5] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[6] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[7] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[8] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[9] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[10] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[11] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[12] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[13] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[14] = vld1_u8(ptr);
    ptr += dest_stride;
    v_output[15] = vld1_u8(ptr);
    //
    right_shift_by_6(v_x);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[0], v_output[0]);
    v_output[0] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[8], v_output[1]);
    v_output[1] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[12], v_output[2]);
    v_output[2] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[4], v_output[3]);
    v_output[3] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[6], v_output[4]);
    v_output[4] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[14], v_output[5]);
    v_output[5] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[10], v_output[6]);
    v_output[6] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[2], v_output[7]);
    v_output[7] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[3], v_output[8]);
    v_output[8] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[11], v_output[9]);
    v_output[9] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[15], v_output[10]);
    v_output[10] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[7], v_output[11]);
    v_output[11] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[5], v_output[12]);
    v_output[12] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[13], v_output[13]);
    v_output[13] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[9], v_output[14]);
    v_output[14] = vqmovun_s16((int16x8_t) v_out_temp);
    v_out_temp = vaddw_u8((uint16x8_t) v_x[1], v_output[15]);
    v_output[15] = vqmovun_s16((int16x8_t) v_out_temp);
    //
    store_output(v_output, (uint8_t*) dest, dest_stride);
  } else {
    int16_t* ptr = (int16_t*) dest;
    int dest_stride_by2 = dest_stride / 2;

    vst1q_s16(ptr, v_x[0]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[8]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[12]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[4]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[6]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[14]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[10]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[2]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[3]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[11]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[15]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[7]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[5]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[13]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[9]);
    ptr += dest_stride_by2;
    vst1q_s16(ptr, v_x[1]);
    ptr += dest_stride_by2;
  }
}

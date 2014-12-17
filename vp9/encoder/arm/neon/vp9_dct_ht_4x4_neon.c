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

static inline void vp9_dct4_neon(int16x4_t *v_x0, int16x4_t *v_x1,
                                 int16x4_t *v_x2, int16x4_t *v_x3) {

  int32x4_t step_0 = vaddl_s16(*v_x0, *v_x3);
  int32x4_t step_1 = vaddl_s16(*v_x1, *v_x2);
  int32x4_t step_2 = vsubl_s16(*v_x1, *v_x2);
  int32x4_t step_3 = vsubl_s16(*v_x0, *v_x3);

  {
    int32x4_t temp_1, temp_2;
    temp_1 = vaddq_s32(step_0, step_1);
    temp_2 = vsubq_s32(step_0, step_1);
    temp_1 = vmulq_n_s32(temp_1, (int32_t) cospi_16_64);
    temp_2 = vmulq_n_s32(temp_2, (int32_t) cospi_16_64);
    *v_x0 = vrshrn_n_s32(temp_1, DCT_CONST_BITS);
    *v_x2 = vrshrn_n_s32(temp_2, DCT_CONST_BITS);
  }
  {
    int32x4_t temp_3, temp_4;
    temp_3 = vmulq_n_s32(step_3, (int32_t) cospi_8_64);
    temp_3 = vmlaq_n_s32(temp_3, step_2, (int32_t) cospi_24_64);
    temp_4 = vmulq_n_s32(step_3, (int32_t) cospi_24_64);
    temp_4 = vmlsq_n_s32(temp_4, step_2, (int32_t) cospi_8_64);

    *v_x1 = vrshrn_n_s32(temp_3, DCT_CONST_BITS);
    *v_x3 = vrshrn_n_s32(temp_4, DCT_CONST_BITS);
  }
}

static inline void vp9_dst4_neon(int16x4_t *v_x0, int16x4_t *v_x1,
                                 int16x4_t *v_x2, int16x4_t *v_x3) {

  int32x4_t x0, x1, x2, x3;

  int32x4_t s0 = vmull_n_s16(*v_x0, (int16_t) sinpi_1_9);
  int32x4_t s1 = vmull_n_s16(*v_x0, (int16_t) sinpi_4_9);
  int32x4_t s2 = vmull_n_s16(*v_x1, (int16_t) sinpi_2_9);
  int32x4_t s3 = vmull_n_s16(*v_x1, (int16_t) sinpi_1_9);
  int32x4_t s4 = vmull_n_s16(*v_x2, (int16_t) sinpi_3_9);
  int32x4_t s5 = vmull_n_s16(*v_x3, (int16_t) sinpi_4_9);
  int32x4_t s6 = vmull_n_s16(*v_x3, (int16_t) sinpi_2_9);
  int32x4_t s7 = vaddl_s16(*v_x0, *v_x1);
  int32x4_t s7_tmp = vmovl_s16(*v_x3);
  s7 = vsubq_s32(s7, s7_tmp);

  x0 = vaddq_s32(s0, s2);
  x0 = vaddq_s32(x0, s5);
  x1 = vmulq_n_s32(s7, (int32_t) sinpi_3_9);
  x2 = vsubq_s32(s1, s3);
  x2 = vaddq_s32(x2, s6);
  x3 = s4;

  {
    int32x4_t x0_t, x1_t, x2_t, x3_t;

    x0_t = vaddq_s32(x0, x3);
    x1_t = x1;
    x2_t = vsubq_s32(x2, x3);
    x3_t = vsubq_s32(x2, x0);
    x3_t = vaddq_s32(x3_t, x3);

    *v_x0 = vrshrn_n_s32(x0_t, DCT_CONST_BITS);
    *v_x1 = vrshrn_n_s32(x1_t, DCT_CONST_BITS);
    *v_x2 = vrshrn_n_s32(x2_t, DCT_CONST_BITS);
    *v_x3 = vrshrn_n_s32(x3_t, DCT_CONST_BITS);
  }
}

static inline void transpose_4x4(int16x4_t *v_x0, int16x4_t *v_x1,
                                 int16x4_t *v_x2, int16x4_t *v_x3) {

  const int16x4x2_t r01_s16 = vtrn_s16(*v_x0, *v_x1);
  const int16x4x2_t r23_s16 = vtrn_s16(*v_x2, *v_x3);

  const int32x2x2_t r01_s32 = vtrn_s32(vreinterpret_s32_s16(r01_s16.val[0]),
                                       vreinterpret_s32_s16(r23_s16.val[0]));
  const int32x2x2_t r23_s32 = vtrn_s32(vreinterpret_s32_s16(r01_s16.val[1]),
                                       vreinterpret_s32_s16(r23_s16.val[1]));

  *v_x0 = vreinterpret_s16_s32(r01_s32.val[0]);
  *v_x2 = vreinterpret_s16_s32(r01_s32.val[1]);
  *v_x1 = vreinterpret_s16_s32(r23_s32.val[0]);
  *v_x3 = vreinterpret_s16_s32(r23_s32.val[1]);
}

void vp9_fdct4x4_neon(const int16_t *input, tran_low_t *output, int stride) {
  int16x4_t input_0 = vld1_s16(&input[0 * stride]);
  int16x4_t input_1 = vld1_s16(&input[1 * stride]);
  int16x4_t input_2 = vld1_s16(&input[2 * stride]);
  int16x4_t input_3 = vld1_s16(&input[3 * stride]);

  // temp_in[j] = input[j * stride + i] * 16;
  input_0 = vqshl_n_s16(input_0, 4);
  input_1 = vqshl_n_s16(input_1, 4);
  input_2 = vqshl_n_s16(input_2, 4);
  input_3 = vqshl_n_s16(input_3, 4);

  {
    // if (i == 0 && temp_in[0])temp_in[0] += 1;
    uint16x4_t allzero;
    uint64x1_t iszero;
    allzero = vmov_n_u16(0);

    iszero = vreinterpret_u64_u16(
        vcgt_u16(vreinterpret_u16_s16(input_0), allzero));
    iszero = vshl_n_u64(iszero, 63);
    iszero = vshr_n_u64(iszero, 63);
    input_0 = vadd_s16(input_0, vreinterpret_s16_u64(iszero));
  }

  vp9_dct4_neon(&input_0, &input_1, &input_2, &input_3);
  // transpose the 4x4 matrix
  transpose_4x4(&input_0, &input_1, &input_2, &input_3);
  vp9_dct4_neon(&input_0, &input_1, &input_2, &input_3);
  // transpose the 4x4 matrix
  transpose_4x4(&input_0, &input_1, &input_2, &input_3);

  input_0 = vrshr_n_s16(input_0, 1);
  input_1 = vrshr_n_s16(input_1, 1);
  input_2 = vrshr_n_s16(input_2, 1);
  input_3 = vrshr_n_s16(input_3, 1);

  input_0 = vshr_n_s16(input_0, 1);
  input_1 = vshr_n_s16(input_1, 1);
  input_2 = vshr_n_s16(input_2, 1);
  input_3 = vshr_n_s16(input_3, 1);

  vst1_s16(output + 0, input_0);
  vst1_s16(output + 4, input_1);
  vst1_s16(output + 8, input_2);
  vst1_s16(output + 12, input_3);

}

void vp9_fht4x4_neon(const int16_t *input, tran_low_t *output, int stride,
                     int tx_type) {
  int16x4_t input_0 = vld1_s16(&input[0 * stride]);
  int16x4_t input_1 = vld1_s16(&input[1 * stride]);
  int16x4_t input_2 = vld1_s16(&input[2 * stride]);
  int16x4_t input_3 = vld1_s16(&input[3 * stride]);

  // temp_in[j] = input[j * stride + i] * 16;
  input_0 = vqshl_n_s16(input_0, 4);
  input_1 = vqshl_n_s16(input_1, 4);
  input_2 = vqshl_n_s16(input_2, 4);
  input_3 = vqshl_n_s16(input_3, 4);

  {
    // if (i == 0 && temp_in[0])temp_in[0] += 1;
    uint16x4_t allzero;
    uint64x1_t iszero;
    allzero = vmov_n_u16(0);

    iszero = vreinterpret_u64_u16(
        vcgt_u16(vreinterpret_u16_s16(input_0), allzero));
    iszero = vshl_n_u64(iszero, 63);
    iszero = vshr_n_u64(iszero, 63);
    input_0 = vadd_s16(input_0, vreinterpret_s16_u64(iszero));
  }

  if (tx_type == ADST_ADST || tx_type == ADST_DCT) {
    vp9_dst4_neon(&input_0, &input_1, &input_2, &input_3);
  } else {
    vp9_dct4_neon(&input_0, &input_1, &input_2, &input_3);
  }

  // transpose the 4x4 matrix
  transpose_4x4(&input_0, &input_1, &input_2, &input_3);

  if (tx_type == ADST_ADST || tx_type == DCT_ADST) {
    vp9_dst4_neon(&input_0, &input_1, &input_2, &input_3);
  } else {
    vp9_dct4_neon(&input_0, &input_1, &input_2, &input_3);
  }
  // transpose the 4x4 matrix
  transpose_4x4(&input_0, &input_1, &input_2, &input_3);

  input_0 = vrshr_n_s16(input_0, 1);
  input_1 = vrshr_n_s16(input_1, 1);
  input_2 = vrshr_n_s16(input_2, 1);
  input_3 = vrshr_n_s16(input_3, 1);

  input_0 = vshr_n_s16(input_0, 1);
  input_1 = vshr_n_s16(input_1, 1);
  input_2 = vshr_n_s16(input_2, 1);
  input_3 = vshr_n_s16(input_3, 1);

  vst1_s16(output + 0, input_0);
  vst1_s16(output + 4, input_1);
  vst1_s16(output + 8, input_2);
  vst1_s16(output + 12, input_3);
}


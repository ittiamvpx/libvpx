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

//Transpose an 8x8 matrix loaded in v_x0,v_x1,v_x2,v_x3,v_x4,v_x4,v_x5,v_x6,v_x7
static inline void transpose_8x8_neon(int16x8_t *v_x0, int16x8_t *v_x1,
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

//Compute 1-D forward DST for 8x8 block
static inline void vp9_fadst8_neon(int16x8_t *v_x0, int16x8_t *v_x1,
                                   int16x8_t *v_x2, int16x8_t *v_x3,
                                   int16x8_t *v_x4, int16x8_t *v_x5,
                                   int16x8_t *v_x6, int16x8_t *v_x7) {
  // Ping pong temporary registers
  int32x4_t v_s0_lo, v_s1_lo, v_s2_lo, v_s3_lo, v_s4_lo, v_s5_lo, v_s6_lo,
      v_s7_lo;
  int32x4_t v_s0_hi, v_s1_hi, v_s2_hi, v_s3_hi, v_s4_hi, v_s5_hi, v_s6_hi,
      v_s7_hi;
  int32x4_t v_x0_lo, v_x1_lo, v_x2_lo, v_x3_lo, v_x4_lo, v_x5_lo, v_x6_lo,
      v_x7_lo;
  int32x4_t v_x0_hi, v_x1_hi, v_x2_hi, v_x3_hi, v_x4_hi, v_x5_hi, v_x6_hi,
      v_x7_hi;

  //Registers for final stage
  int16x4_t a, b, c, d, e, f, g, h;

  //stage 1 s0-s1 scaling part
  v_s0_lo = vmull_n_s16(vget_low_s16(*v_x7), (int16_t) cospi_2_64);
  v_s0_hi = vmull_n_s16(vget_high_s16(*v_x7), (int16_t) cospi_2_64);
  v_s1_lo = vmull_n_s16(vget_low_s16(*v_x7), (int16_t) cospi_30_64);
  v_s1_hi = vmull_n_s16(vget_high_s16(*v_x7), (int16_t) cospi_30_64);
  //stage 1 s4-s5 scaling part
  v_s4_lo = vmull_n_s16(vget_low_s16(*v_x3), (int16_t) cospi_18_64);
  v_s4_hi = vmull_n_s16(vget_high_s16(*v_x3), (int16_t) cospi_18_64);
  v_s5_lo = vmull_n_s16(vget_low_s16(*v_x3), (int16_t) cospi_14_64);
  v_s5_hi = vmull_n_s16(vget_high_s16(*v_x3), (int16_t) cospi_14_64);
  //stage 1 s0-s1 scaling part
  v_s0_lo = vmlal_n_s16(v_s0_lo, vget_low_s16(*v_x0), (int16_t) cospi_30_64);
  v_s0_hi = vmlal_n_s16(v_s0_hi, vget_high_s16(*v_x0), (int16_t) cospi_30_64);
  v_s1_lo = vmlsl_n_s16(v_s1_lo, vget_low_s16(*v_x0), (int16_t) cospi_2_64);
  v_s1_hi = vmlsl_n_s16(v_s1_hi, vget_high_s16(*v_x0), (int16_t) cospi_2_64);
  //stage 1 s4-s5 scaling part
  v_s4_lo = vmlal_n_s16(v_s4_lo, vget_low_s16(*v_x4), (int16_t) cospi_14_64);
  v_s4_hi = vmlal_n_s16(v_s4_hi, vget_high_s16(*v_x4), (int16_t) cospi_14_64);
  v_s5_lo = vmlsl_n_s16(v_s5_lo, vget_low_s16(*v_x4), (int16_t) cospi_18_64);
  v_s5_hi = vmlsl_n_s16(v_s5_hi, vget_high_s16(*v_x4), (int16_t) cospi_18_64);
  //stage 1 s2-s3 scaling part
  v_s2_lo = vmull_n_s16(vget_low_s16(*v_x5), (int16_t) cospi_10_64);
  v_s2_hi = vmull_n_s16(vget_high_s16(*v_x5), (int16_t) cospi_10_64);
  v_s3_lo = vmull_n_s16(vget_low_s16(*v_x5), (int16_t) cospi_22_64);
  v_s3_hi = vmull_n_s16(vget_high_s16(*v_x5), (int16_t) cospi_22_64);
  //stage 1 s6-s7 scaling part
  v_s6_lo = vmull_n_s16(vget_low_s16(*v_x1), (int16_t) cospi_26_64);
  v_s6_hi = vmull_n_s16(vget_high_s16(*v_x1), (int16_t) cospi_26_64);
  v_s7_lo = vmull_n_s16(vget_low_s16(*v_x1), (int16_t) cospi_6_64);
  v_s7_hi = vmull_n_s16(vget_high_s16(*v_x1), (int16_t) cospi_6_64);
  //stage 1 s0-s1 scaling part
  v_s2_lo = vmlal_n_s16(v_s2_lo, vget_low_s16(*v_x2), (int16_t) cospi_22_64);
  v_s2_hi = vmlal_n_s16(v_s2_hi, vget_high_s16(*v_x2), (int16_t) cospi_22_64);
  v_s3_lo = vmlsl_n_s16(v_s3_lo, vget_low_s16(*v_x2), (int16_t) cospi_10_64);
  v_s3_hi = vmlsl_n_s16(v_s3_hi, vget_high_s16(*v_x2), (int16_t) cospi_10_64);
  //stage 1 s6-s7 scaling part
  v_s6_lo = vmlal_n_s16(v_s6_lo, vget_low_s16(*v_x6), (int16_t) cospi_6_64);
  v_s6_hi = vmlal_n_s16(v_s6_hi, vget_high_s16(*v_x6), (int16_t) cospi_6_64);
  v_s7_lo = vmlsl_n_s16(v_s7_lo, vget_low_s16(*v_x6), (int16_t) cospi_26_64);
  v_s7_hi = vmlsl_n_s16(v_s7_hi, vget_high_s16(*v_x6), (int16_t) cospi_26_64);

  //stage 1 s0-s7 addition and subtraction part
  v_x0_lo = vaddq_s32(v_s0_lo, v_s4_lo);
  v_x0_hi = vaddq_s32(v_s0_hi, v_s4_hi);
  v_x4_lo = vsubq_s32(v_s0_lo, v_s4_lo);
  v_x4_hi = vsubq_s32(v_s0_hi, v_s4_hi);
  v_x1_lo = vaddq_s32(v_s1_lo, v_s5_lo);
  v_x1_hi = vaddq_s32(v_s1_hi, v_s5_hi);
  v_x5_lo = vsubq_s32(v_s1_lo, v_s5_lo);
  v_x5_hi = vsubq_s32(v_s1_hi, v_s5_hi);
  v_x2_lo = vaddq_s32(v_s2_lo, v_s6_lo);
  v_x2_hi = vaddq_s32(v_s2_hi, v_s6_hi);
  v_x6_lo = vsubq_s32(v_s2_lo, v_s6_lo);
  v_x6_hi = vsubq_s32(v_s2_hi, v_s6_hi);
  v_x3_lo = vaddq_s32(v_s3_lo, v_s7_lo);
  v_x3_hi = vaddq_s32(v_s3_hi, v_s7_hi);
  v_x7_lo = vsubq_s32(v_s3_lo, v_s7_lo);
  v_x7_hi = vsubq_s32(v_s3_hi, v_s7_hi);

  // stage 1 fdct_round_shift
  v_x0_lo = vrshrq_n_s32(v_x0_lo, DCT_CONST_BITS);
  v_x0_hi = vrshrq_n_s32(v_x0_hi, DCT_CONST_BITS);
  v_x4_lo = vrshrq_n_s32(v_x4_lo, DCT_CONST_BITS);
  v_x4_hi = vrshrq_n_s32(v_x4_hi, DCT_CONST_BITS);
  v_x1_lo = vrshrq_n_s32(v_x1_lo, DCT_CONST_BITS);
  v_x1_hi = vrshrq_n_s32(v_x1_hi, DCT_CONST_BITS);
  v_x5_lo = vrshrq_n_s32(v_x5_lo, DCT_CONST_BITS);
  v_x5_hi = vrshrq_n_s32(v_x5_hi, DCT_CONST_BITS);
  v_x2_lo = vrshrq_n_s32(v_x2_lo, DCT_CONST_BITS);
  v_x2_hi = vrshrq_n_s32(v_x2_hi, DCT_CONST_BITS);
  v_x6_lo = vrshrq_n_s32(v_x6_lo, DCT_CONST_BITS);
  v_x6_hi = vrshrq_n_s32(v_x6_hi, DCT_CONST_BITS);
  v_x3_lo = vrshrq_n_s32(v_x3_lo, DCT_CONST_BITS);
  v_x3_hi = vrshrq_n_s32(v_x3_hi, DCT_CONST_BITS);
  v_x7_lo = vrshrq_n_s32(v_x7_lo, DCT_CONST_BITS);
  v_x7_hi = vrshrq_n_s32(v_x7_hi, DCT_CONST_BITS);

  //stage 2
  v_s0_lo = v_x0_lo;
  v_s0_hi = v_x0_hi;
  v_s1_lo = v_x1_lo;
  v_s1_hi = v_x1_hi;
  v_s2_lo = v_x2_lo;
  v_s2_hi = v_x2_hi;
  v_s3_lo = v_x3_lo;
  v_s3_hi = v_x3_hi;
  //stage 2 x4-x5
  v_s4_lo = vmulq_n_s32(v_x4_lo, cospi_8_64);
  v_s4_hi = vmulq_n_s32(v_x4_hi, cospi_8_64);
  v_s5_lo = vmulq_n_s32(v_x4_lo, cospi_24_64);
  v_s5_hi = vmulq_n_s32(v_x4_hi, cospi_24_64);
  v_s4_lo = vmlaq_n_s32(v_s4_lo, v_x5_lo, cospi_24_64);
  v_s4_hi = vmlaq_n_s32(v_s4_hi, v_x5_hi, cospi_24_64);
  v_s5_lo = vmlsq_n_s32(v_s5_lo, v_x5_lo, cospi_8_64);
  v_s5_hi = vmlsq_n_s32(v_s5_hi, v_x5_hi, cospi_8_64);

  //stage 2 x6-x7
  v_s6_lo = vmulq_n_s32(v_x7_lo, cospi_8_64);
  v_s6_hi = vmulq_n_s32(v_x7_hi, cospi_8_64);
  v_s7_lo = vmulq_n_s32(v_x7_lo, cospi_24_64);
  v_s7_hi = vmulq_n_s32(v_x7_hi, cospi_24_64);
  v_s6_lo = vmlsq_n_s32(v_s6_lo, v_x6_lo, cospi_24_64);
  v_s6_hi = vmlsq_n_s32(v_s6_hi, v_x6_hi, cospi_24_64);
  v_s7_lo = vmlaq_n_s32(v_s7_lo, v_x6_lo, cospi_8_64);
  v_s7_hi = vmlaq_n_s32(v_s7_hi, v_x6_hi, cospi_8_64);

  //stage 2 s0-s3
  v_x0_lo = vaddq_s32(v_s0_lo, v_s2_lo);
  v_x0_hi = vaddq_s32(v_s0_hi, v_s2_hi);
  v_x1_lo = vaddq_s32(v_s1_lo, v_s3_lo);
  v_x1_hi = vaddq_s32(v_s1_hi, v_s3_hi);
  v_x2_lo = vsubq_s32(v_s0_lo, v_s2_lo);
  v_x2_hi = vsubq_s32(v_s0_hi, v_s2_hi);
  v_x3_lo = vsubq_s32(v_s1_lo, v_s3_lo);
  v_x3_hi = vsubq_s32(v_s1_hi, v_s3_hi);
  //stage 2 s4-s6
  v_x4_lo = vaddq_s32(v_s4_lo, v_s6_lo);
  v_x4_hi = vaddq_s32(v_s4_hi, v_s6_hi);
  v_x5_lo = vaddq_s32(v_s5_lo, v_s7_lo);
  v_x5_hi = vaddq_s32(v_s5_hi, v_s7_hi);
  v_x6_lo = vsubq_s32(v_s4_lo, v_s6_lo);
  v_x6_hi = vsubq_s32(v_s4_hi, v_s6_hi);
  v_x7_lo = vsubq_s32(v_s5_lo, v_s7_lo);
  v_x7_hi = vsubq_s32(v_s5_hi, v_s7_hi);

  //Store x0 x1
  a = vmovn_s32(v_x0_lo);
  b = vmovn_s32(v_x0_hi);
  c = vmovn_s32(v_x1_lo);
  d = vmovn_s32(v_x1_hi);

  *v_x0 = vcombine_s16(a, b);
  *v_x7 = vcombine_s16(c, d);
  *v_x7 = vnegq_s16(*v_x7);

  //stage 2 fdct_round_shift s4-s5
  e = vrshrn_n_s32(v_x4_lo, DCT_CONST_BITS);
  f = vrshrn_n_s32(v_x4_hi, DCT_CONST_BITS);
  g = vrshrn_n_s32(v_x5_lo, DCT_CONST_BITS);
  h = vrshrn_n_s32(v_x5_hi, DCT_CONST_BITS);

  //stage 2 fdct_round_shift s6-s7
  v_x6_lo = vrshrq_n_s32(v_x6_lo, DCT_CONST_BITS);
  v_x6_hi = vrshrq_n_s32(v_x6_hi, DCT_CONST_BITS);
  v_x7_lo = vrshrq_n_s32(v_x7_lo, DCT_CONST_BITS);
  v_x7_hi = vrshrq_n_s32(v_x7_hi, DCT_CONST_BITS);

  *v_x1 = vcombine_s16(e, f);
  *v_x1 = vnegq_s16(*v_x1);
  *v_x6 = vcombine_s16(g, h);

  //stage 3
  v_s2_lo = vaddq_s32(v_x2_lo, v_x3_lo);
  v_s2_hi = vaddq_s32(v_x2_hi, v_x3_hi);
  v_s3_lo = vsubq_s32(v_x2_lo, v_x3_lo);
  v_s3_hi = vsubq_s32(v_x2_hi, v_x3_hi);
  v_s6_lo = vaddq_s32(v_x6_lo, v_x7_lo);
  v_s6_hi = vaddq_s32(v_x6_hi, v_x7_hi);
  v_s7_lo = vsubq_s32(v_x6_lo, v_x7_lo);
  v_s7_hi = vsubq_s32(v_x6_hi, v_x7_hi);

  v_s2_lo = vmulq_n_s32(v_s2_lo, cospi_16_64);
  v_s2_hi = vmulq_n_s32(v_s2_hi, cospi_16_64);
  v_s3_lo = vmulq_n_s32(v_s3_lo, cospi_16_64);
  v_s3_hi = vmulq_n_s32(v_s3_hi, cospi_16_64);
  v_s6_lo = vmulq_n_s32(v_s6_lo, cospi_16_64);
  v_s6_hi = vmulq_n_s32(v_s6_hi, cospi_16_64);
  v_s7_lo = vmulq_n_s32(v_s7_lo, cospi_16_64);
  v_s7_hi = vmulq_n_s32(v_s7_hi, cospi_16_64);

  a = vrshrn_n_s32(v_s2_lo, DCT_CONST_BITS);
  b = vrshrn_n_s32(v_s2_hi, DCT_CONST_BITS);
  c = vrshrn_n_s32(v_s3_lo, DCT_CONST_BITS);
  d = vrshrn_n_s32(v_s3_hi, DCT_CONST_BITS);
  e = vrshrn_n_s32(v_s6_lo, DCT_CONST_BITS);
  f = vrshrn_n_s32(v_s6_hi, DCT_CONST_BITS);
  g = vrshrn_n_s32(v_s7_lo, DCT_CONST_BITS);
  h = vrshrn_n_s32(v_s7_hi, DCT_CONST_BITS);

  *v_x3 = vcombine_s16(a, b);  //x3
  *v_x3 = vnegq_s16(*v_x3);

  *v_x4 = vcombine_s16(c, d);  //x4
  *v_x2 = vcombine_s16(e, f);  //x2
  *v_x5 = vcombine_s16(g, h);  //x5
  *v_x5 = vnegq_s16(*v_x5);
}

//Compute 2-D forward DCT for 8x8 block
static inline void vp9_fdct8_neon(int16x8_t *v_x0, int16x8_t *v_x1,
                                  int16x8_t *v_x2, int16x8_t *v_x3,
                                  int16x8_t *v_x4, int16x8_t *v_x5,
                                  int16x8_t *v_x6, int16x8_t *v_x7) {
  // stage 1
  const int16x8_t v_s0 = vaddq_s16(*v_x0, *v_x7);
  const int16x8_t v_s1 = vaddq_s16(*v_x1, *v_x6);
  const int16x8_t v_s2 = vaddq_s16(*v_x2, *v_x5);
  const int16x8_t v_s3 = vaddq_s16(*v_x3, *v_x4);
  const int16x8_t v_s4 = vsubq_s16(*v_x3, *v_x4);
  const int16x8_t v_s5 = vsubq_s16(*v_x2, *v_x5);
  const int16x8_t v_s6 = vsubq_s16(*v_x1, *v_x6);
  const int16x8_t v_s7 = vsubq_s16(*v_x0, *v_x7);

  // fdct4(step, step);
  int16x8_t v_x0_in = vaddq_s16(v_s0, v_s3);
  int16x8_t v_x1_in = vaddq_s16(v_s1, v_s2);
  int16x8_t v_x2_in = vsubq_s16(v_s1, v_s2);
  int16x8_t v_x3_in = vsubq_s16(v_s0, v_s3);
  // fdct4(step, step);
  int32x4_t v_t0_lo = vaddl_s16(vget_low_s16(v_x0_in), vget_low_s16(v_x1_in));
  int32x4_t v_t0_hi = vaddl_s16(vget_high_s16(v_x0_in), vget_high_s16(v_x1_in));
  int32x4_t v_t1_lo = vsubl_s16(vget_low_s16(v_x0_in), vget_low_s16(v_x1_in));
  int32x4_t v_t1_hi = vsubl_s16(vget_high_s16(v_x0_in), vget_high_s16(v_x1_in));
  int32x4_t v_t2_lo = vmull_n_s16(vget_low_s16(v_x2_in), (int16_t) cospi_24_64);
  int32x4_t v_t2_hi = vmull_n_s16(vget_high_s16(v_x2_in),
                                  (int16_t) cospi_24_64);
  int32x4_t v_t3_lo = vmull_n_s16(vget_low_s16(v_x3_in), (int16_t) cospi_24_64);
  int32x4_t v_t3_hi = vmull_n_s16(vget_high_s16(v_x3_in),
                                  (int16_t) cospi_24_64);
  v_t2_lo = vmlal_n_s16(v_t2_lo, vget_low_s16(v_x3_in), (int16_t) cospi_8_64);
  v_t2_hi = vmlal_n_s16(v_t2_hi, vget_high_s16(v_x3_in), (int16_t) cospi_8_64);
  v_t3_lo = vmlsl_n_s16(v_t3_lo, vget_low_s16(v_x2_in), (int16_t) cospi_8_64);
  v_t3_hi = vmlsl_n_s16(v_t3_hi, vget_high_s16(v_x2_in), (int16_t) cospi_8_64);
  v_t0_lo = vmulq_n_s32(v_t0_lo, cospi_16_64);
  v_t0_hi = vmulq_n_s32(v_t0_hi, cospi_16_64);
  v_t1_lo = vmulq_n_s32(v_t1_lo, cospi_16_64);
  v_t1_hi = vmulq_n_s32(v_t1_hi, cospi_16_64);
  {
    const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
    const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
    const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
    const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
    const int16x4_t e = vrshrn_n_s32(v_t2_lo, DCT_CONST_BITS);
    const int16x4_t f = vrshrn_n_s32(v_t2_hi, DCT_CONST_BITS);
    const int16x4_t g = vrshrn_n_s32(v_t3_lo, DCT_CONST_BITS);
    const int16x4_t h = vrshrn_n_s32(v_t3_hi, DCT_CONST_BITS);

    *v_x0 = vcombine_s16(a, b);
    *v_x4 = vcombine_s16(c, d);
    *v_x2 = vcombine_s16(e, f);
    *v_x6 = vcombine_s16(g, h);
  }
  // Stage 2
  v_x0_in = vsubq_s16(v_s6, v_s5);
  v_x1_in = vaddq_s16(v_s6, v_s5);
  v_t0_lo = vmull_n_s16(vget_low_s16(v_x0_in), (int16_t) cospi_16_64);
  v_t0_hi = vmull_n_s16(vget_high_s16(v_x0_in), (int16_t) cospi_16_64);
  v_t1_lo = vmull_n_s16(vget_low_s16(v_x1_in), (int16_t) cospi_16_64);
  v_t1_hi = vmull_n_s16(vget_high_s16(v_x1_in), (int16_t) cospi_16_64);
  {
    const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
    const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
    const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
    const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
    const int16x8_t ab = vcombine_s16(a, b);
    const int16x8_t cd = vcombine_s16(c, d);
    // Stage 3
    v_x0_in = vaddq_s16(v_s4, ab);
    v_x1_in = vsubq_s16(v_s4, ab);
    v_x2_in = vsubq_s16(v_s7, cd);
    v_x3_in = vaddq_s16(v_s7, cd);
  }
  // Stage 4
  v_t0_lo = vmull_n_s16(vget_low_s16(v_x3_in), (int16_t) cospi_4_64);
  v_t0_hi = vmull_n_s16(vget_high_s16(v_x3_in), (int16_t) cospi_4_64);
  v_t0_lo = vmlal_n_s16(v_t0_lo, vget_low_s16(v_x0_in), (int16_t) cospi_28_64);
  v_t0_hi = vmlal_n_s16(v_t0_hi, vget_high_s16(v_x0_in), (int16_t) cospi_28_64);
  v_t1_lo = vmull_n_s16(vget_low_s16(v_x1_in), (int16_t) cospi_12_64);
  v_t1_hi = vmull_n_s16(vget_high_s16(v_x1_in), (int16_t) cospi_12_64);
  v_t1_lo = vmlal_n_s16(v_t1_lo, vget_low_s16(v_x2_in), (int16_t) cospi_20_64);
  v_t1_hi = vmlal_n_s16(v_t1_hi, vget_high_s16(v_x2_in), (int16_t) cospi_20_64);
  v_t2_lo = vmull_n_s16(vget_low_s16(v_x2_in), (int16_t) cospi_12_64);
  v_t2_hi = vmull_n_s16(vget_high_s16(v_x2_in), (int16_t) cospi_12_64);
  v_t2_lo = vmlsl_n_s16(v_t2_lo, vget_low_s16(v_x1_in), (int16_t) cospi_20_64);
  v_t2_hi = vmlsl_n_s16(v_t2_hi, vget_high_s16(v_x1_in), (int16_t) cospi_20_64);
  v_t3_lo = vmull_n_s16(vget_low_s16(v_x3_in), (int16_t) cospi_28_64);
  v_t3_hi = vmull_n_s16(vget_high_s16(v_x3_in), (int16_t) cospi_28_64);
  v_t3_lo = vmlsl_n_s16(v_t3_lo, vget_low_s16(v_x0_in), (int16_t) cospi_4_64);
  v_t3_hi = vmlsl_n_s16(v_t3_hi, vget_high_s16(v_x0_in), (int16_t) cospi_4_64);
  {
    const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
    const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
    const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
    const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
    const int16x4_t e = vrshrn_n_s32(v_t2_lo, DCT_CONST_BITS);
    const int16x4_t f = vrshrn_n_s32(v_t2_hi, DCT_CONST_BITS);
    const int16x4_t g = vrshrn_n_s32(v_t3_lo, DCT_CONST_BITS);
    const int16x4_t h = vrshrn_n_s32(v_t3_hi, DCT_CONST_BITS);
    *v_x1 = vcombine_s16(a, b);
    *v_x5 = vcombine_s16(c, d);
    *v_x3 = vcombine_s16(e, f);
    *v_x7 = vcombine_s16(g, h);
  }
}

void vp9_fdct8x8_1_neon(const int16_t *input, int16_t *output, int stride) {
  int r;
  int16x8_t sum = vld1q_s16(&input[0]);
  for (r = 1; r < 8; ++r) {
    const int16x8_t input_00 = vld1q_s16(&input[r * stride]);
    sum = vaddq_s16(sum, input_00);
  }
  {
    const int32x4_t a = vpaddlq_s16(sum);
    const int64x2_t b = vpaddlq_s32(a);
    const int32x2_t c = vadd_s32(vreinterpret_s32_s64(vget_low_s64(b)),
                                 vreinterpret_s32_s64(vget_high_s64(b)));
    output[0] = vget_lane_s16(vreinterpret_s16_s32(c), 0);
    output[1] = 0;
  }
}

void vp9_fdct8x8_neon(const int16_t *input, int16_t *final_output, int stride) {
  int i;
  // stage 1
  int16x8_t input_0 = vshlq_n_s16(vld1q_s16(&input[0 * stride]), 2);
  int16x8_t input_1 = vshlq_n_s16(vld1q_s16(&input[1 * stride]), 2);
  int16x8_t input_2 = vshlq_n_s16(vld1q_s16(&input[2 * stride]), 2);
  int16x8_t input_3 = vshlq_n_s16(vld1q_s16(&input[3 * stride]), 2);
  int16x8_t input_4 = vshlq_n_s16(vld1q_s16(&input[4 * stride]), 2);
  int16x8_t input_5 = vshlq_n_s16(vld1q_s16(&input[5 * stride]), 2);
  int16x8_t input_6 = vshlq_n_s16(vld1q_s16(&input[6 * stride]), 2);
  int16x8_t input_7 = vshlq_n_s16(vld1q_s16(&input[7 * stride]), 2);
  for (i = 0; i < 2; ++i) {
    int16x8_t out_0, out_1, out_2, out_3, out_4, out_5, out_6, out_7;
    const int16x8_t v_s0 = vaddq_s16(input_0, input_7);
    const int16x8_t v_s1 = vaddq_s16(input_1, input_6);
    const int16x8_t v_s2 = vaddq_s16(input_2, input_5);
    const int16x8_t v_s3 = vaddq_s16(input_3, input_4);
    const int16x8_t v_s4 = vsubq_s16(input_3, input_4);
    const int16x8_t v_s5 = vsubq_s16(input_2, input_5);
    const int16x8_t v_s6 = vsubq_s16(input_1, input_6);
    const int16x8_t v_s7 = vsubq_s16(input_0, input_7);
    // fdct4(step, step);
    int16x8_t v_x0 = vaddq_s16(v_s0, v_s3);
    int16x8_t v_x1 = vaddq_s16(v_s1, v_s2);
    int16x8_t v_x2 = vsubq_s16(v_s1, v_s2);
    int16x8_t v_x3 = vsubq_s16(v_s0, v_s3);
    // fdct4(step, step);
    int32x4_t v_t0_lo = vaddl_s16(vget_low_s16(v_x0), vget_low_s16(v_x1));
    int32x4_t v_t0_hi = vaddl_s16(vget_high_s16(v_x0), vget_high_s16(v_x1));
    int32x4_t v_t1_lo = vsubl_s16(vget_low_s16(v_x0), vget_low_s16(v_x1));
    int32x4_t v_t1_hi = vsubl_s16(vget_high_s16(v_x0), vget_high_s16(v_x1));
    int32x4_t v_t2_lo = vmull_n_s16(vget_low_s16(v_x2), (int16_t) cospi_24_64);
    int32x4_t v_t2_hi = vmull_n_s16(vget_high_s16(v_x2), (int16_t) cospi_24_64);
    int32x4_t v_t3_lo = vmull_n_s16(vget_low_s16(v_x3), (int16_t) cospi_24_64);
    int32x4_t v_t3_hi = vmull_n_s16(vget_high_s16(v_x3), (int16_t) cospi_24_64);
    v_t2_lo = vmlal_n_s16(v_t2_lo, vget_low_s16(v_x3), (int16_t) cospi_8_64);
    v_t2_hi = vmlal_n_s16(v_t2_hi, vget_high_s16(v_x3), (int16_t) cospi_8_64);
    v_t3_lo = vmlsl_n_s16(v_t3_lo, vget_low_s16(v_x2), (int16_t) cospi_8_64);
    v_t3_hi = vmlsl_n_s16(v_t3_hi, vget_high_s16(v_x2), (int16_t) cospi_8_64);
    v_t0_lo = vmulq_n_s32(v_t0_lo, cospi_16_64);
    v_t0_hi = vmulq_n_s32(v_t0_hi, cospi_16_64);
    v_t1_lo = vmulq_n_s32(v_t1_lo, cospi_16_64);
    v_t1_hi = vmulq_n_s32(v_t1_hi, cospi_16_64);
    {
      const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
      const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
      const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
      const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
      const int16x4_t e = vrshrn_n_s32(v_t2_lo, DCT_CONST_BITS);
      const int16x4_t f = vrshrn_n_s32(v_t2_hi, DCT_CONST_BITS);
      const int16x4_t g = vrshrn_n_s32(v_t3_lo, DCT_CONST_BITS);
      const int16x4_t h = vrshrn_n_s32(v_t3_hi, DCT_CONST_BITS);
      out_0 = vcombine_s16(a, c);  // 00 01 02 03 40 41 42 43
      out_2 = vcombine_s16(e, g);  // 20 21 22 23 60 61 62 63
      out_4 = vcombine_s16(b, d);  // 04 05 06 07 44 45 46 47
      out_6 = vcombine_s16(f, h);  // 24 25 26 27 64 65 66 67
    }
    // Stage 2
    v_x0 = vsubq_s16(v_s6, v_s5);
    v_x1 = vaddq_s16(v_s6, v_s5);
    v_t0_lo = vmull_n_s16(vget_low_s16(v_x0), (int16_t) cospi_16_64);
    v_t0_hi = vmull_n_s16(vget_high_s16(v_x0), (int16_t) cospi_16_64);
    v_t1_lo = vmull_n_s16(vget_low_s16(v_x1), (int16_t) cospi_16_64);
    v_t1_hi = vmull_n_s16(vget_high_s16(v_x1), (int16_t) cospi_16_64);
    {
      const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
      const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
      const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
      const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
      const int16x8_t ab = vcombine_s16(a, b);
      const int16x8_t cd = vcombine_s16(c, d);
      // Stage 3
      v_x0 = vaddq_s16(v_s4, ab);
      v_x1 = vsubq_s16(v_s4, ab);
      v_x2 = vsubq_s16(v_s7, cd);
      v_x3 = vaddq_s16(v_s7, cd);
    }
    // Stage 4
    v_t0_lo = vmull_n_s16(vget_low_s16(v_x3), (int16_t) cospi_4_64);
    v_t0_hi = vmull_n_s16(vget_high_s16(v_x3), (int16_t) cospi_4_64);
    v_t0_lo = vmlal_n_s16(v_t0_lo, vget_low_s16(v_x0), (int16_t) cospi_28_64);
    v_t0_hi = vmlal_n_s16(v_t0_hi, vget_high_s16(v_x0), (int16_t) cospi_28_64);
    v_t1_lo = vmull_n_s16(vget_low_s16(v_x1), (int16_t) cospi_12_64);
    v_t1_hi = vmull_n_s16(vget_high_s16(v_x1), (int16_t) cospi_12_64);
    v_t1_lo = vmlal_n_s16(v_t1_lo, vget_low_s16(v_x2), (int16_t) cospi_20_64);
    v_t1_hi = vmlal_n_s16(v_t1_hi, vget_high_s16(v_x2), (int16_t) cospi_20_64);
    v_t2_lo = vmull_n_s16(vget_low_s16(v_x2), (int16_t) cospi_12_64);
    v_t2_hi = vmull_n_s16(vget_high_s16(v_x2), (int16_t) cospi_12_64);
    v_t2_lo = vmlsl_n_s16(v_t2_lo, vget_low_s16(v_x1), (int16_t) cospi_20_64);
    v_t2_hi = vmlsl_n_s16(v_t2_hi, vget_high_s16(v_x1), (int16_t) cospi_20_64);
    v_t3_lo = vmull_n_s16(vget_low_s16(v_x3), (int16_t) cospi_28_64);
    v_t3_hi = vmull_n_s16(vget_high_s16(v_x3), (int16_t) cospi_28_64);
    v_t3_lo = vmlsl_n_s16(v_t3_lo, vget_low_s16(v_x0), (int16_t) cospi_4_64);
    v_t3_hi = vmlsl_n_s16(v_t3_hi, vget_high_s16(v_x0), (int16_t) cospi_4_64);
    {
      const int16x4_t a = vrshrn_n_s32(v_t0_lo, DCT_CONST_BITS);
      const int16x4_t b = vrshrn_n_s32(v_t0_hi, DCT_CONST_BITS);
      const int16x4_t c = vrshrn_n_s32(v_t1_lo, DCT_CONST_BITS);
      const int16x4_t d = vrshrn_n_s32(v_t1_hi, DCT_CONST_BITS);
      const int16x4_t e = vrshrn_n_s32(v_t2_lo, DCT_CONST_BITS);
      const int16x4_t f = vrshrn_n_s32(v_t2_hi, DCT_CONST_BITS);
      const int16x4_t g = vrshrn_n_s32(v_t3_lo, DCT_CONST_BITS);
      const int16x4_t h = vrshrn_n_s32(v_t3_hi, DCT_CONST_BITS);
      out_1 = vcombine_s16(a, c);  // 10 11 12 13 50 51 52 53
      out_3 = vcombine_s16(e, g);  // 30 31 32 33 70 71 72 73
      out_5 = vcombine_s16(b, d);  // 14 15 16 17 54 55 56 57
      out_7 = vcombine_s16(f, h);  // 34 35 36 37 74 75 76 77
    }
    // transpose 8x8
    {
      // 00 01 02 03 40 41 42 43
      // 10 11 12 13 50 51 52 53
      // 20 21 22 23 60 61 62 63
      // 30 31 32 33 70 71 72 73
      // 04 05 06 07 44 45 46 47
      // 14 15 16 17 54 55 56 57
      // 24 25 26 27 64 65 66 67
      // 34 35 36 37 74 75 76 77
      const int32x4x2_t r02_s32 = vtrnq_s32(vreinterpretq_s32_s16(out_0),
                                            vreinterpretq_s32_s16(out_2));
      const int32x4x2_t r13_s32 = vtrnq_s32(vreinterpretq_s32_s16(out_1),
                                            vreinterpretq_s32_s16(out_3));
      const int32x4x2_t r46_s32 = vtrnq_s32(vreinterpretq_s32_s16(out_4),
                                            vreinterpretq_s32_s16(out_6));
      const int32x4x2_t r57_s32 = vtrnq_s32(vreinterpretq_s32_s16(out_5),
                                            vreinterpretq_s32_s16(out_7));
      const int16x8x2_t r01_s16 = vtrnq_s16(
          vreinterpretq_s16_s32(r02_s32.val[0]),
          vreinterpretq_s16_s32(r13_s32.val[0]));
      const int16x8x2_t r23_s16 = vtrnq_s16(
          vreinterpretq_s16_s32(r02_s32.val[1]),
          vreinterpretq_s16_s32(r13_s32.val[1]));
      const int16x8x2_t r45_s16 = vtrnq_s16(
          vreinterpretq_s16_s32(r46_s32.val[0]),
          vreinterpretq_s16_s32(r57_s32.val[0]));
      const int16x8x2_t r67_s16 = vtrnq_s16(
          vreinterpretq_s16_s32(r46_s32.val[1]),
          vreinterpretq_s16_s32(r57_s32.val[1]));
      input_0 = r01_s16.val[0];
      input_1 = r01_s16.val[1];
      input_2 = r23_s16.val[0];
      input_3 = r23_s16.val[1];
      input_4 = r45_s16.val[0];
      input_5 = r45_s16.val[1];
      input_6 = r67_s16.val[0];
      input_7 = r67_s16.val[1];
      // 00 10 20 30 40 50 60 70
      // 01 11 21 31 41 51 61 71
      // 02 12 22 32 42 52 62 72
      // 03 13 23 33 43 53 63 73
      // 04 14 24 34 44 54 64 74
      // 05 15 25 35 45 55 65 75
      // 06 16 26 36 46 56 66 76
      // 07 17 27 37 47 57 67 77
    }
  }  // for
  {
    // from vp9_dct_sse2.c
    // Post-condition (division by two)
    //    division of two 16 bits signed numbers using shifts
    //    n / 2 = (n - (n >> 15)) >> 1
    const int16x8_t sign_in0 = vshrq_n_s16(input_0, 15);
    const int16x8_t sign_in1 = vshrq_n_s16(input_1, 15);
    const int16x8_t sign_in2 = vshrq_n_s16(input_2, 15);
    const int16x8_t sign_in3 = vshrq_n_s16(input_3, 15);
    const int16x8_t sign_in4 = vshrq_n_s16(input_4, 15);
    const int16x8_t sign_in5 = vshrq_n_s16(input_5, 15);
    const int16x8_t sign_in6 = vshrq_n_s16(input_6, 15);
    const int16x8_t sign_in7 = vshrq_n_s16(input_7, 15);
    input_0 = vhsubq_s16(input_0, sign_in0);
    input_1 = vhsubq_s16(input_1, sign_in1);
    input_2 = vhsubq_s16(input_2, sign_in2);
    input_3 = vhsubq_s16(input_3, sign_in3);
    input_4 = vhsubq_s16(input_4, sign_in4);
    input_5 = vhsubq_s16(input_5, sign_in5);
    input_6 = vhsubq_s16(input_6, sign_in6);
    input_7 = vhsubq_s16(input_7, sign_in7);
    // store results
    vst1q_s16(&final_output[0 * 8], input_0);
    vst1q_s16(&final_output[1 * 8], input_1);
    vst1q_s16(&final_output[2 * 8], input_2);
    vst1q_s16(&final_output[3 * 8], input_3);
    vst1q_s16(&final_output[4 * 8], input_4);
    vst1q_s16(&final_output[5 * 8], input_5);
    vst1q_s16(&final_output[6 * 8], input_6);
    vst1q_s16(&final_output[7 * 8], input_7);
  }
}

//Compute 2-D forward hybrid transform for 8x8 block
void vp9_fht8x8_neon(const int16_t *input, tran_low_t *output, int stride,
                     int tx_type) {
  //Registers used for input and output for 1-D transform
  int16x8_t v_x0, v_x1, v_x2, v_x3, v_x4, v_x5, v_x6, v_x7;
  //Temporary registers for downshifting output
  uint16x8_t v_l0, v_l1, v_l2, v_l3, v_l4, v_l5, v_l6, v_l7;

  //Variable to hold current pass
  int pass;

  if (tx_type == DCT_DCT) {
    vp9_fdct8x8_neon(input, output, stride);
    return;
  }

  //load all the data into registers
  v_x0 = vshlq_n_s16(vld1q_s16(&input[0 * stride]), 2);
  v_x1 = vshlq_n_s16(vld1q_s16(&input[1 * stride]), 2);
  v_x2 = vshlq_n_s16(vld1q_s16(&input[2 * stride]), 2);
  v_x3 = vshlq_n_s16(vld1q_s16(&input[3 * stride]), 2);
  v_x4 = vshlq_n_s16(vld1q_s16(&input[4 * stride]), 2);
  v_x5 = vshlq_n_s16(vld1q_s16(&input[5 * stride]), 2);
  v_x6 = vshlq_n_s16(vld1q_s16(&input[6 * stride]), 2);
  v_x7 = vshlq_n_s16(vld1q_s16(&input[7 * stride]), 2);

  for (pass = 0; pass < 2; pass++) {
    if ((pass == 0 && (tx_type == ADST_ADST || tx_type == ADST_DCT))
        || (pass == 1 && (tx_type == ADST_ADST || tx_type == DCT_ADST))) {
      vp9_fadst8_neon(&v_x0, &v_x1, &v_x2, &v_x3, &v_x4, &v_x5, &v_x6, &v_x7);
    } else {
      vp9_fdct8_neon(&v_x0, &v_x1, &v_x2, &v_x3, &v_x4, &v_x5, &v_x6, &v_x7);
    }
    transpose_8x8_neon(&v_x0, &v_x1, &v_x2, &v_x3, &v_x4, &v_x5, &v_x6, &v_x7);
  }

  //output[j + i * 8] = (temp_out[j] + (temp_out[j] < 0)) >> 1;
  v_l0 = vshrq_n_u16(vreinterpretq_u16_s16(v_x0), 15);
  v_l1 = vshrq_n_u16(vreinterpretq_u16_s16(v_x1), 15);
  v_l2 = vshrq_n_u16(vreinterpretq_u16_s16(v_x2), 15);
  v_l3 = vshrq_n_u16(vreinterpretq_u16_s16(v_x3), 15);
  v_l4 = vshrq_n_u16(vreinterpretq_u16_s16(v_x4), 15);
  v_l5 = vshrq_n_u16(vreinterpretq_u16_s16(v_x5), 15);
  v_l6 = vshrq_n_u16(vreinterpretq_u16_s16(v_x6), 15);
  v_l7 = vshrq_n_u16(vreinterpretq_u16_s16(v_x7), 15);

  v_x0 = vaddq_s16(v_x0, vreinterpretq_s16_u16(v_l0));
  v_x1 = vaddq_s16(v_x1, vreinterpretq_s16_u16(v_l1));
  v_x2 = vaddq_s16(v_x2, vreinterpretq_s16_u16(v_l2));
  v_x3 = vaddq_s16(v_x3, vreinterpretq_s16_u16(v_l3));
  v_x4 = vaddq_s16(v_x4, vreinterpretq_s16_u16(v_l4));
  v_x5 = vaddq_s16(v_x5, vreinterpretq_s16_u16(v_l5));
  v_x6 = vaddq_s16(v_x6, vreinterpretq_s16_u16(v_l6));
  v_x7 = vaddq_s16(v_x7, vreinterpretq_s16_u16(v_l7));

  v_x0 = vshrq_n_s16(v_x0, 1);
  v_x1 = vshrq_n_s16(v_x1, 1);
  v_x2 = vshrq_n_s16(v_x2, 1);
  v_x3 = vshrq_n_s16(v_x3, 1);
  v_x4 = vshrq_n_s16(v_x4, 1);
  v_x5 = vshrq_n_s16(v_x5, 1);
  v_x6 = vshrq_n_s16(v_x6, 1);
  v_x7 = vshrq_n_s16(v_x7, 1);

  vst1q_s16(&output[0 * 8], v_x0);
  vst1q_s16(&output[1 * 8], v_x1);
  vst1q_s16(&output[2 * 8], v_x2);
  vst1q_s16(&output[3 * 8], v_x3);
  vst1q_s16(&output[4 * 8], v_x4);
  vst1q_s16(&output[5 * 8], v_x5);
  vst1q_s16(&output[6 * 8], v_x6);
  vst1q_s16(&output[7 * 8], v_x7);

  return;
}

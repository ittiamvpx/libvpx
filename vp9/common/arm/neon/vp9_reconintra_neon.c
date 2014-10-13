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

static INLINE void vp9_dc_predictor_neon(uint8_t *dst, ptrdiff_t y_stride,
                                         const int bs, const uint8_t *above,
                                         const uint8_t *left) {
  const uint16x4_t count_by_2 = { bs };
  // Using a Look-up table by dividing by count.
  const int16x4_t count_shift = { -4, -5, 0, -6 };
  int i, j;

  uint16x8_t sum = { 0 };
  uint16x4_t sum_padd;
  int16x4_t shift;
  uint8x8_t out;
  uint16x4_t expected_dc;

  for (i = 0; i < (bs / 8); i++) {
    sum = vaddw_u8(sum, vld1_u8(above));
    sum = vaddw_u8(sum, vld1_u8(left));
    above += 8;
    left += 8;
  }

  // Get the final sum, from vector
  sum_padd = vpadd_u16(vget_high_u16(sum), vget_low_u16(sum));
  sum_padd = vpadd_u16(sum_padd, sum_padd);
  sum_padd = vpadd_u16(sum_padd, sum_padd);

  // We are performing intermediate scalar computations in vector itself,
  // because the cost of moving the data between NEON register and ARM
  // register is costly in some platforms
  // expected_dc = (sum + (count >> 1)) / count;
  expected_dc = vadd_u16(sum_padd, count_by_2);
  // These look-up tables are optimized by the compiler, in this use-case
  shift = vdup_n_s16(count_shift[(bs / 8) - 1]);
  expected_dc = vshl_u16(expected_dc, shift);
  out = vdup_n_u8(expected_dc[0]);

  // These nested loops are unrolled and optimized by the compiler, in this
  // use-case
  for (i = 0; i < bs; i++) {
    for (j = 0; j < (bs / 8); j++) {
      vst1_u8(dst + (j * 8), out);
    }
    dst += y_stride;
  }
}

#define intra_pred_sized(size) \
  void vp9_dc_predictor_##size##x##size##_neon(uint8_t *dst, \
                                               ptrdiff_t stride, \
                                               const uint8_t *above, \
                                               const uint8_t *left) { \
    vp9_dc_predictor_neon(dst, stride, size, above, left); \
  }
intra_pred_sized(8)
intra_pred_sized(16)
intra_pred_sized(32)

#undef intra_pred_sized

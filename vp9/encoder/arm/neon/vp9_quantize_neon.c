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

#include <math.h>

#include "vpx_mem/vpx_mem.h"

#include "vp9/common/vp9_quant_common.h"
#include "vp9/common/vp9_seg_common.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_quantize.h"
#include "vp9/encoder/vp9_rd.h"


//------------------------------------------------------------------------------
//  Check whether absolute value of all the elements in a 4x4 sub-block is
//   greater than or equal to the given threshold value.
//
// Inputs
//   v_abs_coeff_0  - Absolute value of Row1 and Row2
//   v_abs_coeff_1  - Absolute value of Row3 and Row4
//   v_threshold    - Threshold value
// Outputs
//   v_cge_flag_0   - Is Row1 and Row2  greater than or equal to threshold
//   v_cge_flag_1   - Is Row3 and Row4  greater than or equal to threshold
//   is_all_zero    - Set to '0', if all the values in 4x4 is lesser
//                                                  than the threshold
//------------------------------------------------------------------------------
#define THRESHOLD_4X4_BLOCK(is_first_block) {                                  \
  v_cge_flag_0                  = vcgeq_s16(v_abs_coeff_0, v_threshold);       \
  if (is_first_block)                                                          \
    v_threshold                 = vdupq_lane_s16(vget_low_s16(v_threshold), 1);\
  v_cge_flag_1                  = vcgeq_s16(v_abs_coeff_1, v_threshold);       \
  {                                                                            \
    const uint16x8_t v_temp1    = vorrq_u16(v_cge_flag_0, v_cge_flag_1);       \
    const uint16x4_t v_temp2    = vorr_u16(vget_low_u16(v_temp1),              \
                                           vget_high_u16(v_temp1));            \
    const int32x2_t  v_temp3    = vpadd_s32(vreinterpret_s32_u16(v_temp2),     \
                                            vreinterpret_s32_u16(v_temp2));    \
    is_all_zero                 = vget_lane_s32(v_temp3, 0);                   \
  }                                                                            \
}
// END of THRESHOLD_4X4_BLOCK


// -----------------------------------------------------------------------------
// Find the maximum 16 bit value among iscan values
// Input
//   v_eobmax_7654321 - Maximum of iscan values corresponding to quantized
//                                                                       values
// Output
//   eob_ptr - Address to the variable with maximum iscan value
// -----------------------------------------------------------------------------

#define FIND_EOB                                                               \
  {                                                                            \
    const int16x4_t eob_max_tmp1 = vmax_s16(vget_low_s16(v_eobmax_76543210),   \
                                            vget_high_s16(v_eobmax_76543210)); \
    const int16x4_t eob_max_tmp2 = vext_s16(eob_max_tmp1, eob_max_tmp1, 2);    \
    const int16x4_t eob_max_tmp3 = vmax_s16(eob_max_tmp1, eob_max_tmp2);       \
    const int16x4_t eob_max_tmp4 = vext_s16(eob_max_tmp3, eob_max_tmp3, 1);    \
    const int16x4_t eob_max_tmp5 = vmax_s16(eob_max_tmp3, eob_max_tmp4);       \
    *eob_ptr                     = (uint16_t)vget_lane_s16(eob_max_tmp5, 0);   \
    if(*eob_ptr == (0xffff))                                                   \
      *eob_ptr = 0;                                                            \
  }// END of FIND_EOB





static inline void quantize_fp_neon(const int16_t *coeff_ptr,
                                    intptr_t count,
                                    int skip_block,
                                    const int16_t *round_ptr,
                                    const int16_t *quant_ptr,
                                    int16_t *qcoeff_ptr,
                                    int16_t *dqcoeff_ptr,
                                    const int16_t *dequant_ptr,
                                    uint16_t *eob_ptr,
                                    const int16_t *iscan,
                                    int width) {
  if (!skip_block) {
    // Quantization pass: All coefficients with index >= zero_flag are
    // skippable. Note: zero_flag can be zero.
    int i; int32_t  is_all_zero = 1;
    const int32_t  narrow_factor = (16 - (width >> 5));
    const int16x8_t v_zero = vdupq_n_s16(0);
    const int16x8_t v_one  = vdupq_n_s16(1);
    int16x8_t v_eobmax_76543210 = vdupq_n_s16(-1);
    int16x8_t v_round_1    = vmovq_n_s16(round_ptr[1]);
    int16x8_t v_quant_1    = vmovq_n_s16(quant_ptr[1]);
    int16x8_t v_dequant_1  = vmovq_n_s16(dequant_ptr[1]);
    int16x8_t v_round_0    = vsetq_lane_s16(round_ptr[0], v_round_1, 0);
    int16x8_t v_quant_0    = vsetq_lane_s16(quant_ptr[0], v_quant_1, 0);
    int16x8_t v_dequant_0  = vsetq_lane_s16(dequant_ptr[0], v_dequant_1, 0);
    int16x8_t v_threshold  = vshrq_n_s16(v_dequant_0, 2);
    uint16x8_t  v_cge_flag_0 = vdupq_n_u16(0xFFFF);
    uint16x8_t  v_cge_flag_1 = vdupq_n_u16(0xFFFF);
    if(width == 32){
      v_round_0 = vrshrq_n_s16(v_round_0, 1);
      v_round_1 = vrshrq_n_s16(v_round_1, 1);
    }
    // process dc and the first fifteen ac coeffs
    {
      const int16x8_t v_iscan_0     = vld1q_s16(&iscan[0]);
      const int16x8_t v_iscan_1     = vld1q_s16(&iscan[8]);
      const int16x8_t v_coeff_0     = vld1q_s16(&coeff_ptr[0]);
      const int16x8_t v_coeff_1     = vld1q_s16(&coeff_ptr[8]);
      const int16x8_t v_abs_coeff_0 = vabsq_s16(v_coeff_0);
      const int16x8_t v_abs_coeff_1 = vabsq_s16(v_coeff_1);
      if (width==32) {
        THRESHOLD_4X4_BLOCK(1)
      }
      if (is_all_zero == 0) {
         vst1q_s16(&qcoeff_ptr[0], v_zero);
         vst1q_s16(&dqcoeff_ptr[0], v_zero);
         vst1q_s16(&qcoeff_ptr[8], v_zero);
         vst1q_s16(&dqcoeff_ptr[8], v_zero);
       } else {
        // copy all the values greater than the threshold
        const int16x8_t v_coeff_gt_0   = vbslq_s16(v_cge_flag_0, v_abs_coeff_0,
                                                                       v_zero);
        const int16x8_t v_coeff_gt_1   = vbslq_s16(v_cge_flag_1, v_abs_coeff_1,
                                                                       v_zero);
        const int16x8_t v_coeff_sign_0 = vshrq_n_s16(v_coeff_0, 15);
        const int16x8_t v_coeff_sign_1 = vshrq_n_s16(v_coeff_1, 15);
        const int16x8_t v_tmp_0        = vqaddq_s16(v_coeff_gt_0,  v_round_0);
        const int16x8_t v_tmp_1        = vqaddq_s16(v_coeff_gt_1,  v_round_1);
        // multiplying with  quantization values
        const int32x4_t v_tmp_lo_0     = vmull_s16(vget_low_s16(v_tmp_0),
                                                   vget_low_s16(v_quant_0));
        const int32x4_t v_tmp_hi_0     = vmull_s16(vget_high_s16(v_tmp_0),
                                                   vget_high_s16(v_quant_0));
        const int32x4_t v_tmp_lo_1     = vmull_s16(vget_low_s16(v_tmp_1),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_1     = vmull_s16(vget_high_s16(v_tmp_1),
                                                   vget_high_s16(v_quant_1));
        //right shift by 15 for width = 32 or by 16 for rest,
                                    //and narrow down to 16bit
        const int16x8_t v_tmp2_0 = vcombine_s16(
                                        vshrn_n_s32(v_tmp_lo_0, narrow_factor),
                                        vshrn_n_s32(v_tmp_hi_0, narrow_factor));
        const int16x8_t v_tmp2_1 = vcombine_s16(
                                        vshrn_n_s32(v_tmp_lo_1, narrow_factor),
                                        vshrn_n_s32(v_tmp_hi_1, narrow_factor));
        const uint16x8_t v_nz_mask_0    = vceqq_s16(v_tmp2_0, v_zero);
        const uint16x8_t v_nz_mask_1    = vceqq_s16(v_tmp2_1, v_zero);
        const int16x8_t v_iscan_plus1_0 = vaddq_s16(v_iscan_0, v_one);
        const int16x8_t v_iscan_plus1_1 = vaddq_s16(v_iscan_1, v_one);
        // if quantized values > 0, corresponding iscan values are copied
        const int16x8_t v_nz_iscan_0    = vbslq_s16(v_nz_mask_0, v_zero,
                                                             v_iscan_plus1_0);
        const int16x8_t v_nz_iscan_1    = vbslq_s16(v_nz_mask_1, v_zero,
                                                              v_iscan_plus1_1);
        // restoring sign for quantized values
        const int16x8_t v_qcoeff_a_0    = veorq_s16(v_tmp2_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_a_1    = veorq_s16(v_tmp2_1, v_coeff_sign_1);
        const int16x8_t v_qcoeff_0   = vsubq_s16(v_qcoeff_a_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_1   = vsubq_s16(v_qcoeff_a_1, v_coeff_sign_1);
        const int16x8_t v_nz_max_tmp = vmaxq_s16(v_nz_iscan_0, v_nz_iscan_1);
        // multiplying quantized values with dequantization factors
        int16x8_t v_dqcoeff_0        = vmulq_s16(v_qcoeff_0, v_dequant_0);
        int16x8_t v_dqcoeff_1        = vmulq_s16(v_qcoeff_1, v_dequant_1);
        if(width ==32) {
          v_dqcoeff_0 = vhsubq_s16(v_dqcoeff_0, v_coeff_sign_0);
          v_dqcoeff_1 = vhsubq_s16(v_dqcoeff_1, v_coeff_sign_1);
        }
        v_eobmax_76543210 = vmaxq_s16(v_eobmax_76543210, v_nz_max_tmp);
        vst1q_s16(&qcoeff_ptr[0], v_qcoeff_0);
        vst1q_s16(&qcoeff_ptr[8], v_qcoeff_1);
        vst1q_s16(&dqcoeff_ptr[0], v_dqcoeff_0);
        vst1q_s16(&dqcoeff_ptr[8], v_dqcoeff_1);
      }
    }
    // now process the rest of the ac coeffs
    for (i = 16; i < count; i += 16) {
      const int16x8_t v_iscan_0     = vld1q_s16(&iscan[i]);
      const int16x8_t v_iscan_1     = vld1q_s16(&iscan[i+8]);
      const int16x8_t v_coeff_0     = vld1q_s16(&coeff_ptr[i]);
      const int16x8_t v_coeff_1     = vld1q_s16(&coeff_ptr[i+8]);
      const int16x8_t v_abs_coeff_0 = vabsq_s16(v_coeff_0);
      const int16x8_t v_abs_coeff_1 = vabsq_s16(v_coeff_1);
      if(width==32) {
        THRESHOLD_4X4_BLOCK(1)
      }
      if (is_all_zero == 0) {
        vst1q_s16(&qcoeff_ptr[i], v_zero);
        vst1q_s16(&dqcoeff_ptr[i], v_zero);
        vst1q_s16(&qcoeff_ptr[i+8], v_zero);
        vst1q_s16(&dqcoeff_ptr[i+8], v_zero);
      } else {
      // copy all the values greater than the threshold
        const int16x8_t v_coeff_gt_0   = vbslq_s16(v_cge_flag_0, v_abs_coeff_0,
                                                                       v_zero);
        const int16x8_t v_coeff_gt_1   = vbslq_s16(v_cge_flag_1, v_abs_coeff_1,
                                                                     v_zero);
        const int16x8_t v_coeff_sign_0 = vshrq_n_s16(v_coeff_0, 15);
        const int16x8_t v_coeff_sign_1 = vshrq_n_s16(v_coeff_1, 15);
        const int16x8_t v_tmp_0        = vqaddq_s16(v_coeff_gt_0,  v_round_1);
        const int16x8_t v_tmp_1        = vqaddq_s16(v_coeff_gt_1,  v_round_1);
        // multiplying with  quantization values
        const int32x4_t v_tmp_lo_0     = vmull_s16(vget_low_s16(v_tmp_0),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_0     = vmull_s16(vget_high_s16(v_tmp_0),
                                                   vget_high_s16(v_quant_1));
        const int32x4_t v_tmp_lo_1     = vmull_s16(vget_low_s16(v_tmp_1),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_1     = vmull_s16(vget_high_s16(v_tmp_1),
                                                   vget_high_s16(v_quant_1));
        //right shift by 15 for width = 32 or by 16 for rest,
                                    //and narrow down to 16bit
        const int16x8_t v_tmp2_0       = vcombine_s16(
                                        vshrn_n_s32(v_tmp_lo_0, narrow_factor),
                                        vshrn_n_s32(v_tmp_hi_0, narrow_factor));
        const int16x8_t v_tmp2_1       = vcombine_s16(
                                        vshrn_n_s32(v_tmp_lo_1, narrow_factor),
                                        vshrn_n_s32(v_tmp_hi_1, narrow_factor));
        const uint16x8_t v_nz_mask_0    = vceqq_s16(v_tmp2_0, v_zero);
        const uint16x8_t v_nz_mask_1    = vceqq_s16(v_tmp2_1, v_zero);
        const int16x8_t v_iscan_plus1_0 = vaddq_s16(v_iscan_0, v_one);
        const int16x8_t v_iscan_plus1_1 = vaddq_s16(v_iscan_1, v_one);
        // if quantized values > 0, corresponding iscan values are copied
        const int16x8_t v_nz_iscan_0    = vbslq_s16(v_nz_mask_0, v_zero,
                                                    v_iscan_plus1_0);
        const int16x8_t v_nz_iscan_1    = vbslq_s16(v_nz_mask_1, v_zero,
                                                    v_iscan_plus1_1);
        // restoring sign for quantized values
        const int16x8_t v_qcoeff_a_0    = veorq_s16(v_tmp2_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_a_1    = veorq_s16(v_tmp2_1, v_coeff_sign_1);
        const int16x8_t v_qcoeff_0    = vsubq_s16(v_qcoeff_a_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_1    = vsubq_s16(v_qcoeff_a_1, v_coeff_sign_1);
        const int16x8_t v_nz_max_tmp  = vmaxq_s16(v_nz_iscan_0, v_nz_iscan_1);
        // multiplying quantized values with dequantization factors
        int16x8_t v_dqcoeff_0         = vmulq_s16(v_qcoeff_0, v_dequant_1);
        int16x8_t v_dqcoeff_1         = vmulq_s16(v_qcoeff_1, v_dequant_1);
        if(width ==32) {
          v_dqcoeff_0 = vhsubq_s16(v_dqcoeff_0, v_coeff_sign_0);
          v_dqcoeff_1 = vhsubq_s16(v_dqcoeff_1, v_coeff_sign_1);
        }
        v_eobmax_76543210 = vmaxq_s16(v_eobmax_76543210, v_nz_max_tmp);
        vst1q_s16(&qcoeff_ptr[i], v_qcoeff_0);
        vst1q_s16(&qcoeff_ptr[i+8], v_qcoeff_1);
        vst1q_s16(&dqcoeff_ptr[i], v_dqcoeff_0);
        vst1q_s16(&dqcoeff_ptr[i+8], v_dqcoeff_1);
      }
    }
    FIND_EOB
  } else {
    vpx_memset(qcoeff_ptr, 0, count * sizeof(int16_t));
    vpx_memset(dqcoeff_ptr, 0, count * sizeof(int16_t));
    *eob_ptr = 0;
  }
}


static inline void quantize_b_neon(const tran_low_t *coeff_ptr, intptr_t count,
                                   int skip_block, const int16_t *zbin_ptr,
                                   const int16_t *round_ptr,
                                   const int16_t *quant_ptr,
                                   const int16_t *quant_shift_ptr,
                                   tran_low_t *qcoeff_ptr,
                                   tran_low_t *dqcoeff_ptr,
                                   const int16_t *dequant_ptr,
                                   int zbin_oq_value, uint16_t *eob_ptr,
                                   const int16_t *iscan, int width) {
  if (!skip_block) {
  // Quantization pass: All coefficients with index >= zero_flag are
  // skippable. Note: zero_flag can be zero.
    int i; int32_t  is_all_zero  = 1;
    const int32_t  narrow_factor = (16 - (width >> 5));
    const int16x8_t v_zero       = vdupq_n_s16(0);
    const int16x8_t v_one        = vdupq_n_s16(1);
    int16x8_t v_eobmax_76543210  = vdupq_n_s16(-1);
    int16x8_t v_round_1          = vmovq_n_s16(round_ptr[1]);
    int16x8_t v_quant_1          = vmovq_n_s16(quant_ptr[1]);
    int16x8_t v_dequant_1        = vmovq_n_s16(dequant_ptr[1]);
    int16x8_t v_round_0          = vsetq_lane_s16(round_ptr[0], v_round_1, 0);
    int16x8_t v_quant_0          = vsetq_lane_s16(quant_ptr[0], v_quant_1, 0);
    int16x8_t v_dequant_0      = vsetq_lane_s16(dequant_ptr[0], v_dequant_1, 0);
    int16x8_t v_temp16x8       = vdupq_n_s16(zbin_oq_value);
    int16x8_t v_zbin_t         = vmovq_n_s16(zbin_ptr[1]);
    int16x8_t v_zbin_t1        = vsetq_lane_s16(zbin_ptr[0], v_zbin_t, 0);
    int16x8_t v_threshold      = vqaddq_s16(v_zbin_t1, v_temp16x8);
    const int16x4_t v_quant_shift_d_1  = vmov_n_s16(quant_shift_ptr[1]);
    const int16x4_t v_quant_shift_d_0  = vset_lane_s16(quant_shift_ptr[0],
                                                       v_quant_shift_d_1, 0);
    const int32x4_t v_quant_shift_0    = vmovl_s16(v_quant_shift_d_0 );
    const int32x4_t v_quant_shift_1    = vmovl_s16(v_quant_shift_d_1 );
    if(width == 32) {
      v_threshold = vrshrq_n_s16(v_threshold, 1);
      v_round_0   = vrshrq_n_s16(v_round_0, 1);
      v_round_1   = vrshrq_n_s16(v_round_1, 1);
    }
    // process dc and the first fifteen ac coeffs
    {
      const int16x8_t v_iscan_0     = vld1q_s16(&iscan[0]);
      const int16x8_t v_iscan_1     = vld1q_s16(&iscan[8]);
      const int16x8_t v_coeff_0     = vld1q_s16(&coeff_ptr[0]);
      const int16x8_t v_coeff_1     = vld1q_s16(&coeff_ptr[8]);
      const int16x8_t v_abs_coeff_0 = vabsq_s16(v_coeff_0);
      const int16x8_t v_abs_coeff_1 = vabsq_s16(v_coeff_1);
      uint16x8_t  v_cge_flag_0;
      uint16x8_t  v_cge_flag_1;
      THRESHOLD_4X4_BLOCK(1)
      if (is_all_zero == 0) {
        vst1q_s16(&qcoeff_ptr[0], v_zero);
        vst1q_s16(&dqcoeff_ptr[0], v_zero);
        vst1q_s16(&qcoeff_ptr[8],  v_zero);
        vst1q_s16(&dqcoeff_ptr[8], v_zero);
      } else  {
        const int16x8_t v_coeff_gt_0   = vbslq_s16(v_cge_flag_0, v_abs_coeff_0,
                                                                        v_zero);
        const int16x8_t v_coeff_gt_1   = vbslq_s16(v_cge_flag_1, v_abs_coeff_1,
                                                                        v_zero);
        const int16x8_t v_coeff_sign_0 = vshrq_n_s16(v_coeff_0, 15);
        const int16x8_t v_coeff_sign_1 = vshrq_n_s16(v_coeff_1, 15);
        const int16x8_t v_tmp_0        = vqaddq_s16(v_coeff_gt_0,  v_round_0);
        const int16x8_t v_tmp_1        = vqaddq_s16(v_coeff_gt_1,  v_round_1);
        const int32x4_t v_tmp_lo_0     = vmull_s16(vget_low_s16(v_tmp_0),
                                                   vget_low_s16(v_quant_0));
        const int32x4_t v_tmp_hi_0     = vmull_s16(vget_high_s16(v_tmp_0),
                                                   vget_high_s16(v_quant_0));
        const int32x4_t v_tmp_lo_1     = vmull_s16(vget_low_s16(v_tmp_1),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_1     = vmull_s16(vget_high_s16(v_tmp_1),
                                                   vget_high_s16(v_quant_1));
        // right shift by 16, and narrow down to 16 bits
        const int16x8_t v_mul_tmp_0    = vcombine_s16(vshrn_n_s32(v_tmp_lo_0,
                                           16 ), vshrn_n_s32(v_tmp_hi_0,16));
        const int16x8_t v_mul_tmp_1    = vcombine_s16(vshrn_n_s32(v_tmp_lo_1,
                                           16), vshrn_n_s32(v_tmp_hi_1, 16));
        //add with rounded coeffs
        const int32x4_t v_add_tmp_lo_0 = vaddl_s16(vget_low_s16(v_mul_tmp_0 ),
                                                   vget_low_s16(v_tmp_0));
        const int32x4_t v_add_tmp_hi_0 = vaddl_s16(vget_high_s16(v_mul_tmp_0 ),
                                                   vget_high_s16(v_tmp_0));
        const int32x4_t v_add_tmp_lo_1 = vaddl_s16(vget_low_s16(v_mul_tmp_1 ),
                                                   vget_low_s16(v_tmp_1));
        const int32x4_t v_add_tmp_hi_1 = vaddl_s16(vget_high_s16(v_mul_tmp_1 ),
                                                   vget_high_s16(v_tmp_1));
        //multiplying with quant_shift value
        const int32x4_t v_mul_tmp_lo_0 = vmulq_s32(v_add_tmp_lo_0,
                                                   v_quant_shift_0);
        const int32x4_t v_mul_tmp_hi_0 = vmulq_s32(v_add_tmp_hi_0,
                                                   v_quant_shift_1);
        const int32x4_t v_mul_tmp_lo_1 = vmulq_s32(v_add_tmp_lo_1,
                                                   v_quant_shift_1);
        const int32x4_t v_mul_tmp_hi_1 = vmulq_s32(v_add_tmp_hi_1,
                                                   v_quant_shift_1);
        //right shift by 15 for width = 32 or by 16 for rest,
                                    //and narrow down to 16bit
        const int16x8_t v_tmp2_0 = vcombine_s16(
                                   vshrn_n_s32(v_mul_tmp_lo_0, narrow_factor),
                                   vshrn_n_s32(v_mul_tmp_hi_0, narrow_factor));
        const int16x8_t v_tmp2_1 = vcombine_s16(
                                   vshrn_n_s32(v_mul_tmp_lo_1, narrow_factor),
                                   vshrn_n_s32(v_mul_tmp_hi_1, narrow_factor));
        const uint16x8_t v_nz_mask_0    = vceqq_s16(v_tmp2_0, v_zero);
        const uint16x8_t v_nz_mask_1    = vceqq_s16(v_tmp2_1, v_zero);
        const int16x8_t v_iscan_plus1_0 = vaddq_s16(v_iscan_0, v_one);
        const int16x8_t v_iscan_plus1_1 = vaddq_s16(v_iscan_1, v_one);
        const int16x8_t v_nz_iscan_0 = vbslq_s16(v_nz_mask_0, v_zero,
                                                 v_iscan_plus1_0);
        const int16x8_t v_nz_iscan_1 = vbslq_s16(v_nz_mask_1, v_zero,
                                                 v_iscan_plus1_1);
        // restoring sign for quantized values
        const int16x8_t v_qcoeff_a_0 = veorq_s16(v_tmp2_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_a_1 = veorq_s16(v_tmp2_1, v_coeff_sign_1);
        const int16x8_t v_qcoeff_0   = vsubq_s16(v_qcoeff_a_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_1   = vsubq_s16(v_qcoeff_a_1, v_coeff_sign_1);
        const int16x8_t v_nz_max_tmp = vmaxq_s16(v_nz_iscan_0, v_nz_iscan_1);
        int16x8_t v_dqcoeff_0        = vmulq_s16(v_qcoeff_0, v_dequant_0);
        int16x8_t v_dqcoeff_1        = vmulq_s16(v_qcoeff_1, v_dequant_1);
        if(width ==32) {
          v_dqcoeff_0 = vhsubq_s16(v_dqcoeff_0, v_coeff_sign_0);
          v_dqcoeff_1 = vhsubq_s16(v_dqcoeff_1, v_coeff_sign_1);
        }
        v_eobmax_76543210 = vmaxq_s16(v_eobmax_76543210, v_nz_max_tmp);
        vst1q_s16(&qcoeff_ptr[0], v_qcoeff_0);
        vst1q_s16(&qcoeff_ptr[8], v_qcoeff_1);
        vst1q_s16(&dqcoeff_ptr[0], v_dqcoeff_0);
        vst1q_s16(&dqcoeff_ptr[8], v_dqcoeff_1);
      }
    }
    // now process the rest of the ac coeffs
    for (i = 16; i < count; i += 16) {
      const int16x8_t v_iscan_0     = vld1q_s16(&iscan[i]);
      const int16x8_t v_iscan_1     = vld1q_s16(&iscan[i+8]);
      const int16x8_t v_coeff_0     = vld1q_s16(&coeff_ptr[i]);
      const int16x8_t v_coeff_1     = vld1q_s16(&coeff_ptr[i+8]);
      const int16x8_t v_abs_coeff_0 = vabsq_s16(v_coeff_0);
      const int16x8_t v_abs_coeff_1 = vabsq_s16(v_coeff_1);
      uint16x8_t  v_cge_flag_0;
      uint16x8_t  v_cge_flag_1;
      THRESHOLD_4X4_BLOCK(0)
      if (is_all_zero == 0) {
        vst1q_s16(&qcoeff_ptr[i],   v_zero);
        vst1q_s16(&qcoeff_ptr[i+8], v_zero);
        vst1q_s16(&dqcoeff_ptr[i],   v_zero);
        vst1q_s16(&dqcoeff_ptr[i+8], v_zero);
      } else  {
        const int16x8_t v_coeff_gt_0   = vbslq_s16(v_cge_flag_0, v_abs_coeff_0,
                                                                       v_zero);
        const int16x8_t v_coeff_gt_1   = vbslq_s16(v_cge_flag_1, v_abs_coeff_1,
                                                                        v_zero);
        const int16x8_t v_coeff_sign_0 = vshrq_n_s16(v_coeff_0, 15);
        const int16x8_t v_coeff_sign_1 = vshrq_n_s16(v_coeff_1, 15);
        const int16x8_t v_tmp_0        = vqaddq_s16(v_coeff_gt_0,  v_round_1);
        const int16x8_t v_tmp_1        = vqaddq_s16(v_coeff_gt_1,  v_round_1);
        const int32x4_t v_tmp_lo_0     = vmull_s16(vget_low_s16(v_tmp_0),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_0     = vmull_s16(vget_high_s16(v_tmp_0),
                                                   vget_high_s16(v_quant_1));
        const int32x4_t v_tmp_lo_1     = vmull_s16(vget_low_s16(v_tmp_1),
                                                   vget_low_s16(v_quant_1));
        const int32x4_t v_tmp_hi_1     = vmull_s16(vget_high_s16(v_tmp_1),
                                                   vget_high_s16(v_quant_1));
        // right shift by 16, and narrow down to 16 bits
        const int16x8_t v_mul_tmp_0 = vcombine_s16(vshrn_n_s32(v_tmp_lo_0, 16),
                                                   vshrn_n_s32(v_tmp_hi_0, 16));
        const int16x8_t v_mul_tmp_1 = vcombine_s16(vshrn_n_s32(v_tmp_lo_1, 16),
                                                   vshrn_n_s32(v_tmp_hi_1, 16));
        //add with rounded coeffs
        const int32x4_t v_add_tmp_lo_0 = vaddl_s16(vget_low_s16(v_mul_tmp_0),
                                                   vget_low_s16(v_tmp_0));
        const int32x4_t v_add_tmp_hi_0 = vaddl_s16(vget_high_s16(v_mul_tmp_0),
                                                   vget_high_s16(v_tmp_0));
        const int32x4_t v_add_tmp_lo_1 = vaddl_s16(vget_low_s16(v_mul_tmp_1),
                                                   vget_low_s16(v_tmp_1));
        const int32x4_t v_add_tmp_hi_1 = vaddl_s16(vget_high_s16(v_mul_tmp_1),
                                                   vget_high_s16(v_tmp_1));
        //multiplying with quant_shift value
        const int32x4_t v_mul_tmp_lo_0 = vmulq_s32(v_add_tmp_lo_0,
                                                   v_quant_shift_1);
        const int32x4_t v_mul_tmp_hi_0 = vmulq_s32(v_add_tmp_hi_0,
                                                   v_quant_shift_1);
        const int32x4_t v_mul_tmp_lo_1 = vmulq_s32(v_add_tmp_lo_1,
                                                   v_quant_shift_1);
        const int32x4_t v_mul_tmp_hi_1 = vmulq_s32(v_add_tmp_hi_1,
                                                   v_quant_shift_1);
        //right shift by 15 for width = 32 or by 16 for rest,
                                    //and narrow down to 16bit
        const int16x8_t v_tmp2_0 = vcombine_s16(
                                   vshrn_n_s32(v_mul_tmp_lo_0, narrow_factor),
                                   vshrn_n_s32(v_mul_tmp_hi_0, narrow_factor));
        const int16x8_t v_tmp2_1 = vcombine_s16(
                                   vshrn_n_s32(v_mul_tmp_lo_1, narrow_factor),
                                   vshrn_n_s32(v_mul_tmp_hi_1, narrow_factor));
        const uint16x8_t v_nz_mask_0    = vceqq_s16(v_tmp2_0, v_zero);
        const uint16x8_t v_nz_mask_1    = vceqq_s16(v_tmp2_1, v_zero);
        const int16x8_t v_iscan_plus1_0 = vaddq_s16(v_iscan_0, v_one);
        const int16x8_t v_iscan_plus1_1 = vaddq_s16(v_iscan_1, v_one);
        const int16x8_t v_nz_iscan_0    = vbslq_s16(v_nz_mask_0, v_zero,
                                                    v_iscan_plus1_0);
        const int16x8_t v_nz_iscan_1    = vbslq_s16(v_nz_mask_1, v_zero,
                                                    v_iscan_plus1_1);
        // restoring sign for quantized values
        const int16x8_t v_qcoeff_a_0 = veorq_s16(v_tmp2_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_a_1 = veorq_s16(v_tmp2_1, v_coeff_sign_1);
        const int16x8_t v_qcoeff_0   = vsubq_s16(v_qcoeff_a_0, v_coeff_sign_0);
        const int16x8_t v_qcoeff_1   = vsubq_s16(v_qcoeff_a_1, v_coeff_sign_1);
        const int16x8_t v_nz_max_tmp = vmaxq_s16(v_nz_iscan_0, v_nz_iscan_1);
        int16x8_t v_dqcoeff_0        = vmulq_s16(v_qcoeff_0, v_dequant_1);
        int16x8_t v_dqcoeff_1        = vmulq_s16(v_qcoeff_1, v_dequant_1);
        if(width ==32) {
          v_dqcoeff_0 = vhsubq_s16(v_dqcoeff_0, v_coeff_sign_0);
          v_dqcoeff_1 = vhsubq_s16(v_dqcoeff_1, v_coeff_sign_1);
        }
        v_eobmax_76543210 = vmaxq_s16(v_eobmax_76543210, v_nz_max_tmp);
        vst1q_s16(&qcoeff_ptr[i], v_qcoeff_0);
        vst1q_s16(&qcoeff_ptr[i+8], v_qcoeff_1);
        vst1q_s16(&dqcoeff_ptr[i], v_dqcoeff_0);
        vst1q_s16(&dqcoeff_ptr[i+8], v_dqcoeff_1);
      }
    }
    FIND_EOB
  } else {
    vpx_memset(qcoeff_ptr, 0, count * sizeof(int16_t));
    vpx_memset(dqcoeff_ptr, 0, count * sizeof(int16_t));
    *eob_ptr = 0;
  }
}



#define QUANTIZE_FP(type,width)                                                \
void vp9_##type##_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,         \
                       int skip_block,const int16_t *zbin_ptr,                 \
                       const int16_t *round_ptr, const int16_t *quant_ptr,     \
                       const int16_t *quant_shift_ptr,                         \
                       tran_low_t *qcoeff_ptr, tran_low_t *dqcoeff_ptr,        \
                       const int16_t *dequant_ptr, int zbin_oq_value,          \
                       uint16_t *eob_ptr, const int16_t *scan,                 \
                       const int16_t *iscan) {                                 \
  (void)zbin_ptr;                                                              \
  (void)quant_shift_ptr;                                                       \
  (void)zbin_oq_value;                                                         \
  (void)scan;                                                                  \
quantize_fp_neon(coeff_ptr, n_coeffs, skip_block,round_ptr, quant_ptr,         \
                 qcoeff_ptr,dqcoeff_ptr, dequant_ptr,eob_ptr,iscan, width);    \
}

#define QUANTIZE_B(type,width)                                                 \
void vp9_##type##_neon(const tran_low_t *coeff_ptr, intptr_t n_coeffs,         \
                       int skip_block, const int16_t *zbin_ptr,                \
                       const int16_t *round_ptr, const int16_t *quant_ptr,     \
                       const int16_t *quant_shift_ptr, tran_low_t *qcoeff_ptr, \
                       tran_low_t *dqcoeff_ptr, const int16_t *dequant_ptr,    \
                       int zbin_oq_value, uint16_t *eob_ptr,                   \
                       const int16_t *scan, const int16_t *iscan) {            \
  (void)scan;                                                                  \
  quantize_b_neon(coeff_ptr,  n_coeffs, skip_block, zbin_ptr, round_ptr,       \
                  quant_ptr, quant_shift_ptr, qcoeff_ptr, dqcoeff_ptr,         \
                  dequant_ptr, zbin_oq_value, eob_ptr, iscan, width);          \
}

QUANTIZE_FP(quantize_fp,0);
QUANTIZE_FP(quantize_fp_32x32,32);
QUANTIZE_B(quantize_b,0);
QUANTIZE_B(quantize_b_32x32,32);











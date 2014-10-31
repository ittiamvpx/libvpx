/*
 *  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vp9_rtcd.h"
#include "vp9/common/vp9_common.h"

void vp9_idct16x16_256_add_neon_pass1(const int16_t *input,
                                      int16_t *output,
                                      int output_stride);
void vp9_idct16x16_256_add_neon_pass2(const int16_t *src,
                                      int16_t *output,
                                      int16_t *pass1Output,
                                      int16_t skip_adding,
                                      uint8_t *dest,
                                      int dest_stride);
void vp9_idct16x16_10_add_neon_pass1(const int16_t *input,
                                     int16_t *output,
                                     int output_stride);
void vp9_idct16x16_10_add_neon_pass2(const int16_t *src,
                                     int16_t *output,
                                     int16_t *pass1Output,
                                     int16_t skip_adding,
                                     uint8_t *dest,
                                     int dest_stride);
// Performs ADST for a 8x16 buffer
// This function should be called twice for full 16x16 transform
// The output pointer *dest points to
//    int16_t* when processed for row transform
//    uint8_t* when processed for column transform
// When boolean flag 'do_adding' is 1, the output is added to the 'dest' buffer
void vp9_iadst16x16_256_add_neon_single_pass(const int16_t *src,
                                             int16_t *temp_buffer,
                                             int  do_adding,
                                             void *dest,
                                             int dest_stride);


/* For ARM NEON, d8-d15 are callee-saved registers, and need to be saved. */
extern void vp9_push_neon(int64_t *store);
extern void vp9_pop_neon(int64_t *store);

void vp9_idct16x16_256_add_neon(const int16_t *input,
                                uint8_t *dest, int dest_stride) {
  int64_t store_reg[8];
  int16_t pass1_output[16*16] = {0};
  int16_t row_idct_output[16*16] = {0};

  // save d8-d15 register values.
  vp9_push_neon(store_reg);

  /* Parallel idct on the upper 8 rows */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(input, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7
  // which will be saved into row_idct_output.
  vp9_idct16x16_256_add_neon_pass2(input+1,
                                     row_idct_output,
                                     pass1_output,
                                     0,
                                     dest,
                                     dest_stride);

  /* Parallel idct on the lower 8 rows */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(input+8*16, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7
  // which will be saved into row_idct_output.
  vp9_idct16x16_256_add_neon_pass2(input+8*16+1,
                                     row_idct_output+8,
                                     pass1_output,
                                     0,
                                     dest,
                                     dest_stride);

  /* Parallel idct on the left 8 columns */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(row_idct_output, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7.
  // Then add the result to the destination data.
  vp9_idct16x16_256_add_neon_pass2(row_idct_output+1,
                                     row_idct_output,
                                     pass1_output,
                                     1,
                                     dest,
                                     dest_stride);

  /* Parallel idct on the right 8 columns */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(row_idct_output+8*16, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7.
  // Then add the result to the destination data.
  vp9_idct16x16_256_add_neon_pass2(row_idct_output+8*16+1,
                                     row_idct_output+8,
                                     pass1_output,
                                     1,
                                     dest+8,
                                     dest_stride);

  // restore d8-d15 register values.
  vp9_pop_neon(store_reg);

  return;
}

void vp9_idct16x16_10_add_neon(const int16_t *input,
                               uint8_t *dest, int dest_stride) {
  int64_t store_reg[8];
  int16_t pass1_output[16*16] = {0};
  int16_t row_idct_output[16*16] = {0};

  // save d8-d15 register values.
  vp9_push_neon(store_reg);

  /* Parallel idct on the upper 8 rows */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_10_add_neon_pass1(input, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7
  // which will be saved into row_idct_output.
  vp9_idct16x16_10_add_neon_pass2(input+1,
                                        row_idct_output,
                                        pass1_output,
                                        0,
                                        dest,
                                        dest_stride);

  /* Skip Parallel idct on the lower 8 rows as they are all 0s */

  /* Parallel idct on the left 8 columns */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(row_idct_output, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7.
  // Then add the result to the destination data.
  vp9_idct16x16_256_add_neon_pass2(row_idct_output+1,
                                     row_idct_output,
                                     pass1_output,
                                     1,
                                     dest,
                                     dest_stride);

  /* Parallel idct on the right 8 columns */
  // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
  // stage 6 result in pass1_output.
  vp9_idct16x16_256_add_neon_pass1(row_idct_output+8*16, pass1_output, 8);

  // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
  // with result in pass1(pass1_output) to calculate final result in stage 7.
  // Then add the result to the destination data.
  vp9_idct16x16_256_add_neon_pass2(row_idct_output+8*16+1,
                                     row_idct_output+8,
                                     pass1_output,
                                     1,
                                     dest+8,
                                     dest_stride);

  // restore d8-d15 register values.
  vp9_pop_neon(store_reg);

  return;
}
void vp9_iht16x16_256_add_neon(const int16_t *input, uint8_t *dest,
                               int dest_stride, int type) {
  int64_t store_reg[8];
  int16_t temp_buffer[16 * 16] = {0};
  int16_t row_iht_output[16 * 16] = {0};

  // save d8-d15 register values.
  vp9_push_neon  (store_reg);

  // Row transformation
  if (type == DCT_ADST || type == ADST_ADST) {
    // iadst for upper 8 rows
    // temp_buffer is used temporary buffer and result is
    // stored in row_iht_output.
    vp9_iadst16x16_256_add_neon_single_pass(input,
                                            temp_buffer,
                                            0,
                                            row_iht_output,
                                            32);
    // iadst for lower 8 rows
    vp9_iadst16x16_256_add_neon_single_pass(input + 128,
                                            temp_buffer + 8,
                                            0,
                                            row_iht_output + 8,
                                            32);

  } else {   // if (type == DCT_DCT || type ==ADST_DCT)
    /* Parallel idct on the upper 8 rows */
    // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
    // stage 6 result in temp_buffer.
    vp9_idct16x16_256_add_neon_pass1(input, temp_buffer, 8);

    // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
    // with result in pass1(temp_buffer) to calculate final result in stage 7
    // which will be saved into row_iht_output.
    vp9_idct16x16_256_add_neon_pass2(input + 1,
                                     row_iht_output,
                                     temp_buffer,
                                     0,
                                     dest,
                                     dest_stride);

    /* Parallel idct on the lower 8 rows */
    // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
    // stage 6 result in temp_buffer.
    vp9_idct16x16_256_add_neon_pass1(input + 8 * 16, temp_buffer, 8);

    // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
    // with result in pass1(temp_buffer) to calculate final result in stage 7
    // which will be saved into row_iht_output.
    vp9_idct16x16_256_add_neon_pass2(input + 8 * 16 + 1,
                                     row_iht_output + 8,
                                     temp_buffer,
                                     0, dest,
                                     dest_stride);

  }

  // Column Transformation
  if (type == ADST_DCT || type == ADST_ADST) {
    // iadst for upper 8 rows
    vp9_iadst16x16_256_add_neon_single_pass(row_iht_output,
                                            temp_buffer, 1,
                                            dest, dest_stride);
    // iadst for lower 8 rows
    vp9_iadst16x16_256_add_neon_single_pass(row_iht_output + 8 * 16,
                                            temp_buffer + 8,
                                            1,
                                            dest + 8,
                                            dest_stride);

  } else {   //if (type == DCT_DCT || type ==DCT_ADST)

    /* Parallel idct on the left 8 columns */
    // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
    // stage 6 result in temp_buffer.
    vp9_idct16x16_256_add_neon_pass1(row_iht_output, temp_buffer, 8);

    // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
    // with result in pass1(temp_buffer) to calculate final result in stage 7.
    // Then add the result to the destination data.
    vp9_idct16x16_256_add_neon_pass2(row_iht_output + 1,
                                     row_iht_output,
                                     temp_buffer,
                                     1,
                                     dest, dest_stride);

    /* Parallel idct on the right 8 columns */
    // First pass processes even elements 0, 2, 4, 6, 8, 10, 12, 14 and save the
    // stage 6 result in temp_buffer.
    vp9_idct16x16_256_add_neon_pass1(row_iht_output + 8 * 16, temp_buffer, 8);

    // Second pass processes odd elements 1, 3, 5, 7, 9, 11, 13, 15 and combines
    // with result in pass1(pass1_output) to calculate final result in stage 7.
    // Then add the result to the destination data.
    vp9_idct16x16_256_add_neon_pass2(row_iht_output + 8 * 16 + 1,
                                     row_iht_output + 8,
                                     temp_buffer,
                                     1,
                                     dest + 8,
                                     dest_stride);
  }
  // restore d8-d15 register values.
  vp9_pop_neon(store_reg);

  return;
}

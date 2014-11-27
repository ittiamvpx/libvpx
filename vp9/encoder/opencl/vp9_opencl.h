/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_OPENCL_H_
#define VP9_ENCODER_VP9_OPENCL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <vp9/encoder/opencl/CL/cl.h>

#define NUM_PIXELS_PER_WORKITEM 8

struct VP9_COMP;
typedef struct VP9_OPENCL {
  cl_command_queue cmd_queue;
  cl_context       context;
  cl_mem           reference_frame;
  size_t           reference_frame_size;
  cl_mem           current_frame;
  size_t           current_frame_size;
  // Input to the GPU
  cl_mem           input_mv[GPU_BLOCK_SIZES];
  size_t           input_mv_size[GPU_BLOCK_SIZES];
  void*            input_mv_mapped[GPU_BLOCK_SIZES];
  // Output from the GPU
  cl_mem           output_rd[GPU_BLOCK_SIZES];
  size_t           output_rd_size[GPU_BLOCK_SIZES];
  void*            output_rd_mapped[GPU_BLOCK_SIZES];
  cl_mem           rd_parameters;
  void*            rd_parameters_mapped;
  cl_kernel        vp9_pick_inter_mode_part1[GPU_BLOCK_SIZES];
  cl_kernel        vp9_pick_inter_mode_part2[GPU_BLOCK_SIZES];
  cl_kernel        vp9_pick_inter_mode_part3[GPU_BLOCK_SIZES];
  cl_int           mi_rows;
  cl_int           mi_cols;
  cl_int           stride;
} VP9_OPENCL;

int vp9_opencl_init(VP9_GPU *gpu);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* VP9_ENCODER_VP9_OPENCL_H_ */

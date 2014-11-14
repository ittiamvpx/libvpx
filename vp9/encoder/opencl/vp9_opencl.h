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
  cl_mem           input_mv[GPU_BLOCK_SIZES];
  size_t           input_mv_size[GPU_BLOCK_SIZES];
  void*            input_mv_mapped[GPU_BLOCK_SIZES];
  cl_mem           output_rd[GPU_BLOCK_SIZES];
  size_t           output_rd_size[GPU_BLOCK_SIZES];
  void*            output_rd_mapped[GPU_BLOCK_SIZES];
  cl_mem           rd_parameters;
  cl_kernel        inter_pred_and_rd_calc[GPU_BLOCK_SIZES];
} VP9_OPENCL;

int vp9_opencl_init(VP9_GPU *gpu);
void vp9_opencl_alloc_buffers(struct VP9_COMP *cpi);
void *vp9_opencl_acquire_input_buffer(struct VP9_COMP *cpi,
                                      GPU_BLOCK_SIZE gpu_bsize);
void vp9_opencl_execute(struct VP9_COMP *cpi,
                        uint8_t* reference_frame, uint8_t* current_frame,
                        GPU_INPUT *gpu_input, GPU_OUTPUT **gpu_output,
                        GPU_RD_PARAMETERS *gpu_rd_constants,
                        GPU_BLOCK_SIZE gpu_bsize);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* VP9_ENCODER_VP9_OPENCL_H_ */

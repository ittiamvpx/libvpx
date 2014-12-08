/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_EOPENCL_H_
#define VP9_ENCODER_VP9_EOPENCL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include <vp9/common/opencl/CL/cl.h>

#include "vp9/encoder/vp9_encoder.h"

#define NUM_PIXELS_PER_WORKITEM 8
#define NUM_KERNELS             5

typedef struct VP9_EOPENCL {
  VP9_OPENCL *opencl;

  opencl_buffer gpu_input[GPU_BLOCK_SIZES];
  cl_mem gpu_input_sub_buffer[GPU_BLOCK_SIZES][MAX_SUB_FRAMES];

  opencl_buffer gpu_output[GPU_BLOCK_SIZES];
  cl_mem gpu_output_sub_buffer[GPU_BLOCK_SIZES][MAX_SUB_FRAMES];

  opencl_buffer rdopt_parameters[GPU_BLOCK_SIZES];

  cl_kernel full_pixel_search[GPU_BLOCK_SIZES];
  cl_kernel rd_calculation_zeromv[GPU_BLOCK_SIZES];
  cl_kernel sub_pixel_search[GPU_BLOCK_SIZES];
  cl_kernel inter_prediction_and_sse[GPU_BLOCK_SIZES];
  cl_kernel rd_calculation[GPU_BLOCK_SIZES];
  cl_kernel vp9_is_8x8_required;

  cl_event event[MAX_SUB_FRAMES];
#if OPENCL_PROFILING
  cl_ulong total_time_taken[GPU_BLOCK_SIZES][NUM_KERNELS];
#endif
} VP9_EOPENCL;

int vp9_eopencl_init(VP9_COMP *cpi);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* VP9_ENCODER_VP9_EOPENCL_H_ */

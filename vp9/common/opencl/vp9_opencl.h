/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_COMMON_VP9_OPENCL_H_
#define VP9_COMMON_VP9_OPENCL_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/common/vp9_onyxc_int.h"
#include <vp9/common/opencl/CL/cl.h>

#define OPENCL_PROFILING 0

typedef struct opencl_buffer {
  cl_mem opencl_mem;
  void *mapped_pointer;
  int size;
}opencl_buffer;

typedef struct VP9_OPENCL {
  cl_device_id device;
  cl_context context;
  cl_command_queue cmd_queue;
} VP9_OPENCL;

int vp9_opencl_init(VP9_COMMON *cm);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif /* VP9_COMMON_VP9_OPENCL_H_ */

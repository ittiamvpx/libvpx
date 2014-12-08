/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_COMMON_VP9_GPU_H_
#define VP9_COMMON_VP9_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_OPENCL
#define CONFIG_GPU_COMPUTE 1
#else
#define CONFIG_GPU_COMPUTE 0
#endif

struct VP9Common;

typedef struct VP9GPU {
  void *compute_framework;

  void *(*alloc_frame_buffers)(struct VP9Common *cm, int frame_size,
                               void **opencl_mem);
  void (*release_frame_buffers)(struct VP9Common *cm, void **opencl_mem,
                                void **mapped_pointer);
  void (*acquire_frame_buffers)(struct VP9Common *cm, void **opencl_mem,
                                void **mapped_pointer, int size);
  void (*remove)(struct VP9Common *cm);
} VP9_GPU;

typedef struct gpu_cb_priv {
  struct VP9Common *cm;
  YV12_BUFFER_CONFIG *ybf;
} gpu_cb_priv;

#if CONFIG_GPU_COMPUTE

int vp9_gpu_get_frame_buffer(void *cb_priv, size_t min_size,
                             vpx_codec_frame_buffer_t *fb);
int vp9_gpu_free_frame_buffer(struct VP9Common *cm, YV12_BUFFER_CONFIG *ybf);
void vp9_acquire_frame_buffer(struct VP9Common *cm, YV12_BUFFER_CONFIG *ybf);

int vp9_gpu_init(struct VP9Common *cm);

#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_COMMON_VP9_GPU_H_

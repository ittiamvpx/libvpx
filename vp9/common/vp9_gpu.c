/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/common/vp9_onyxc_int.h"

#if CONFIG_OPENCL
#include "vp9/common/opencl/vp9_opencl.h"
#endif

#if CONFIG_GPU_COMPUTE

int vp9_gpu_get_frame_buffer(void *cb_priv, size_t min_size,
                             vpx_codec_frame_buffer_t *fb) {
  gpu_cb_priv *const priv = (gpu_cb_priv *)cb_priv;
  VP9_COMMON *cm = priv->cm;

  fb->data = cm->gpu.alloc_frame_buffers(cm, min_size,
                                         &priv->ybf->gpu_mem);
  priv->ybf->buffer_alloc_sz = min_size;
  fb->size = min_size;
  fb->priv = NULL;

  vpx_memset(fb->data, 0, min_size);

  return 0;
}

int vp9_gpu_free_frame_buffer(VP9_COMMON *cm, YV12_BUFFER_CONFIG *ybf) {
  if (ybf) {
    if (ybf->buffer_alloc_sz > 0) {
      cm->gpu.release_frame_buffers(cm, &ybf->gpu_mem,
                                    (void **)&ybf->buffer_alloc);
    }
    /* buffer_alloc isn't accessed by most functions.  Rather y_buffer,
     * u_buffer and v_buffer point to buffer_alloc and are used.  Clear out
     * all of this so that a freed pointer isn't inadvertently used */
    vpx_memset(ybf, 0, sizeof(YV12_BUFFER_CONFIG));
  } else {
    return -1;
  }

  return 0;
}

void vp9_acquire_frame_buffer(VP9_COMMON *cm, YV12_BUFFER_CONFIG *ybf) {
  void *host_ptr = ybf->buffer_alloc;

  cm->gpu.acquire_frame_buffers(cm, &ybf->gpu_mem,
                                (void **)&ybf->buffer_alloc,
                                ybf->buffer_alloc_sz);
  if (host_ptr != ybf->buffer_alloc) {
    // the host pointer is modified, correct the pointers in buffer config
    const int border = VP9_ENC_BORDER_IN_PIXELS;
    const uint64_t yplane_size = (ybf->y_height + 2 * border) *
        (uint64_t)ybf->y_stride;
    const int uv_border_w = border >> cm->subsampling_x;
    const int uv_border_h = border >> cm->subsampling_y;
    const int uv_height = ybf->y_height >> cm->subsampling_y;
    const uint64_t uvplane_size = (uv_height + 2 * uv_border_h) *
        (uint64_t)ybf->uv_stride;
    ybf->y_buffer = ybf->buffer_alloc + (border * ybf->y_stride) + border;
    ybf->u_buffer = ybf->buffer_alloc + yplane_size +
        (uv_border_h * ybf->uv_stride) + uv_border_w;
    ybf->v_buffer = ybf->buffer_alloc + yplane_size + uvplane_size +
        (uv_border_h * ybf->uv_stride) + uv_border_w;
  }
}

int vp9_gpu_init(VP9_COMMON *cm) {
#if CONFIG_OPENCL
  return vp9_opencl_init(cm);
#else
  return 1;
#endif
}

#endif

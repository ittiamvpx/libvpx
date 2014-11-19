/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_GPU_H_
#define VP9_ENCODER_VP9_GPU_H_

#ifdef __cplusplus
extern "C" {
#endif

#if CONFIG_OPENCL
#define CONFIG_GPU_COMPUTE 1
#else
#define CONFIG_GPU_COMPUTE 0
#endif

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_filter.h"
#include "vp9/common/vp9_mv.h"
#include "vp9/common/vp9_onyxc_int.h"

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_rd.h"

// Block sizes for which MV computations are done in GPU
typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32,
  GPU_BLOCK_16X16,
  GPU_BLOCK_8X8,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

#define GPU_INTER_MODES 2 // ZEROMV and NEWMV
struct VP9_COMP;

typedef struct GPU_INPUT {
  MV mv;
  INTERP_FILTER filter_type;
  int mode_context;
  int rate_mv;
  int do_newmv;
} GPU_INPUT;

typedef struct GPU_OUTPUT {
  int returnrate;
  int64_t returndistortion;
  int64_t best_rd;
  PREDICTION_MODE best_mode;
  INTERP_FILTER best_pred_filter;
  int skip_txfm;
  TX_SIZE tx_size;
} GPU_OUTPUT;

typedef struct GPU_RD_PARAMETERS {
  int rd_mult;
  int rd_div;
  int switchable_interp_costs[SWITCHABLE_FILTERS];
  unsigned int inter_mode_cost[INTER_MODE_CONTEXTS][GPU_INTER_MODES];
  int threshes[MAX_MODES];
  int thresh_fact_newmv;
  TX_MODE tx_mode;
  int dc_quant;
  int ac_quant;
} GPU_RD_PARAMETERS;

typedef struct VP9_GPU {
  void *compute_framework;
  void (*alloc_buffers)(struct VP9_COMP *cpi);
  void (*free_buffers)(struct VP9_COMP *cpi);
  void *(*acquire_input_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize);
  void (*execute)(struct VP9_COMP *cpi, uint8_t* reference_frame,
                  uint8_t* current_frame, GPU_INPUT *gpu_input,
                  GPU_OUTPUT **gpu_output, GPU_RD_PARAMETERS *gpu_rd_parameters,
                  GPU_BLOCK_SIZE gpu_bsize);
  void (*remove)(struct VP9_COMP *cpi);
  GPU_INPUT *gpu_input[GPU_BLOCK_SIZES];
} VP9_GPU;

extern const BLOCK_SIZE vp9_actual_block_size_lookup[GPU_BLOCK_SIZES];
extern const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES];

static INLINE BLOCK_SIZE get_actual_block_size(GPU_BLOCK_SIZE sb_type) {
  return vp9_actual_block_size_lookup[sb_type];
}

static INLINE GPU_BLOCK_SIZE get_gpu_block_size(BLOCK_SIZE sb_type) {
  return vp9_gpu_block_size_lookup[sb_type];
}

static INLINE int get_gpu_buffer_index(VP9_COMMON *const cm,
                                       int mi_row, int mi_col,
                                       BLOCK_SIZE bsize) {
  int group_stride = cm->width >> (b_width_log2(bsize) + 2);
  int bsl = mi_width_log2(bsize);
  return ((mi_row >> bsl) * group_stride) + (mi_col >> bsl);
}

int vp9_gpu_init(VP9_GPU *gpu);

void vp9_gpu_set_mvinfo_offsets(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                int mi_row, int mi_col, BLOCK_SIZE bsize);

void vp9_gpu_copy_rd_parameters(struct VP9_COMP *cpi, MACROBLOCK *const x,
                               GPU_RD_PARAMETERS *gpu_rd_constants,
                               GPU_BLOCK_SIZE gpu_bsize);

void vp9_gpu_copy_output(struct VP9_COMP *cpi, MACROBLOCK *const x,
                         GPU_BLOCK_SIZE gpu_bsize, GPU_OUTPUT *gpu_output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_GPU_H_

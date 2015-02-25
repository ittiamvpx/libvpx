/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ENCODER_VP9_EGPU_H_
#define VP9_ENCODER_VP9_EGPU_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "vp9/common/vp9_blockd.h"
#include "vp9/common/vp9_entropymode.h"
#include "vp9/common/vp9_enums.h"
#include "vp9/common/vp9_filter.h"
#include "vp9/common/vp9_mv.h"
#include "vp9/common/vp9_onyxc_int.h"

#include "vp9/encoder/vp9_block.h"
#include "vp9/encoder/vp9_rd.h"

#define GPU_INTER_MODES 2 // ZEROMV and NEWMV
#define MAX_SUB_FRAMES 4
#define CPU_SUB_FRAMES 0

// Block sizes for which MV computations are done in GPU
typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32,
  GPU_BLOCK_16X16,
  GPU_BLOCK_8X8,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

// Defines the last block size (from GPU_BLOCK_SIZE enum) that would run on GPU
// If the last GPU block size is 32x32, 16x16 block size would still run in data
// parallel mode on CPU without using the parent MV from 32x32.
#define LAST_GPU_BLOCK_SIZE GPU_BLOCK_32X32
#if CONFIG_GPU_COMPUTE
#define BLOCKS_PROCESSED_ON_GPU (LAST_GPU_BLOCK_SIZE + 1)
#else
#define BLOCKS_PROCESSED_ON_GPU (MAX(LAST_GPU_BLOCK_SIZE, GPU_BLOCK_16X16) + 1)
#endif

struct VP9_COMP;
struct macroblock;

struct GPU_INPUT {
  MV nearest_mv;
  MV near_mv;
  INTERP_FILTER filter_type;
  int mode_context;
  int rate_mv;
  int do_newmv;
  int do_compute;
} __attribute__ ((aligned(32)));
typedef struct GPU_INPUT GPU_INPUT;

struct GPU_OUTPUT {
  MV mv;
  int rate_mv;
  int sum[EIGHTTAP_SHARP + 1];
  unsigned int sse[EIGHTTAP_SHARP + 1];
  int returnrate;
  int64_t returndistortion;
  int64_t best_rd;
  PREDICTION_MODE best_mode;
  INTERP_FILTER best_pred_filter;
  int skip_txfm;
  TX_SIZE tx_size;
} __attribute__ ((aligned(32)));
typedef struct GPU_OUTPUT GPU_OUTPUT;

typedef struct GPU_RD_PARAMETERS {
  int rd_mult;
  int rd_div;
  int switchable_interp_costs[SWITCHABLE_FILTERS];
  unsigned int inter_mode_cost[INTER_MODE_CONTEXTS][GPU_INTER_MODES];
  int threshes[GPU_BLOCK_SIZES][MAX_MODES];
  int thresh_fact_newmv[GPU_BLOCK_SIZES];
  TX_MODE tx_mode;
  int dc_quant;
  int ac_quant;
  int nmvsadcost[2][MV_VALS];
  int mvcost[2][MV_VALS];
  int sad_per_bit;
  int error_per_bit;
  int nmvjointcost[MV_JOINTS];
} GPU_RD_PARAMETERS;

typedef struct SubFrameInfo {
  int mi_row_start, mi_row_end;
} SubFrameInfo;

typedef struct VP9_EGPU {
  void *compute_framework;
  GPU_INPUT *gpu_input[GPU_BLOCK_SIZES];

  void (*alloc_buffers)(struct VP9_COMP *cpi);
  void (*free_buffers)(struct VP9_COMP *cpi);
  void (*acquire_input_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                               void **host_ptr);
  void (*acquire_output_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                                void **host_ptr, int sub_frame_idx);
  void (*acquire_rd_param_buffer)(struct VP9_COMP *cpi, void **host_ptr);
  void (*enc_sync_read)(struct VP9_COMP *cpi, int event_id);
  void (*execute)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                  int sub_frame_idx);
  void (*remove)(struct VP9_COMP *cpi);
} VP9_EGPU;

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

void vp9_gpu_set_mvinfo_offsets(struct VP9_COMP *const cpi,
                                struct macroblock *const x,
                                int mi_row, int mi_col, BLOCK_SIZE bsize);

void vp9_find_mv_refs_rt(const VP9_COMMON *cm, const struct macroblock *x,
                         const TileInfo *const tile,
                         MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                         int_mv *mv_ref_list,
                         int mi_row, int mi_col);

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int row);
int vp9_get_subframe_index(const VP9_COMMON *cm, int mi_row);

void vp9_alloc_gpu_interface_buffers(struct VP9_COMP *cpi);
void vp9_free_gpu_interface_buffers(struct VP9_COMP *cpi);

#if CONFIG_GPU_COMPUTE

int vp9_egpu_init(struct VP9_COMP *cpi);

void vp9_gpu_mv_compute(struct VP9_COMP *cpi, struct macroblock *const x);

#endif

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // VP9_ENCODER_VP9_EGPU_H_

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
#define MAX_SUB_FRAMES 3
#define CPU_SUB_FRAMES 0

// Block sizes for which MV computations are done in GPU
typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32,
  GPU_BLOCK_16X16,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

struct VP9_COMP;
struct macroblock;

struct GPU_INPUT {
  MV nearest_mv;
  MV near_mv;
  char filter_type;
  char mode_context;
  char do_newmv;
  char do_compute;
};
typedef struct GPU_INPUT GPU_INPUT;

struct GPU_OUTPUT {
  int64_t returndistortion;
  int64_t best_rd;
  MV mv;
  int rate_mv;
  int returnrate;
  char best_mode;
  char best_pred_filter;
  char skip_txfm;
  char tx_size;
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
  GPU_RD_PARAMETERS *gpu_rd_parameters;
  void (*alloc_buffers)(struct VP9_COMP *cpi);
  void (*free_buffers)(struct VP9_COMP *cpi);
  void (*acquire_output_buffer)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                                void **host_ptr, int sub_frame_idx);
  void (*enc_sync_read)(struct VP9_COMP *cpi, int event_id);
  void (*execute)(struct VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                  int sub_frame_idx);
  void (*remove)(struct VP9_COMP *cpi);
  void (*prepare_control_buffers)(struct VP9_COMP *cpi);
  void (*frame_cache_sync)(struct VP9_COMP *cpi, YV12_BUFFER_CONFIG *yuv);
} VP9_EGPU;

extern const BLOCK_SIZE vp9_actual_block_size_lookup[GPU_BLOCK_SIZES];
extern const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES];

static INLINE BLOCK_SIZE get_actual_block_size(GPU_BLOCK_SIZE sb_type) {
  return vp9_actual_block_size_lookup[sb_type];
}

static INLINE GPU_BLOCK_SIZE get_gpu_block_size(BLOCK_SIZE sb_type) {
  return vp9_gpu_block_size_lookup[sb_type];
}

// If there are odd number of blocks in a row, then disable the filter search
// for the last block of the row (so that it is GPU-friendly).
static INLINE int vp9_gpu_is_filter_search_disabled(const VP9_COMMON *cm,
                                                    int mi_col,
                                                    BLOCK_SIZE bsize) {
  const int bsl = mi_width_log2(bsize);
  const int ms = num_8x8_blocks_wide_lookup[bsize];
  return (mi_col + ms + ms / 2 >= cm->mi_cols && !((mi_col >> bsl) & 1));
}

void vp9_set_gpu_block_sizes(struct VP9_COMP *const cpi);

int get_gpu_buffer_index(struct VP9_COMP *const cpi, int mi_row, int mi_col,
                         GPU_BLOCK_SIZE gpu_bsize);

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

/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "./vpx_config.h"
#include "vpx_mem/vpx_mem.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_gpu.h"

#if CONFIG_OPENCL
#include "vp9/encoder/opencl/vp9_opencl.h"
#endif

const BLOCK_SIZE vp9_actual_block_size_lookup[GPU_BLOCK_SIZES] = {
    BLOCK_32X32,
    BLOCK_16X16,
    BLOCK_8X8
};

const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES] = {
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_8X8,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_16X16,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_32X32,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
};

void vp9_gpu_set_mvinfo_offsets(VP9_COMMON *const cm, MACROBLOCKD *const xd,
                                int mi_row, int mi_col, BLOCK_SIZE bsize) {
  const int blocks_in_row = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
  const int block_index_row = (mi_row >> mi_height_log2(bsize));
  const int block_index_col = (mi_col >> mi_width_log2(bsize));
  xd->gpu_mvinfo[bsize] = cm->gpu_mvinfo_base_array[bsize] +
      (block_index_row * blocks_in_row) + block_index_col;
}

#if CONFIG_GPU_COMPUTE
int vp9_gpu_init(VP9_GPU *gpu) {
#if CONFIG_OPENCL
  return vp9_opencl_init(gpu);
#else
  return 1;
#endif
}


void vp9_gpu_copy_rd_parameters(VP9_COMP *cpi, MACROBLOCK *const x,
                                GPU_RD_PARAMETERS *gpu_rd_parameters,
                                GPU_BLOCK_SIZE gpu_bsize)
{
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblockd_plane *const pd = &xd->plane[0];
  BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  int i;

  gpu_rd_parameters->rd_mult = x->rdmult;
  gpu_rd_parameters->rd_div = x->rddiv;
  for (i = 0; i < SWITCHABLE_FILTERS; i++)
    gpu_rd_parameters->switchable_interp_costs[i] =
        cpi->switchable_interp_costs[SWITCHABLE_FILTERS][i];
  for (i = 0; i < INTER_MODE_CONTEXTS; i++) {
    gpu_rd_parameters->inter_mode_cost[i][0] =
        cpi->inter_mode_cost[i][INTER_OFFSET(ZEROMV)];
    gpu_rd_parameters->inter_mode_cost[i][1] =
        cpi->inter_mode_cost[i][INTER_OFFSET(NEWMV)];
  }
  // Assuming a segment ID of 0 here
  vpx_memcpy(gpu_rd_parameters->threshes, cpi->rd.threshes[0][bsize],
             sizeof(cpi->rd.threshes[0][bsize]));
  gpu_rd_parameters->tx_mode = cpi->common.tx_mode;
  gpu_rd_parameters->thresh_fact_newmv = cpi->rd.thresh_freq_fact[bsize][NEWMV];
  gpu_rd_parameters->dc_quant = pd->dequant[0];
  gpu_rd_parameters->ac_quant = pd->dequant[1];

}

// TODO(karthick-ittiam) : Ideally GPU_OUTPUT and GPU_MV_INFO structs should be
// merged/unified and this copy should be removed altogether
void vp9_gpu_copy_output(VP9_COMP *cpi, MACROBLOCK *const x,
                         GPU_BLOCK_SIZE gpu_bsize, GPU_OUTPUT *gpu_output)
{
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  int mi_row, mi_col;
  const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
  const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];

  for (mi_row = 0; mi_row < cm->mi_rows; mi_row += mi_row_step) {
    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += mi_col_step) {
      const int idx = get_gpu_buffer_index(cm, mi_row, mi_col, bsize);
      GPU_OUTPUT *const gpu_output_block = gpu_output + idx;
      const int ms = num_8x8_blocks_wide_lookup[bsize] / 2;
      const int force_horz_split = (mi_row + ms >= cm->mi_rows);
      const int force_vert_split = (mi_col + ms >= cm->mi_cols);
      if (force_horz_split | force_vert_split) {
        continue;
      }

      vp9_gpu_set_mvinfo_offsets(cm, xd, mi_row, mi_col,
          bsize);

      xd->gpu_mvinfo[bsize]->mode         = gpu_output_block->best_mode;
      xd->gpu_mvinfo[bsize]->best_rd      = gpu_output_block->best_rd;
      xd->gpu_mvinfo[bsize]->returnrate   = gpu_output_block->returnrate;
      xd->gpu_mvinfo[bsize]->returndistortion =
          gpu_output_block->returndistortion;
      xd->gpu_mvinfo[bsize]->interp_filter =
          gpu_output_block->best_pred_filter;
      xd->gpu_mvinfo[bsize]->skip_txfm    = gpu_output_block->skip_txfm;
      xd->gpu_mvinfo[bsize]->tx_size      = gpu_output_block->tx_size;
      // Assuming reference frame is only LAST_FRAME.
      xd->gpu_mvinfo[bsize]->ref_frame[0] = LAST_FRAME;
      // Assuming segment ID is 0.
      xd->gpu_mvinfo[bsize]->segment_id   = 0;
      if (xd->gpu_mvinfo[bsize]->mode == ZEROMV)
        xd->gpu_mvinfo[bsize]->mv[0].as_int = 0;
    }
  }
}
#endif

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

#include "vp9/common/vp9_mvref_common.h"

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

void vp9_gpu_fill_rd_parameters_common(VP9_COMP *cpi, MACROBLOCK *const x)
{
  VP9_GPU *gpu = &cpi->gpu;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblockd_plane *const pd = &xd->plane[0];
  GPU_RD_PARAMETERS *gpu_rd_parameters = gpu->acquire_rd_parameters(cpi);
  int i;

  gpu_rd_parameters->rd_mult = cpi->rd.RDMULT;
  gpu_rd_parameters->rd_div = cpi->rd.RDDIV;
  for (i = 0; i < SWITCHABLE_FILTERS; i++)
    gpu_rd_parameters->switchable_interp_costs[i] =
        cpi->switchable_interp_costs[SWITCHABLE_FILTERS][i];
  for (i = 0; i < INTER_MODE_CONTEXTS; i++) {
    gpu_rd_parameters->inter_mode_cost[i][0] =
        cpi->inter_mode_cost[i][INTER_OFFSET(ZEROMV)];
    gpu_rd_parameters->inter_mode_cost[i][1] =
        cpi->inter_mode_cost[i][INTER_OFFSET(NEWMV)];
  }
  gpu_rd_parameters->tx_mode = cpi->common.tx_mode;
  gpu_rd_parameters->dc_quant = pd->dequant[0];
  gpu_rd_parameters->ac_quant = pd->dequant[1];
  vpx_memcpy(gpu_rd_parameters->nmvsadcost[0], cpi->mb.nmvsadcost[0] - MV_MAX,
             sizeof(gpu_rd_parameters->nmvsadcost[0]));
  vpx_memcpy(gpu_rd_parameters->nmvsadcost[1], cpi->mb.nmvsadcost[1] - MV_MAX,
             sizeof(gpu_rd_parameters->nmvsadcost[1]));
  vpx_memcpy(gpu_rd_parameters->mvcost[0], cpi->mb.mvcost[0] - MV_MAX,
             sizeof(gpu_rd_parameters->mvcost[0]));
  vpx_memcpy(gpu_rd_parameters->mvcost[1], cpi->mb.mvcost[1] - MV_MAX,
             sizeof(gpu_rd_parameters->mvcost[1]));
  gpu_rd_parameters->sad_per_bit = cpi->mb.sadperbit16;
  gpu_rd_parameters->error_per_bit = cpi->mb.errorperbit;
}

void vp9_gpu_fill_rd_parameters_block(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize)
{
  VP9_GPU *gpu = &cpi->gpu;
  BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  GPU_RD_PARAMETERS *gpu_rd_parameters = gpu->acquire_rd_parameters(cpi);
  // Assuming a segment ID of 0 here
  vpx_memcpy(gpu_rd_parameters->threshes, cpi->rd.threshes[0][bsize],
             sizeof(cpi->rd.threshes[0][bsize]));
  gpu_rd_parameters->thresh_fact_newmv = cpi->rd.thresh_freq_fact[bsize][NEWMV];
}

void vp9_gpu_fill_mv_input(VP9_COMP *cpi, const TileInfo * const tile) {
  int mi_row, mi_col;
  int mi_width, mi_height;
  VP9_GPU *gpu = &cpi->gpu;
  MACROBLOCK *const x = &cpi->mb;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = &xd->mi[0]->mbmi;
  MV_REFERENCE_FRAME ref_frame = LAST_FRAME;
  int_mv *const candidates = mbmi->ref_mvs[ref_frame];
  GPU_INPUT_STAGE1 *value_meta_data;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; ++gpu_bsize) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
    const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];
    value_meta_data = gpu->acquire_input_buffer_stage1(cpi, gpu_bsize);

    for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end; mi_row +=
        mi_row_step) {
      for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end; mi_col +=
          mi_col_step) {
        GPU_INPUT_STAGE1 *meta_data = value_meta_data
            + get_gpu_buffer_index(cm, mi_row, mi_col, bsize);

        const int ms = num_8x8_blocks_wide_lookup[bsize] / 2;
        const int force_horz_split = (mi_row + ms >= cm->mi_rows);
        const int force_vert_split = (mi_col + ms >= cm->mi_cols);
        if (force_horz_split | force_vert_split) {
          continue;
        }

        mi_width = num_8x8_blocks_wide_lookup[bsize];
        mi_height = num_8x8_blocks_high_lookup[bsize];

        vpx_memset(mbmi, 0, sizeof(MB_MODE_INFO));

        vp9_init_plane_quantizers(cpi, x);

        set_mi_row_col(xd, tile, mi_row, mi_height, mi_col, mi_width,
                       cm->mi_rows, cm->mi_cols);

        mbmi->sb_type = bsize;
        mbmi->ref_frame[0] = NONE;
        mbmi->ref_frame[1] = NONE;
        mbmi->tx_size = MIN(max_txsize_lookup[bsize],
                            tx_mode_to_biggest_tx_size[cm->tx_mode]);

        mbmi->interp_filter =
            cm->interp_filter == SWITCHABLE ? EIGHTTAP : cm->interp_filter;

        vp9_find_mv_refs_rt(cm, x, tile, xd->mi[0], ref_frame, candidates,
                            mi_row, mi_col);

        vp9_find_best_ref_mvs(xd, cm->allow_high_precision_mv, candidates,
                              (int_mv *)&meta_data->nearest_mv,
                              (int_mv *)&meta_data->near_mv);

        clamp_mv2(&meta_data->nearest_mv, xd);
        clamp_mv2(&meta_data->near_mv, xd);

      }
    }
  }
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

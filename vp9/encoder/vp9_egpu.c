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
#include "vp9/common/vp9_alloccommon.h"
#include "vp9/common/vp9_onyxc_int.h"
#if CONFIG_OPENCL
#include "vp9/common/opencl/vp9_opencl.h"
#endif

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_encodeframe.h"
#if CONFIG_OPENCL
#include "vp9/encoder/opencl/vp9_eopencl.h"
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

int vp9_egpu_init(VP9_COMP *cpi) {
#if CONFIG_OPENCL
  return vp9_eopencl_init(cpi);
#else
  return 1;
#endif
}

static int get_subframe_offset(int idx, int mi_rows, int sb_rows) {
  const int offset = ((idx * sb_rows) / MAX_SUB_FRAMES) << MI_BLOCK_SIZE_LOG2;
  return MIN(offset, mi_rows);
}

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int idx) {
  subframe->mi_row_start = get_subframe_offset(idx, cm->mi_rows, cm->sb_rows);
  subframe->mi_row_end = get_subframe_offset(idx + 1, cm->mi_rows, cm->sb_rows);
}

int vp9_get_subframe_index(SubFrameInfo *subframe, const VP9_COMMON *cm,
                           int mi_row) {
  int idx;

  for (idx = 0; idx < MAX_SUB_FRAMES; ++idx) {
    vp9_subframe_init(subframe, cm, idx);
    if (mi_row >= subframe->mi_row_start && mi_row < subframe->mi_row_end) {
      break;
    }
  }
  assert(idx < MAX_SUB_FRAMES);
  return idx;
}

static void vp9_gpu_fill_rd_parameters(VP9_COMP *cpi, MACROBLOCK *const x) {
  VP9_EGPU *egpu = &cpi->egpu;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblockd_plane *const pd = &xd->plane[0];
  GPU_BLOCK_SIZE gpu_bsize;
  int i;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    GPU_RD_PARAMETERS *rd_param_ptr;

    egpu->acquire_rd_param_buffer(cpi, gpu_bsize, (void **)&rd_param_ptr);

    rd_param_ptr->rd_mult = cpi->rd.RDMULT;
    rd_param_ptr->rd_div = cpi->rd.RDDIV;
    for (i = 0; i < SWITCHABLE_FILTERS; i++)
      rd_param_ptr->switchable_interp_costs[i] =
          cpi->switchable_interp_costs[SWITCHABLE_FILTERS][i];
    for (i = 0; i < INTER_MODE_CONTEXTS; i++) {
      rd_param_ptr->inter_mode_cost[i][0] =
          cpi->inter_mode_cost[i][INTER_OFFSET(ZEROMV)];
      rd_param_ptr->inter_mode_cost[i][1] =
          cpi->inter_mode_cost[i][INTER_OFFSET(NEWMV)];
    }
    rd_param_ptr->tx_mode = cpi->common.tx_mode;
    rd_param_ptr->dc_quant = pd->dequant[0];
    rd_param_ptr->ac_quant = pd->dequant[1];
    vpx_memcpy(rd_param_ptr->nmvsadcost[0], cpi->mb.nmvsadcost[0] - MV_MAX,
               sizeof(rd_param_ptr->nmvsadcost[0]));
    vpx_memcpy(rd_param_ptr->nmvsadcost[1], cpi->mb.nmvsadcost[1] - MV_MAX,
               sizeof(rd_param_ptr->nmvsadcost[1]));
    vpx_memcpy(rd_param_ptr->mvcost[0], cpi->mb.mvcost[0] - MV_MAX,
               sizeof(rd_param_ptr->mvcost[0]));
    vpx_memcpy(rd_param_ptr->mvcost[1], cpi->mb.mvcost[1] - MV_MAX,
               sizeof(rd_param_ptr->mvcost[1]));
    rd_param_ptr->sad_per_bit = cpi->mb.sadperbit16;
    rd_param_ptr->error_per_bit = cpi->mb.errorperbit;
    for(i = 0; i < MV_JOINTS; i++) {
      rd_param_ptr->nmvjointcost[i] = x->nmvjointcost[i];
    }

    // assuming segmentation is disabled and segement id for the frame is '0'
    vpx_memcpy(rd_param_ptr->threshes, cpi->rd.threshes[0][bsize],
               sizeof(cpi->rd.threshes[0][bsize]));
    rd_param_ptr->thresh_fact_newmv = cpi->rd.thresh_freq_fact[bsize][NEWMV];
  }
}

static void vp9_gpu_fill_mv_input(VP9_COMP *cpi, const TileInfo * const tile) {
  SPEED_FEATURES *const sf = &cpi->sf;
  int mi_row, mi_col;
  int mi_width, mi_height;
  VP9_EGPU *egpu = &cpi->egpu;
  MACROBLOCK *const x = &cpi->mb;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = &xd->mi[0]->mbmi;
  MV_REFERENCE_FRAME ref_frame = LAST_FRAME;
  int_mv *const candidates = mbmi->ref_mvs[ref_frame];
  GPU_INPUT *gpu_input_base;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; ++gpu_bsize) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
    const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];

    egpu->acquire_input_buffer(cpi, gpu_bsize, (void **)&gpu_input_base);
    cpi->egpu.gpu_input[gpu_bsize] = gpu_input_base;

    for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end; mi_row +=
        mi_row_step) {
      for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end; mi_col +=
          mi_col_step) {
        GPU_INPUT *gpu_input = gpu_input_base
            + get_gpu_buffer_index(cm, mi_row, mi_col, bsize);
        const int bsl = mi_width_log2(bsize);
        const int pred_filter_search = cm->interp_filter == SWITCHABLE ?
            (((mi_row + mi_col) >> bsl) +
             get_chessboard_index(cm->current_video_frame)) & 0x1 : 0;

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
                              (int_mv *)&gpu_input->nearest_mv,
                              (int_mv *)&gpu_input->near_mv);

        clamp_mv2(&gpu_input->nearest_mv, xd);
        clamp_mv2(&gpu_input->near_mv, xd);

        gpu_input->do_newmv     = 1;
        gpu_input->do_compute   = 1;
        gpu_input->mode_context = mbmi->mode_context[ref_frame];

        if(pred_filter_search)
          gpu_input->filter_type = SWITCHABLE;
        else
          gpu_input->filter_type = EIGHTTAP;
      }
    }
  }

  for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end; mi_row +=
      MI_BLOCK_SIZE) {
    for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end; mi_col +=
        MI_BLOCK_SIZE) {
      const int is_static_area = is_background(cpi, tile, mi_row, mi_col);
      const int sb_row_index = mi_row / MI_SIZE;
      const int sb_col_index = mi_col / MI_SIZE;
      const int sb_index = (cm->sb_cols * sb_row_index + sb_col_index);

      cm->is_background_map[sb_index] = is_static_area;

      if (!sf->partition_check && is_static_area) {
        for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; ++gpu_bsize) {
          const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
          GPU_INPUT *gpu_input_base;
          const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
          const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];
          int block_row, block_col;

          egpu->acquire_input_buffer(cpi, gpu_bsize, (void **)&gpu_input_base);

          for (block_row = 0; block_row < MI_BLOCK_SIZE; block_row +=
              mi_row_step) {
            for (block_col = 0; block_col < MI_BLOCK_SIZE; block_col +=
                mi_col_step) {

              const int actual_mi_row = mi_row + block_row;
              const int actual_mi_col = mi_col + block_col;
              const int idx_str = cm->mi_stride * actual_mi_row + actual_mi_col;
              MODE_INFO **prev_mi = cm->prev_mi_grid_visible + idx_str;
              if(actual_mi_row >= cm->mi_rows || actual_mi_col >= cm->mi_cols)
                break;
              assert(prev_mi[0] != NULL);

              if (prev_mi[0]->mbmi.sb_type != bsize) {
                GPU_INPUT *gpu_input = gpu_input_base
                    + get_gpu_buffer_index(cm, actual_mi_row, actual_mi_col,
                                           bsize);
                gpu_input->do_compute = 0;
              }
            }
          }
        }
      }
    }
  }

}

// TODO(karthick-ittiam) : Ideally GPU_OUTPUT and GPU_MV_INFO structs should be
// merged/unified and this copy should be removed altogether
void vp9_gpu_copy_output_subframe(VP9_COMP *cpi, MACROBLOCK *const x,
                                  GPU_BLOCK_SIZE gpu_bsize,
                                  GPU_OUTPUT *gpu_output,
                                  SubFrameInfo *subframe) {
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  int mi_row, mi_col;
  const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
  const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];

  for (mi_row = subframe->mi_row_start; mi_row < subframe->mi_row_end;
       mi_row += mi_row_step) {
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
      else
        xd->gpu_mvinfo[bsize]->mv[0].as_mv  = gpu_output_block->mv;
    }
  }
}

void vp9_gpu_mv_compute(VP9_COMP *cpi, MACROBLOCK *const x) {
  VP9_COMMON *const cm = &cpi->common;
  const int tile_cols = 1 << cm->log2_tile_cols;
  int tile_col, tile_row;
  const int tile_rows = 1 << cm->log2_tile_rows;
  VP9_EGPU * const egpu = &cpi->egpu;
  GPU_BLOCK_SIZE gpu_bsize;
  int subframe_index;

  x->data_parallel_processing = 1;

  // fill rd param info
  vp9_gpu_fill_rd_parameters(cpi, x);
  // fill mv info
  for (tile_row = 0; tile_row < tile_rows; ++tile_row) {
    for (tile_col = 0; tile_col < tile_cols; ++tile_col) {
      TileInfo tile;

      vp9_tile_init(&tile, cm, tile_row, tile_col);
      vp9_gpu_fill_mv_input(cpi, &tile);
    }
  }
  // enqueue kernels for gpu
  for (subframe_index = 0; subframe_index < MAX_SUB_FRAMES; subframe_index++) {
    for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
      egpu->execute(cpi, gpu_bsize, subframe_index);
    }
  }
  // re-map source and reference pointers before starting cpu side processing
  vp9_acquire_frame_buffer(cm, cpi->Source);
  vp9_acquire_frame_buffer(cm, get_ref_frame_buffer(cpi, LAST_FRAME));

  x->data_parallel_processing = 0;
}

#endif

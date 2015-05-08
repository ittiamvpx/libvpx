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
};

const BLOCK_SIZE vp9_gpu_block_size_lookup[BLOCK_SIZES] = {
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
    GPU_BLOCK_INVALID,
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

const int gpu_mi_size_log2[GPU_BLOCK_SIZES] = {2, 1};

int get_gpu_buffer_index(VP9_COMP *const cpi, int mi_row, int mi_col,
                         GPU_BLOCK_SIZE gpu_bsize) {
  int group_stride = cpi->blocks_in_row[gpu_bsize];
  int bsl = gpu_mi_size_log2[gpu_bsize];
  return ((mi_row >> bsl) * group_stride) + (mi_col >> bsl);
}

void vp9_gpu_set_mvinfo_offsets(VP9_COMP *const cpi, MACROBLOCK *const x,
                                int mi_row, int mi_col, BLOCK_SIZE bsize) {
  const GPU_BLOCK_SIZE gpu_bsize = get_gpu_block_size(bsize);

  if (gpu_bsize != GPU_BLOCK_INVALID)
    x->gpu_output[gpu_bsize] = cpi->gpu_output_base[gpu_bsize] +
      get_gpu_buffer_index(cpi, mi_row, mi_col, gpu_bsize);
}

void vp9_find_mv_refs_rt(const VP9_COMMON *cm, const MACROBLOCK *x,
                         const TileInfo *const tile,
                         MODE_INFO *mi, MV_REFERENCE_FRAME ref_frame,
                         int_mv *mv_ref_list,
                         int mi_row, int mi_col) {
  find_mv_refs_idx(cm, &x->e_mbd, tile, mi, ref_frame, mv_ref_list, -1,
                   mi_row, mi_col, x->data_parallel_processing);
}

static int get_subframe_offset(int idx, int mi_rows, int sb_rows) {
  const int offset = ((idx * sb_rows) / MAX_SUB_FRAMES) << MI_BLOCK_SIZE_LOG2;
  return MIN(offset, mi_rows);
}

void vp9_subframe_init(SubFrameInfo *subframe, const VP9_COMMON *cm, int idx) {
  subframe->mi_row_start = get_subframe_offset(idx, cm->mi_rows, cm->sb_rows);
  subframe->mi_row_end = get_subframe_offset(idx + 1, cm->mi_rows, cm->sb_rows);
}

int vp9_get_subframe_index(const VP9_COMMON *cm, int mi_row) {
  int idx;

  for (idx = 0; idx < MAX_SUB_FRAMES; ++idx) {
    int mi_row_end = get_subframe_offset(idx + 1, cm->mi_rows, cm->sb_rows);
    if (mi_row < mi_row_end) {
      break;
    }
  }
  assert(idx < MAX_SUB_FRAMES);
  return idx;
}


void vp9_alloc_gpu_interface_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = GPU_BLOCK_32X32; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const int b_width_in_pixels_log2 = gpu_mi_size_log2[gpu_bsize] + 3;
    const int b_width_in_pixels = 1 << b_width_in_pixels_log2;
    const int ms_pixels = b_width_in_pixels / 2;
    const int b_width_mask = b_width_in_pixels - 1;
    int blocks_in_row = cm->width >> b_width_in_pixels_log2;
    // If width is not a multiple of block size
    if ((cm->width & b_width_mask) > ms_pixels)
      blocks_in_row++;
    cpi->blocks_in_row[gpu_bsize] = blocks_in_row;
  }

#if !CONFIG_GPU_COMPUTE
  for (gpu_bsize = GPU_BLOCK_32X32; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
#else
  cpi->egpu.alloc_buffers(cpi);
  if (LAST_GPU_BLOCK_SIZE >= GPU_BLOCK_16X16)
    return;
  // Allocate only for 16x16 block size, if required
  gpu_bsize = GPU_BLOCK_16X16;
  {
#endif
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int blocks_in_row = cpi->blocks_in_row[gpu_bsize];
    const int blocks_in_col = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);

    CHECK_MEM_ERROR(cm, cpi->gpu_output_base[gpu_bsize],
                    vpx_calloc(blocks_in_row * blocks_in_col,
                               sizeof(*cpi->gpu_output_base[gpu_bsize])));
  }
}

void vp9_free_gpu_interface_buffers(VP9_COMP *cpi) {
  GPU_BLOCK_SIZE gpu_bsize;

#if !CONFIG_GPU_COMPUTE
  for (gpu_bsize = GPU_BLOCK_32X32; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
#else
  cpi->egpu.remove(cpi);
  cpi->common.gpu.remove(&cpi->common);
  if (LAST_GPU_BLOCK_SIZE >= GPU_BLOCK_16X16)
    return;
  // Free only for 16x16 block size, if required
  gpu_bsize = GPU_BLOCK_16X16;
  {
#endif
    vpx_free(cpi->gpu_output_base[gpu_bsize]);
    cpi->gpu_output_base[gpu_bsize] = NULL;
  }
}

#if CONFIG_GPU_COMPUTE

int vp9_egpu_init(VP9_COMP *cpi) {
#if CONFIG_OPENCL
  return vp9_eopencl_init(cpi);
#else
  return 1;
#endif
}

static void vp9_gpu_fill_rd_parameters(VP9_COMP *cpi, MACROBLOCK *const x) {
  VP9_EGPU *egpu = &cpi->egpu;
  MACROBLOCKD *const xd = &x->e_mbd;
  struct macroblockd_plane *const pd = &xd->plane[0];
  GPU_BLOCK_SIZE gpu_bsize;
  GPU_RD_PARAMETERS *rd_param_ptr = egpu->gpu_rd_parameters;
  int i;

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
  for (i = 0; i < MV_JOINTS; i++) {
    rd_param_ptr->nmvjointcost[i] = x->nmvjointcost[i];
  }

  for (gpu_bsize = 0; gpu_bsize < BLOCKS_PROCESSED_ON_GPU; gpu_bsize++) {
    BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);

    // assuming segmentation is disabled and segement id for the frame is '0'
    vpx_memcpy(rd_param_ptr->threshes[gpu_bsize], cpi->rd.threshes[0][bsize],
               sizeof(cpi->rd.threshes[0][bsize]));
    rd_param_ptr->thresh_fact_newmv[gpu_bsize] =
        cpi->rd.thresh_freq_fact[bsize][NEWMV];
  }
}

static void vp9_gpu_fill_input_block(VP9_COMP *cpi, const TileInfo *const tile,
                                     GPU_INPUT *gpu_input,
                                     int mi_row, int mi_col,
                                     BLOCK_SIZE bsize) {

  MACROBLOCK *const x = &cpi->mb;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &x->e_mbd;
  MB_MODE_INFO *mbmi = &xd->mi[0]->mbmi;
  int mi_width, mi_height;
  const MV_REFERENCE_FRAME ref_frame = LAST_FRAME;
  int_mv *const candidates = mbmi->ref_mvs[ref_frame];
  const int bsl = mi_width_log2(bsize);
  int pred_filter_search = cm->interp_filter == SWITCHABLE ?
      (((mi_row + mi_col) >> bsl) +
       get_chessboard_index(cm->current_video_frame)) & 0x1 : 0;

  const int ms = num_8x8_blocks_wide_lookup[bsize] / 2;
  const int force_horz_split = (mi_row + ms >= cm->mi_rows);
  const int force_vert_split = (mi_col + ms >= cm->mi_cols);
  if (force_horz_split | force_vert_split) {
    return;
  }

  if (vp9_gpu_is_filter_search_disabled(cm, mi_col, bsize))
    pred_filter_search = 0;

  mi_width = num_8x8_blocks_wide_lookup[bsize];
  mi_height = num_8x8_blocks_high_lookup[bsize];

  set_mi_row_col(xd, tile, mi_row, mi_height,
                 mi_col, mi_width,
                 cm->mi_rows, cm->mi_cols);

  mbmi->sb_type = bsize;

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

  if (pred_filter_search)
    gpu_input->filter_type = SWITCHABLE;
  else
    gpu_input->filter_type = EIGHTTAP;

}

static void vp9_gpu_fill_mv_input(VP9_COMP *cpi, const TileInfo * const tile) {
  SPEED_FEATURES * const sf = &cpi->sf;
  int mi_row, mi_col;
  VP9_EGPU *egpu = &cpi->egpu;
  VP9_COMMON * const cm = &cpi->common;
  GPU_BLOCK_SIZE gpu_bsize;

  if (!sf->partition_check) {
    for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end; mi_row +=
        MI_BLOCK_SIZE) {
      int subframe_idx;

      subframe_idx = vp9_get_subframe_index(cm, mi_row);
      if (subframe_idx < CPU_SUB_FRAMES)
        continue;

      for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end; mi_col +=
          MI_BLOCK_SIZE) {
        const int sb_row_index = mi_row / MI_SIZE;
        const int sb_col_index = mi_col / MI_SIZE;
        const int sb_index = (cm->sb_cols * sb_row_index + sb_col_index);

        cm->is_background_map[sb_index] = is_background(cpi, tile, mi_row,
                                                        mi_col);
      }
    }
  }

  for (gpu_bsize = 0; gpu_bsize < BLOCKS_PROCESSED_ON_GPU; ++gpu_bsize) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int mi_row_step = num_8x8_blocks_high_lookup[bsize];
    const int mi_col_step = num_8x8_blocks_wide_lookup[bsize];
    GPU_INPUT *gpu_input_base = egpu->gpu_input[gpu_bsize];

    for (mi_row = tile->mi_row_start; mi_row < tile->mi_row_end; mi_row +=
        mi_row_step) {
      int subframe_idx;

      subframe_idx = vp9_get_subframe_index(cm, mi_row);
      if (subframe_idx < CPU_SUB_FRAMES)
        continue;

      for (mi_col = tile->mi_col_start; mi_col < tile->mi_col_end; mi_col +=
          mi_col_step) {
        GPU_INPUT *gpu_input = gpu_input_base
            + get_gpu_buffer_index(cpi, mi_row, mi_col, gpu_bsize);

        if (!sf->partition_check) {
          const int sb_row_index = mi_row / MI_SIZE;
          const int sb_col_index = mi_col / MI_SIZE;
          const int sb_index = (cm->sb_cols * sb_row_index + sb_col_index);
          const int is_static_area = cm->is_background_map[sb_index];
          const int idx_str = cm->mi_stride * mi_row + mi_col;
          MODE_INFO **prev_mi = cm->prev_mi_grid_visible + idx_str;
          assert(prev_mi[0] != NULL);

          if (prev_mi[0]->mbmi.sb_type != bsize && is_static_area) {
            const int bsl = mi_width_log2(bsize);
            int pred_filter_search = cm->interp_filter == SWITCHABLE ?
                (((mi_row + mi_col) >> bsl) +
                 get_chessboard_index(cm->current_video_frame)) & 0x1 : 0;

            if (vp9_gpu_is_filter_search_disabled(cm, mi_col, bsize))
              pred_filter_search = 0;

            if (pred_filter_search)
              gpu_input->filter_type = SWITCHABLE;
            else
              gpu_input->filter_type = EIGHTTAP;
            gpu_input->do_compute = 0;

            continue;
          }
        }
        vp9_gpu_fill_input_block(cpi, tile, gpu_input, mi_row, mi_col,
                                 bsize);
      }
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
  int subframe_idx;

  egpu->frame_cache_sync(cpi, cpi->Source);

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
  for (subframe_idx = CPU_SUB_FRAMES; subframe_idx < MAX_SUB_FRAMES;
       subframe_idx++) {
    for (gpu_bsize = 0; gpu_bsize < BLOCKS_PROCESSED_ON_GPU; gpu_bsize++) {
      egpu->execute(cpi, gpu_bsize, subframe_idx);
    }
  }
}

#endif

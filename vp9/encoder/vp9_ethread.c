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

#include "vp9/common/vp9_reconinter.h"

#include "vp9/encoder/vp9_encoder.h"
#include "vp9/encoder/vp9_encodeframe.h"

// synchronize encoder threads
void vp9_enc_sync_read(VP9_COMP *cpi, int sb_row, int sb_col) {
  const VP9_COMMON *const cm = &cpi->common;
  const volatile int *const top_sb_col = cpi->cur_sb_col + (sb_row - 1);

  // Check if the dependencies necessary to encode the current SB are
  // resolved. If the dependencies are resolved encode else do a busy wait.
  while (sb_row && !(sb_col & (cpi->sync_range - 1))) {
    // top right dependency
    int idx = sb_col + cpi->sync_range;

    idx = MIN(idx, (cm->sb_cols - 1));
    if (*top_sb_col >= idx)
      break;
    x86_pause_hint();
    thread_sleep(0);
  }
}

// synchronize encoder threads
void vp9_enc_sync_write(struct VP9_COMP *cpi, int sb_row) {
  int *const cur_sb_col = cpi->cur_sb_col + sb_row;

  // update the cur sb col
  (*cur_sb_col)++;
}

// Set up nsync by width.
// The optimal sync_range for different resolution and platform should be
// determined by testing. Currently, it is chosen to be a power-of-2 number.
static int get_sync_range(int width) {
  // TODO(ram-ittiam): nsync numbers have to be picked by testing
  if (width < 640)
    return 1;
  else if (width <= 1280)
    return 2;
  else if (width <= 4096)
    return 4;
  else
    return 8;
}

void vp9_create_encoding_threads(VP9_COMP *cpi) {
  VP9_COMMON * const cm = &cpi->common;
  const VP9WorkerInterface * const winterface = vp9_get_worker_interface();
  int i;

  CHECK_MEM_ERROR(cm, cpi->enc_thread_hndl,
                  vpx_malloc(sizeof(*cpi->enc_thread_hndl) * cpi->max_threads));
  for (i = 0; i < cpi->max_threads; ++i) {
    VP9Worker * const worker = &cpi->enc_thread_hndl[i];
    winterface->init(worker);
    CHECK_MEM_ERROR(cm, worker->data1,
                    vpx_memalign(32, sizeof(thread_context)));
    worker->data2 = NULL;
    if (i < cpi->max_threads - 1 && !winterface->reset(worker)) {
      vpx_internal_error(&cm->error, VPX_CODEC_ERROR,
                         "Tile decoder thread creation failed");
    }
  }
  // set row encoding hook
  for (i = 0; i < cpi->max_threads; ++i) {
    winterface->sync(&cpi->enc_thread_hndl[i]);
    cpi->enc_thread_hndl[i].hook = (VP9WorkerHook) encoding_thread_process;
  }
  CHECK_MEM_ERROR(cm, cpi->cur_sb_col,
                  vpx_malloc(sizeof(*cpi->cur_sb_col) * cm->sb_rows));
  // init cur sb col
  vpx_memset(cpi->cur_sb_col, -1, (sizeof(*cpi->cur_sb_col) * cm->sb_rows));
  // set up nsync (currently unused).
  cpi->sync_range = get_sync_range(cpi->oxcf.width);
}

void add_up_frame_counts(VP9_COMP *cpi, MACROBLOCK *x_thread) {
  VP9_COMMON *cm = &cpi->common;
  FRAME_COUNTS *dst = &cm->counts;
  FRAME_COUNTS *src = &x_thread->counts;
  int i, j, k;

  ADD_UP_2D_ARRAYS(dst->y_mode, src->y_mode, BLOCK_SIZE_GROUPS, INTRA_MODES);
  ADD_UP_2D_ARRAYS(dst->uv_mode, src->uv_mode, INTRA_MODES, INTRA_MODES);
  ADD_UP_2D_ARRAYS(dst->partition, src->partition, PARTITION_CONTEXTS,
                   PARTITION_TYPES);
  for (i = 0; i < TX_SIZES; ++i)
    for (j = 0; j < PLANE_TYPES; ++j)
      ADD_UP_3D_ARRAYS(dst->eob_branch[i][j], src->eob_branch[i][j], REF_TYPES,
                       COEF_BANDS, COEFF_CONTEXTS);
  ADD_UP_2D_ARRAYS(dst->switchable_interp, src->switchable_interp,
                   SWITCHABLE_FILTER_CONTEXTS, SWITCHABLE_FILTERS);
  ADD_UP_2D_ARRAYS(dst->inter_mode, src->inter_mode, INTER_MODE_CONTEXTS,
                   INTER_MODES);
  ADD_UP_2D_ARRAYS(dst->intra_inter, src->intra_inter, INTRA_INTER_CONTEXTS, 2);
  ADD_UP_2D_ARRAYS(dst->comp_inter, src->comp_inter, COMP_INTER_CONTEXTS, 2);
  ADD_UP_3D_ARRAYS(dst->single_ref, src->single_ref, REF_CONTEXTS, 2, 2);
  ADD_UP_2D_ARRAYS(dst->comp_ref, src->comp_ref, REF_CONTEXTS, 2);
  ADD_UP_2D_ARRAYS(dst->tx.p32x32, src->tx.p32x32, TX_SIZE_CONTEXTS, TX_SIZES);
  ADD_UP_2D_ARRAYS(dst->tx.p16x16, src->tx.p16x16, TX_SIZE_CONTEXTS,
                   TX_SIZES - 1);
  ADD_UP_2D_ARRAYS(dst->tx.p8x8, src->tx.p8x8, TX_SIZE_CONTEXTS, TX_SIZES - 2);
  ADD_UP_2D_ARRAYS(dst->skip, src->skip, SKIP_CONTEXTS, 2);
  ADD_UP_1D_ARRAYS(dst->mv.joints, src->mv.joints, MV_JOINTS);
  for (i = 0; i < 2; i++) {
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].sign, src->mv.comps[i].sign, 2);
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].classes, src->mv.comps[i].classes,
                     MV_CLASSES);
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].class0, src->mv.comps[i].class0,
                     CLASS0_SIZE);
    ADD_UP_2D_ARRAYS(dst->mv.comps[i].bits, src->mv.comps[i].bits,
                     MV_OFFSET_BITS, 2);
    ADD_UP_2D_ARRAYS(dst->mv.comps[i].class0_fp, src->mv.comps[i].class0_fp,
                     CLASS0_SIZE, MV_FP_SIZE);
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].fp, src->mv.comps[i].fp, MV_FP_SIZE);
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].class0_hp, src->mv.comps[i].class0_hp, 2);
    ADD_UP_1D_ARRAYS(dst->mv.comps[i].hp, src->mv.comps[i].hp, 2);
  }

  for (i = 0; i < TX_SIZES; ++i)
    for (j = 0; j < PLANE_TYPES; ++j)
      for (k = 0; k < REF_TYPES; ++k)
        ADD_UP_3D_ARRAYS(cpi->coef_counts[i][j][k],
                         x_thread->coef_counts[i][j][k], COEF_BANDS,
                         COEFF_CONTEXTS, ENTROPY_TOKENS);

  ADD_UP_1D_ARRAYS(cpi->rd.comp_pred_diff, x_thread->rd.comp_pred_diff,
                   REFERENCE_MODES);
  ADD_UP_1D_ARRAYS(cpi->rd.tx_select_diff, x_thread->rd.tx_select_diff,
                   TX_MODES);
  ADD_UP_1D_ARRAYS(cpi->rd.filter_diff, x_thread->rd.filter_diff,
                   SWITCHABLE_FILTER_CONTEXTS);
}

void vp9_mb_copy(VP9_COMP *cpi, MACROBLOCK *x_dst, MACROBLOCK *x_src) {
  VP9_COMMON *cm = &cpi->common;
  MACROBLOCKD *const xd_dst = &x_dst->e_mbd;
  MACROBLOCKD *const xd_src = &x_src->e_mbd;
  int i;

  for (i = 0; i < MAX_MB_PLANE; ++i) {
    x_dst->plane[i] = x_src->plane[i];
    xd_dst->plane[i] = xd_src->plane[i];
  }
  xd_dst->mi_stride = xd_src->mi_stride;
  xd_dst->mi = xd_src->mi;
  xd_dst->mi[0] = xd_src->mi[0];
  xd_dst->block_refs[0] = xd_src->block_refs[0];
  xd_dst->block_refs[1] = xd_src->block_refs[1];
  xd_dst->cur_buf = xd_src->cur_buf;
#if CONFIG_VP9_HIGHBITDEPTH
  xd_dst->bd = xd_src->bd;
#endif
  xd_dst->lossless = xd_src->lossless;
  xd_dst->corrupted = 0;
  for (i = 0; i < MAX_MB_PLANE; i++) {
    xd_dst->above_context[i] = xd_src->above_context[i];
  }
  xd_dst->above_seg_context = xd_src->above_seg_context;

  x_dst->skip_block = x_src->skip_block;
  x_dst->select_tx_size = x_src->select_tx_size;
  x_dst->skip_recode = x_src->skip_recode;
  x_dst->skip_optimize = x_src->skip_optimize;
  x_dst->q_index = x_src->q_index;

  x_dst->errorperbit = x_src->errorperbit;
  x_dst->sadperbit16 = x_src->sadperbit16;
  x_dst->sadperbit4 = x_src->sadperbit4;
  x_dst->rddiv = x_src->rddiv;
  x_dst->rdmult = x_src->rdmult;
  x_dst->mb_energy = x_src->mb_energy;

  for (i = 0; i < MV_JOINTS; i++) {
    x_dst->nmvjointcost[i] = x_src->nmvjointcost[i];
    x_dst->nmvjointsadcost[i] = x_src->nmvjointsadcost[i];
  }
  x_dst->nmvcost[0] = x_src->nmvcost[0];
  x_dst->nmvcost[1] = x_src->nmvcost[1];
  x_dst->nmvcost_hp[0] = x_src->nmvcost_hp[0];
  x_dst->nmvcost_hp[1] = x_src->nmvcost_hp[1];
  x_dst->mvcost = x_src->mvcost;
  x_dst->nmvsadcost[0] = x_src->nmvsadcost[0];
  x_dst->nmvsadcost[1] = x_src->nmvsadcost[1];
  x_dst->nmvsadcost_hp[0] = x_src->nmvsadcost_hp[0];
  x_dst->nmvsadcost_hp[1] = x_src->nmvsadcost_hp[1];
  x_dst->mvsadcost = x_src->mvsadcost;

  x_dst->min_partition_size = x_src->min_partition_size;
  x_dst->max_partition_size = x_src->max_partition_size;

  vpx_memcpy(x_dst->token_costs, x_src->token_costs,
             sizeof(x_src->token_costs));

  vp9_zero(x_dst->counts);
  vp9_zero(x_dst->coef_counts);
  vpx_memcpy(x_dst->rd.threshes, cpi->rd.threshes, sizeof(cpi->rd.threshes));
  // freq scaling factors initialization has to happen only for video frame 1.
  // For all other frames, It self corrects itself while encoding.
  if (cm->current_video_frame == 0)
    vpx_memcpy(x_dst->rd.thresh_freq_fact, cpi->rd.thresh_freq_fact,
               sizeof(cpi->rd.thresh_freq_fact));
  vp9_zero(x_dst->rd.comp_pred_diff);
  vp9_zero(x_dst->rd.tx_select_diff);
  vp9_zero(x_dst->rd.tx_select_threshes);
  vp9_zero(x_dst->rd.filter_diff);
  x_dst->rd.RDMULT = cpi->rd.RDMULT;
  x_dst->rd.RDDIV = cpi->rd.RDDIV;

  x_dst->data_parallel_processing = 0;

  x_dst->optimize = x_src->optimize;
  x_dst->quant_fp = x_src->quant_fp;
  vp9_zero(x_dst->skip_txfm);
  vp9_zero(x_dst->bsse);

  x_dst->fwd_txm4x4 = x_src->fwd_txm4x4;
  x_dst->itxm_add = x_src->itxm_add;
#if CONFIG_VP9_HIGHBITDEPTH
  x_dst->itxm_add = x_src->itxm_add;
#endif
}

// Row-based multi-threaded loopfilter hook
static int loop_filter_row_worker(thread_context *const thread_ctxt,
                                  void *unused) {
  VP9_COMP *const cpi = thread_ctxt->cpi;
  VP9_COMMON *const cm = &cpi->common;
  MACROBLOCKD *const xd = &cpi->mb.e_mbd;
  const YV12_BUFFER_CONFIG *const frame_buffer = cm->frame_to_show;
  struct macroblockd_plane planes[MAX_MB_PLANE];
  const int num_planes = thread_ctxt->y_only ? 1 : MAX_MB_PLANE;
  int mi_row, mi_col;

  (void)unused;
  vp9_copy(planes, xd->plane);
  for (mi_row = thread_ctxt->mi_row_start; mi_row < thread_ctxt->mi_row_end;
       mi_row += thread_ctxt->mi_row_step) {
    const int sb_row = mi_row >> MI_BLOCK_SIZE_LOG2;
    MODE_INFO **const mi = cm->mi_grid_visible + mi_row * cm->mi_stride;

    for (mi_col = 0; mi_col < cm->mi_cols; mi_col += MI_BLOCK_SIZE) {
      const int sb_col = mi_col >> MI_BLOCK_SIZE_LOG2;
      LOOP_FILTER_MASK lfm;
      int plane;

      vp9_enc_sync_read(cpi, sb_row, sb_col);

      vp9_setup_dst_planes(planes, frame_buffer, mi_row, mi_col);
      vp9_setup_mask(cm, mi_row, mi_col, mi + mi_col, cm->mi_stride, &lfm);

      for (plane = 0; plane < num_planes; ++plane) {
        vp9_filter_block_plane(cm, &planes[plane], mi_row, &lfm);
      }

      vp9_enc_sync_write(cpi, sb_row);
    }
  }

  return 1;
}

// VP9 Encoder: Implement multi-threaded loopfilter that uses the threads
// used for encoding.
void vp9e_loop_filter_frame_mt(VP9_COMP *cpi, int frame_filter_level,
                               int y_only, int partial_frame) {
  VP9_COMMON *const cm = &cpi->common;
  const int sb_rows = cm->sb_rows;
  const int num_threads = cpi->max_threads;
  const VP9WorkerInterface *const winterface = vp9_get_worker_interface();
  int thread_id;
  int start_mi_row = 0, end_mi_row, mi_rows_to_filter = cm->mi_rows;

  if (!frame_filter_level)
      return;

  vp9_loop_filter_frame_init(cm, frame_filter_level);

  // Initialize cur_sb_col to -1 for all SB rows.
  vpx_memset(cpi->cur_sb_col, -1, sizeof(*cpi->cur_sb_col) * sb_rows);

  if (partial_frame && cm->mi_rows > 8) {
    int i;
    start_mi_row = cm->mi_rows >> 1;
    start_mi_row &= 0xfffffff8;
    mi_rows_to_filter = MAX(cm->mi_rows / 8, 8);

    // Initialize cur_sb_col to sb_cols for top SB rows indicating
    // that deblocking is done.
    for (i = 0; i < start_mi_row >> MI_BLOCK_SIZE_LOG2; i++)
      cpi->cur_sb_col[i] = cm->sb_cols - 1;
  }
  end_mi_row = start_mi_row + mi_rows_to_filter;

  // Set up loopfilter thread data.
  for (thread_id = 0; thread_id < num_threads; ++thread_id) {
    VP9Worker *const worker = &cpi->enc_thread_hndl[thread_id];
    thread_context *const thread_ctxt = (thread_context *)worker->data1;

    worker->hook = (VP9WorkerHook)loop_filter_row_worker;

    // initialize thread context
    thread_ctxt->cpi = cpi;

    // thread start sb row
    thread_ctxt->mi_row_start = start_mi_row + MI_BLOCK_SIZE * thread_id;
    // thread end sb row
    thread_ctxt->mi_row_end = end_mi_row;
    // thread step sb row
    thread_ctxt->mi_row_step = MI_BLOCK_SIZE * cpi->max_threads;

    //thread id
    thread_ctxt->thread_id = thread_id;

    // yonly
    thread_ctxt->y_only = y_only;

    // Start loopfiltering
    if (thread_id == num_threads - 1) {
      winterface->execute(worker);
    } else {
      winterface->launch(worker);
    }
  }

  // Wait till all rows are finished
  for (thread_id = 0; thread_id < num_threads; ++thread_id) {
    winterface->sync(&cpi->enc_thread_hndl[thread_id]);
  }
}

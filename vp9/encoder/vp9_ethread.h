/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#ifndef VP9_ETHREAD_H_VP9_ETHREAD_H_
#define VP9_ETHREAD_H_VP9_ETHREAD_H_

/* Thread management macros */
#ifdef _WIN32
  /* Win32 */
  #include <windows.h>
  #define thread_sleep(nms) Sleep(nms)
#elif defined(__OS2__)
  /* OS/2 */
  #define INCL_DOS
  #include <os2.h>
  #define thread_sleep(nms) DosSleep(nms)
#else
  /* POSIX */
  #include <sched.h>
  #define thread_sleep(nms) sched_yield();
#endif

#if ARCH_X86 || ARCH_X86_64
  #include "vpx_ports/x86.h"
#else
  #define x86_pause_hint()
#endif

struct VP9_COMP;
struct macroblock;

#define ADD_UP_1D_ARRAYS(op1, op2, col) {\
  int a;\
  for (a = 0; a < col; ++a) {\
    op1[a] += op2[a];\
  }\
}

#define ADD_UP_2D_ARRAYS(op1, op2, row, col) {\
  int b;\
  for (b = 0; b < row; ++b) {\
    ADD_UP_1D_ARRAYS(op1[b], op2[b], col);\
  }\
}

#define ADD_UP_3D_ARRAYS(op1, op2, level, row, col) {\
  int c;\
  for (c = 0; c < level; ++c) {\
    ADD_UP_2D_ARRAYS(op1[c], op2[c], row, col);\
  }\
}

typedef struct thread_context {
  struct VP9_COMP *cpi;

  // thread specific mb context
  DECLARE_ALIGNED(16, struct macroblock, mb);

  // threads shall process rows of the video frame. Below params represent
  // the list of row id's the thread processes
  int mi_row_start, mi_row_end;
  int mi_row_step;

  // thread id
  int thread_id;

  // used by loop filter threads to determine if only y plane needs to be
  // filtered or all mb planes have to be filtered
  int y_only;
} thread_context;

void vp9_enc_sync_read(struct VP9_COMP *cpi, int sb_row, int sb_col);

void vp9_enc_sync_write(struct VP9_COMP *cpi, int sb_row);

void vp9_create_encoding_threads(struct VP9_COMP *cpi);

void add_up_frame_counts(struct VP9_COMP *cpi, struct macroblock *x_thread);

void vp9_mb_copy(struct VP9_COMP *cpi, struct macroblock *x_dst,
                 struct macroblock *x_src);

void vp9e_loop_filter_frame_mt(struct VP9_COMP *cpi, int frame_filter_level,
                               int y_only, int partial_frame);

#endif /* VP9_ETHREAD_H_ */

/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


typedef enum {
  MV_JOINT_ZERO = 0,             /* Zero vector */
  MV_JOINT_HNZVZ = 1,            /* Vert zero, hor nonzero */
  MV_JOINT_HZVNZ = 2,            /* Hor zero, vert nonzero */
  MV_JOINT_HNZVNZ = 3,           /* Both components nonzero */
} MV_JOINT_TYPE;

typedef enum {
  ONLY_4X4            = 0,        // only 4x4 transform used
  ALLOW_8X8           = 1,        // allow block transform size up to 8x8
  ALLOW_16X16         = 2,        // allow block transform size up to 16x16
  ALLOW_32X32         = 3,        // allow block transform size up to 32x32
  TX_MODE_SELECT      = 4,        // transform specified for each block
  TX_MODES            = 5,
} TX_MODE;


#define CL_INT_MAX          2147483647
#define VP9_ENC_BORDER_IN_PIXELS    160
#define MAX_PATTERN_SCALES 2
#define NUM_PIXELS_PER_WORKITEM 8
#define PATTERN_CANDIDATES_REF 3
#define MAX_PATTERN_CANDIDATES 8
#define MV_JOINTS 4
#define MI_SIZE 8
#define VP9_INTERP_EXTEND 4
#define SUBPEL_BITS 4
#define FILTER_BITS 7
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define SUBPEL_SHIFTS (1 << SUBPEL_BITS)
#define ROUND_POWER_OF_TWO(value, n) (((value) + (1 << ((n) - 1))) >> (n))
#define MV_CLASSES     11
#define CLASS0_BITS    1  /* bits at integer precision for class 0 */
#define MV_MAX_BITS    (MV_CLASSES + CLASS0_BITS + 2)
#define MV_MAX         ((1 << MV_MAX_BITS) - 1)
#define MV_VALS        ((MV_MAX << 1) + 1)
#define SUBPEL_TAPS 8
#define MAX_MVSEARCH_STEPS 11
#define MAX_FULL_PEL_VAL ((1 << (MAX_MVSEARCH_STEPS - 1)) - 1)
#define MV_IN_USE_BITS 14
#define MV_UPP   ((1 << MV_IN_USE_BITS) - 1)
#define MV_LOW   (-(1 << MV_IN_USE_BITS))
#define INTER_MODE_CONTEXTS 7
#define GPU_INTER_MODES 2 // ZEROMV and NEWMV
#define SWITCHABLE_FILTERS 3   // number of switchable filters
#define MAX_MODES 30
#define MV_COST_WEIGHT      108

/* The maximum number of steps in a step search given the largest allowed
 * initial step
 */
#define MAX_MVSEARCH_STEPS_SUBPEL 8

/* Max full pel mv specified in 1 pel units */
#define MAX_FULL_PEL_VAL_SUBPEL ((1 << (MAX_MVSEARCH_STEPS_SUBPEL)) - 1)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

#define ACCUMULATE(block_size_in_pixels,inter)      \
  if (block_size_in_pixels >= 32) {                 \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if (local_col < 2)                              \
    inter[(local_row * local_stride) + local_col] += inter[(local_row * local_stride) + local_col + 2]; \
  }                                                 \
  if (block_size_in_pixels >= 16) {                 \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if (local_col < 1)                              \
    inter[(local_row * local_stride)] += inter[(local_row * local_stride) + 1];  \
  }                                                 \
  if (block_size_in_pixels >= 32) {                 \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if (local_row < 16)                             \
    inter[(local_row * local_stride)] += inter[(local_row + 16) * local_stride]; \
  }                                                 \
  if (block_size_in_pixels >= 16) {                 \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if (local_row < 8)                              \
    inter[(local_row * local_stride)] += inter[(local_row + 8) * local_stride];  \
  }                                                 \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if (local_row < 4)                                \
  inter[(local_row * local_stride)] += inter[(local_row + 4) * local_stride]; \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if (local_row < 2)                                \
  inter[(local_row * local_stride)] += inter[(local_row + 2) * local_stride]; \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if (local_row < 1)                                \
  inter[(local_row * local_stride)] += inter[(local_row + 1) * local_stride];

#define CHECK_BETTER                                         \
  {                                                          \
    if (thissad < bestsad) {                                 \
     thissad += mvsad_err_cost(&this_mv,&fcenter_mv,         \
                    nmvsadcost_0,nmvsadcost_1,sad_per_bit);  \
      if (thissad < bestsad) {                               \
        bestsad = thissad;                                   \
        best_site = i;                                       \
      }                                                      \
    }                                                        \
  }

#define CHECK_BETTER_CENTER                                  \
  {                                                          \
    if (thissad < bestsad) {                                 \
        bestsad = thissad;                                   \
        best_mv = this_mv;                                   \
    }                                                        \
  }

///* estimated cost of a motion vector (r,c) */
#define MVC(v, r, c)                                         \
     v =((nmvjointcost[((r) != rr) * 2 + ((c) != rc)] +      \
             nmvcost_0[((r) - rr)] + nmvcost_1[((c) - rc)]) *\
             error_per_bit + 4096) >> 13;

#define CHECK_BETTER_SUBPEL(v, r, c)                             \
{                                                                \
  calculate_subpel_variance(ref_frame, cur_frame, stride,        \
       sp(c), sp(r), r, c, &sse, &sum, fdata3);                  \
                                                                 \
  barrier(CLK_LOCAL_MEM_FENCE);                                  \
  intermediate_sum[local_row * local_stride + local_col] = sum;  \
                                                                 \
  ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sum);             \
  barrier(CLK_LOCAL_MEM_FENCE);                                  \
  sum = intermediate_sum[0];                                     \
                                                                 \
  barrier(CLK_LOCAL_MEM_FENCE);                                  \
  intermediate_sse[local_row * local_stride + local_col] = sse;  \
                                                                 \
  ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sse);             \
  barrier(CLK_LOCAL_MEM_FENCE);                                  \
  sse = intermediate_sse[0];                                     \
                                                                 \
  thismse = sse - (((long int)sum * sum)                         \
            / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));    \
                                                                 \
  MVC(v, r, c);                                                  \
  if (c >= minc && c <= maxc && r >= minr && r <= maxr) {        \
    if ((v += thismse) < besterr) {                              \
      besterr = v;                                               \
      br = r;                                                    \
      bc = c;                                                    \
      distortion = thismse;                                      \
    }                                                            \
  }                                                              \
}

// The VP9_BILINEAR_FILTERS_2TAP macro returns a pointer to the bilinear
// filter kernel as a 2 tap filter.
#define BILINEAR_FILTERS_2TAP(x) \
  (vp9_bilinear_filters[(x)])

#define SETUP_SUBPEL_CHECKS                                            \
{                                                                      \
    hstep = 4;                                                         \
    besterr = CL_INT_MAX;                                              \
    br = best_mv.row;                                                  \
    bc = best_mv.col;                                                  \
                                                                       \
    rr = nearest_mv.row;                                               \
    rc = nearest_mv.col;                                               \
                                                                       \
    tr = br;                                                           \
    tc = bc;                                                           \
                                                                       \
    calculate_fullpel_variance(ref_frame, cur_frame, stride,           \
                      &sse, &sum, &best_mv);                           \
                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                      \
    intermediate_sum[local_row * local_stride + local_col] = sum;      \
    ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sum);                 \
    barrier(CLK_LOCAL_MEM_FENCE);                                      \
    sum = intermediate_sum[0];                                         \
                                                                       \
    barrier(CLK_LOCAL_MEM_FENCE);                                      \
    intermediate_sse[local_row * local_stride + local_col] = sse;      \
    ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sse);                 \
    barrier(CLK_LOCAL_MEM_FENCE);                                      \
    sse = intermediate_sse[0];                                         \
                                                                       \
    besterr = sse - (((long int)sum * sum)                             \
                      / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));\
                                                                       \
    besterr += mv_err_cost(&best_mv, &nearest_mv,                      \
                           nmvcost_0, nmvcost_1, error_per_bit);       \
                                                                       \
    minc = MAX(x->mv_col_min * 8, fcenter_mv.col - MV_MAX);            \
    maxc = MIN(x->mv_col_max * 8, fcenter_mv.col + MV_MAX);            \
    minr = MAX(x->mv_row_min * 8, fcenter_mv.row - MV_MAX);            \
    maxr = MIN(x->mv_row_max * 8, fcenter_mv.row + MV_MAX);            \
}

#define FIRST_LEVEL_CHECKS                                             \
{                                                                      \
    CHECK_BETTER_SUBPEL(left, tr, (tc - hstep));                       \
    CHECK_BETTER_SUBPEL(right, tr, (tc + hstep));                      \
    CHECK_BETTER_SUBPEL(up, (tr - hstep), tc);                         \
    CHECK_BETTER_SUBPEL(down, (tr + hstep), tc);                       \
                                                                       \
    whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);           \
                                                                       \
    switch (whichdir) {                                                \
      case 0:                                                          \
        CHECK_BETTER_SUBPEL(diag, (tr - hstep), (tc - hstep));         \
        break;                                                         \
      case 1:                                                          \
        CHECK_BETTER_SUBPEL(diag, (tr - hstep), (tc + hstep));         \
        break;                                                         \
      case 2:                                                          \
        CHECK_BETTER_SUBPEL(diag, (tr + hstep), (tc - hstep));         \
        break;                                                         \
      case 3:                                                          \
        CHECK_BETTER_SUBPEL(diag, (tr + hstep), (tc + hstep));         \
        break;                                                         \
    }                                                                  \
}

typedef unsigned short uint16_t;

typedef struct mv {
  short row;
  short col;
} MV;

typedef struct initvalues INIT;

struct initvalues {
    int mv_row_min;
    int mv_row_max;
    int mv_col_min;
    int mv_col_max;
};

typedef struct GPU_INPUT_STAGE1 {
  MV nearest_mv;
  MV near_mv;
} GPU_INPUT_STAGE1;

typedef struct GPU_OUTPUT_STAGE1 {
  MV mv;
  int rate_mv;
} GPU_OUTPUT_STAGE1;

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
  int nmvsadcost[2][MV_VALS];
  int mvcost[2][MV_VALS];
  int sad_per_bit;
  int error_per_bit;
} GPU_RD_PARAMETERS;

__constant int nmvjointsadcost[MV_JOINTS] = {600,300,300,300};

__constant int nmvjointcost[MV_JOINTS] = {767,561,518,331};

__constant int hex_num_candidates[MAX_PATTERN_SCALES] = {8, 6};

__constant MV hex_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
    {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, { 0, 1}, { -1, 1}, {-1, 0}},
    {{-1, -2}, {1, -2}, {2, 0}, {1, 2}, { -1, 2}, { -2, 0}},
  };

__constant int8 vp9_bilinear_filters[SUBPEL_SHIFTS] = {
  { 0, 0, 0, 128,   0, 0, 0, 0 },
  { 0, 0, 0, 120,   8, 0, 0, 0 },
  { 0, 0, 0, 112,  16, 0, 0, 0 },
  { 0, 0, 0, 104,  24, 0, 0, 0 },
  { 0, 0, 0,  96,  32, 0, 0, 0 },
  { 0, 0, 0,  88,  40, 0, 0, 0 },
  { 0, 0, 0,  80,  48, 0, 0, 0 },
  { 0, 0, 0,  72,  56, 0, 0, 0 },
  { 0, 0, 0,  64,  64, 0, 0, 0 },
  { 0, 0, 0,  56,  72, 0, 0, 0 },
  { 0, 0, 0,  48,  80, 0, 0, 0 },
  { 0, 0, 0,  40,  88, 0, 0, 0 },
  { 0, 0, 0,  32,  96, 0, 0, 0 },
  { 0, 0, 0,  24, 104, 0, 0, 0 },
  { 0, 0, 0,  16, 112, 0, 0, 0 },
  { 0, 0, 0,   8, 120, 0, 0, 0 }
};


__constant int num_8x8_blocks_wide_lookup[3] = {1, 2, 4};
__constant int num_8x8_blocks_high_lookup[3] = {1, 2, 4};

// convert motion vector component to offset for svf calc
inline unsigned int sp(unsigned int x) {
  return (x & 7) << 1;
}

inline MV_JOINT_TYPE vp9_get_mv_joint(const MV *mv) {
  if (mv->row == 0) {
    return mv->col == 0 ? MV_JOINT_ZERO : MV_JOINT_HNZVZ;
  } else {
    return mv->col == 0 ? MV_JOINT_HZVNZ : MV_JOINT_HNZVNZ;
  }
}

inline int mv_cost(MV *mv,
                   __constant int *joint_cost, __global int *comp_cost_0,
                   __global int *comp_cost_1) {
  return joint_cost[vp9_get_mv_joint(mv)] +
                    comp_cost_0[mv->row] + comp_cost_1[mv->col];
}

short mvsad_err_cost(MV *mv,
                     MV *ref,
                     __global int *nmvsadcost_0,
                     __global int *nmvsadcost_1,
                     int sad_perbit){
  MV diff;
  int joint_cost, mv_cost;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  joint_cost = vp9_get_mv_joint(&diff);

  mv_cost = nmvjointsadcost[joint_cost] + nmvsadcost_0[diff.row]
                                                     + nmvsadcost_1[diff.col];

  mv_cost *= sad_perbit;

  mv_cost = ROUND_POWER_OF_TWO(mv_cost,8);

  return mv_cost;
}

short mv_err_cost(MV *mv,
                  MV *ref,
                  __global int *nmvcost_0,
                  __global int *nmvcost_1,
                  int error_per_bit){
  MV diff;
  int joint_cost, mv_cost;
  int mvc_0,mvc_1;

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);

  mvc_0 = 0;
  mvc_1 = 0;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  joint_cost = vp9_get_mv_joint(&diff);

  mvc_0 += nmvcost_0[diff.row];
  mvc_1 += nmvcost_1[diff.col];

  mv_cost = nmvjointcost[joint_cost] + mvc_0 + mvc_1;

  mv_cost *= error_per_bit;

  mv_cost = ROUND_POWER_OF_TWO(mv_cost,13);

  return mv_cost;
}

int gpu_check_bounds(INIT *x,
                     int row,
                     int col,
                     int range) {
  return ((row - range) >= x->mv_row_min) &
         ((row + range) <= x->mv_row_max) &
         ((col - range) >= x->mv_col_min) &
         ((col + range) <= x->mv_col_max);
}

int is_mv_in(INIT *x,
             const MV *mv) {
  return (mv->col >= x->mv_col_min) && (mv->col <= x->mv_col_max) &&
         (mv->row >= x->mv_row_min) && (mv->row <= x->mv_row_max);
}

void vp9_gpu_set_mv_search_range(INIT *x,
                                 int mi_row,
                                 int mi_col,
                                 int mi_rows,
                                 int mi_cols,
                                 int bsize){

  int mi_width  = num_8x8_blocks_wide_lookup[bsize];
  int mi_height = num_8x8_blocks_high_lookup[bsize];

  //Set up limit values for MV components.
  //Mv beyond the range do not produce new/different prediction block.
  x->mv_row_min = -(((mi_row + mi_height) * MI_SIZE) + VP9_INTERP_EXTEND);
  x->mv_col_min = -(((mi_col + mi_width) * MI_SIZE) + VP9_INTERP_EXTEND);
  x->mv_row_max = (mi_rows - mi_row) * MI_SIZE + VP9_INTERP_EXTEND;
  x->mv_col_max = (mi_cols - mi_col) * MI_SIZE + VP9_INTERP_EXTEND;

}

int vp9_mv_bit_cost(MV *mv, MV *ref,
                    __constant int *mvjcost,
                    __global int *mvcost_0,
                    __global int *mvcost_1,
                    int weight) {
  MV diff = { mv->row - ref->row,
              mv->col - ref->col };
  return ROUND_POWER_OF_TWO(mv_cost(&diff, mvjcost, mvcost_0, mvcost_1)
                            * weight, 7);
}

void vp9_set_mv_search_range_step2(INIT *x, const MV *mv) {
  int col_min = (mv->col >> 3) - MAX_FULL_PEL_VAL + (mv->col & 7 ? 1 : 0);
  int row_min = (mv->row >> 3) - MAX_FULL_PEL_VAL + (mv->row & 7 ? 1 : 0);
  int col_max = (mv->col >> 3) + MAX_FULL_PEL_VAL;
  int row_max = (mv->row >> 3) + MAX_FULL_PEL_VAL;

  col_min = MAX(col_min, (MV_LOW >> 3) + 1);
  row_min = MAX(row_min, (MV_LOW >> 3) + 1);
  col_max = MIN(col_max, (MV_UPP >> 3) - 1);
  row_max = MIN(row_max, (MV_UPP >> 3) - 1);

  // Get intersection of UMV window and valid MV window to reduce # of checks
  // in diamond search.
  if (x->mv_col_min < col_min)
    x->mv_col_min = col_min;
  if (x->mv_col_max > col_max)
    x->mv_col_max = col_max;
  if (x->mv_row_min < row_min)
    x->mv_row_min = row_min;
  if (x->mv_row_max > row_max)
    x->mv_row_max = row_max;
}

int clamp_it(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

void clamp_gpu_mv(MV *mv, int min_col, int max_col,
                            int min_row, int max_row) {
  mv->col = clamp_it(mv->col, min_col, max_col);
  mv->row = clamp_it(mv->row, min_row, max_row);
}

short calculate_sad(MV *currentmv,
                    __global uchar *ref_frame,
                    __global uchar *cur_frame,
                    int stride)
{
  uchar8 ref,cur;
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;

  int sad;

  buffer_offset = (currentmv->row * stride) + currentmv->col;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  ref = vload8(0,tmp_ref);
  cur = vload8(0,tmp_cur);

  sad  = abs(ref.s0 - cur.s0);
  sad += abs(ref.s1 - cur.s1);
  sad += abs(ref.s2 - cur.s2);
  sad += abs(ref.s3 - cur.s3);
  sad += abs(ref.s4 - cur.s4);
  sad += abs(ref.s5 - cur.s5);
  sad += abs(ref.s6 - cur.s6);
  sad += abs(ref.s7 - cur.s7);

  return sad;
}

void get_sum_sse(uchar8 ref, uchar8 cur, unsigned int *psse, int *psum)
{
  short8 diff = convert_short8(ref) - convert_short8(cur);
  short sum;
  sum  = diff.s0;
  sum += diff.s1;
  sum += diff.s2;
  sum += diff.s3;
  sum += diff.s4;
  sum += diff.s5;
  sum += diff.s6;
  sum += diff.s7;
  *psum = sum;

  uint8 diff_squared = convert_uint8(convert_int8(diff) * convert_int8(diff));
  uint sse;
  sse  = diff_squared.s0;
  sse += diff_squared.s1;
  sse += diff_squared.s2;
  sse += diff_squared.s3;
  sse += diff_squared.s4;
  sse += diff_squared.s5;
  sse += diff_squared.s6;
  sse += diff_squared.s7;
  *psse = sse;

  return;
}

void calculate_fullpel_variance(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                int stride,
                                unsigned int *sse,
                                int *sum,
                                MV *submv) {
  uchar8 ref,cur;
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;

  int sad,diff;
  buffer_offset = ((submv->row >> 3) * stride) + (submv->col >> 3);
  *sum = 0;
  *sse = 0;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  ref = vload8(0,tmp_ref);
  cur = vload8(0,tmp_cur);

  get_sum_sse(ref, cur, sse, sum);
}

void var_filter_block2d_bil_first_pass_gpu(__global uchar *ref_data,
                                           __local uint16_t *fdata3,
                                           int stride,
                                           int8 vp9_filter) {
   __local ushort8 *output;
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
  int local_height = get_local_size(1);
  uchar8 src_0;
  uchar8 src_1;

  output  = (__local ushort8 *)(fdata3 + local_row * BLOCK_SIZE_IN_PIXELS +
                                         local_col * NUM_PIXELS_PER_WORKITEM);
  src_0 = vload8(0, ref_data);
  src_1 = vload8(0, ref_data + 1);
  barrier(CLK_LOCAL_MEM_FENCE);
  *output =
      convert_ushort8(ROUND_POWER_OF_TWO(convert_int8(src_0) * vp9_filter.s3 +
                         convert_int8(src_1) * vp9_filter.s4,  FILTER_BITS));

  if (local_row == (local_height - 1)) {

    output   = (__local ushort8 *)(fdata3 + (local_row + 1) *
                BLOCK_SIZE_IN_PIXELS + local_col * NUM_PIXELS_PER_WORKITEM);
    ref_data += stride;
    src_0 = vload8(0, ref_data);
    src_1 = vload8(0, ref_data + 1);
    *output =
        convert_ushort8(ROUND_POWER_OF_TWO(convert_int8(src_0) * vp9_filter.s3 +
                            convert_int8(src_1) * vp9_filter.s4,  FILTER_BITS));
  }
}

uchar8 var_filter_block2d_bil_second_pass_gpu(__local uint16_t *fdata3,
                                             int8 vp9_filter) {

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int local_stride = get_local_size(0);
  uchar8 output;
  __local ushort8 *ref_data;
  ushort8 src_0;
  ushort8 src_1;

  ref_data  = (__local ushort8 *)(fdata3 + local_row * BLOCK_SIZE_IN_PIXELS
                                      + local_col * NUM_PIXELS_PER_WORKITEM);
  barrier(CLK_LOCAL_MEM_FENCE);
  src_0 = ref_data[0];
  src_1 = ref_data[BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM];
  output =
      convert_uchar8(ROUND_POWER_OF_TWO(convert_int8(src_0) * vp9_filter.s3 +
                         convert_int8(src_1) * vp9_filter.s4,  FILTER_BITS));

  return output;
}

void calculate_subpel_variance(__global uchar *ref_frame,
                               __global uchar *cur_frame,
                               int stride,
                               int xoffset,
                               int yoffset,
                               int row,
                               int col,
                               unsigned int *sse,
                               int *sum,
                               __local uint16_t *fdata3) {
  uchar8 ref,cur,temp2;
  int buffer_offset, diff;
  __global uchar *tmp_ref,*tmp_cur;

  int i,j,h;
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  int local_stride = get_local_size(0);

  buffer_offset = ((row >> 3) * stride) + (col >> 3);
  *sum = 0;
  *sse = 0;
  diff = 0;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  var_filter_block2d_bil_first_pass_gpu(tmp_ref, fdata3, stride,
                                        BILINEAR_FILTERS_2TAP(xoffset));
  temp2 = var_filter_block2d_bil_second_pass_gpu(fdata3,
                                        BILINEAR_FILTERS_2TAP(yoffset));

  cur   = vload8(0,tmp_cur);

  get_sum_sse(temp2, cur, sse, sum);
}

inline int get_sad(__global uchar *ref_frame, __global uchar *cur_frame,
                   int stride, __local int* intermediate_sad, MV this_mv)
{
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int local_stride = get_local_size(0);

  barrier(CLK_LOCAL_MEM_FENCE);
  intermediate_sad[local_row * local_stride + local_col] =
      calculate_sad(&this_mv, ref_frame, cur_frame, stride);

  ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sad)

  barrier(CLK_LOCAL_MEM_FENCE);
  return intermediate_sad[0];
}

inline MV get_best_mv(__global uchar *ref_frame, __global uchar *cur_frame,
                      int stride, __local int* intermediate_sad,
                      MV nearest_mv, MV near_mv, MV pred_mv, int *pbestsad)
{
  MV this_mv, best_mv;
  int thissad, bestsad = CL_INT_MAX;

  this_mv.row = nearest_mv.row >> 3;
  this_mv.col = nearest_mv.col >> 3;
  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
  CHECK_BETTER_CENTER

  this_mv.row = near_mv.row >> 3;
  this_mv.col = near_mv.col >> 3;
  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
  CHECK_BETTER_CENTER

#if BLOCK_SIZE_IN_PIXELS != 32
  this_mv.row = pred_mv.row >> 3;
  this_mv.col = pred_mv.col >> 3;
  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
  CHECK_BETTER_CENTER
#endif

  *pbestsad = bestsad;
  return best_mv;
}

inline MV full_pixel_search(__global uchar *ref_frame,
                            __global uchar *cur_frame, int stride,
                            __local int* intermediate_sad,
                            MV best_mv, int bestsad, MV fcenter_mv,
                            __global int *nmvsadcost_0,
                            __global int *nmvsadcost_1,
                            INIT *x, int sad_per_bit)
{
  MV this_mv;
  int best_site = -1;
  int i, k;
  int br, bc;
  int pattern = 1;
  int next_chkpts_indices[PATTERN_CANDIDATES_REF];
  int thissad;

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int local_stride = get_local_size(0);

  clamp_gpu_mv(&best_mv, x->mv_col_min, x->mv_col_max,
                            x->mv_row_min , x->mv_row_max);
  br = best_mv.row;
  bc = best_mv.col;

  do {
    best_site = -1;
    if (gpu_check_bounds(x,br,bc,1 << pattern)) {
      for (i = 0; i < hex_num_candidates[pattern]; i++) {
        this_mv.row = br + hex_candidates[pattern][i].row;
        this_mv.col = bc + hex_candidates[pattern][i].col;

        barrier(CLK_LOCAL_MEM_FENCE);
        intermediate_sad[local_row * local_stride + local_col] =
                         calculate_sad(&this_mv,ref_frame,cur_frame,stride);

        ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sad);

        barrier(CLK_LOCAL_MEM_FENCE);
        thissad = intermediate_sad[0];

        CHECK_BETTER
      }
    } else {
      for (i = 0; i < hex_num_candidates[pattern]; i++) {
        this_mv.row = br + hex_candidates[pattern][i].row;
        this_mv.col = bc + hex_candidates[pattern][i].col;

        if (!is_mv_in(x,&this_mv)) {
          continue;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        intermediate_sad[local_row * local_stride + local_col] =
                         calculate_sad(&this_mv,ref_frame,cur_frame,stride);

        ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sad);

        barrier(CLK_LOCAL_MEM_FENCE);
        thissad = intermediate_sad[0];

        CHECK_BETTER
      }
    }

    if (best_site == -1) {
      continue;
    } else {
      br += hex_candidates[pattern][best_site].row;
      bc += hex_candidates[pattern][best_site].col;
      k = best_site;
    }

    do {
      best_site = -1;
      next_chkpts_indices[0] =
             (k == 0) ? hex_num_candidates[pattern] - 1 : k - 1;
      next_chkpts_indices[1] = k;
      next_chkpts_indices[2] =
             (k == hex_num_candidates[pattern] - 1) ? 0 : k + 1;

      if (gpu_check_bounds(x,br,bc,1 << pattern)) {
        for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
          this_mv.row = br + hex_candidates[pattern][next_chkpts_indices[i]].row;
          this_mv.col = bc + hex_candidates[pattern][next_chkpts_indices[i]].col;

          barrier(CLK_LOCAL_MEM_FENCE);
          intermediate_sad[local_row * local_stride + local_col] =
                            calculate_sad(&this_mv,ref_frame,cur_frame,stride);

          ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sad)
          barrier(CLK_LOCAL_MEM_FENCE);
          thissad = intermediate_sad[0];
          CHECK_BETTER
        }
      } else {
        for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
          this_mv.row = br + hex_candidates[pattern][next_chkpts_indices[i]].row;
          this_mv.col = bc + hex_candidates[pattern][next_chkpts_indices[i]].col;

          if (!is_mv_in(x,&this_mv)) {
            continue;
          }

          barrier(CLK_LOCAL_MEM_FENCE);
          intermediate_sad[local_row * local_stride + local_col] =
                          calculate_sad(&this_mv,ref_frame,cur_frame,stride);

          ACCUMULATE(BLOCK_SIZE_IN_PIXELS,intermediate_sad)
          barrier(CLK_LOCAL_MEM_FENCE);
          thissad = intermediate_sad[0];
          CHECK_BETTER
        }
      }

      if (best_site != -1) {
        k = next_chkpts_indices[best_site];
        br += hex_candidates[pattern][k].row;
        bc += hex_candidates[pattern][k].col;
      }
    } while(best_site != -1);
  } while(pattern--);
  best_mv.row = br * 8;
  best_mv.col = bc * 8;
  return best_mv;
}

inline MV vp9_find_best_sub_pixel_tree(__global uchar *ref_frame,
                                       __global uchar *cur_frame,
                                       int stride,
                                       __local ushort *fdata3,
                                       MV best_mv,
                                       MV nearest_mv,
                                       MV fcenter_mv,
                                       __global int *nmvcost_0,
                                       __global int *nmvcost_1,
                                       INIT *x,
                                       int error_per_bit) {
  __local int *intermediate_sum = (__local int *)fdata3;
  __local uint *intermediate_sse = (__local uint *)fdata3;
                                       
  int br,bc;
  int sum, besterr, thismse, tr, tc;
  int distortion;
  int minc, maxc, minr, maxr, hstep, rr,rc;
  unsigned int left, right, up, down, diag, whichdir;
  unsigned int sse;

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  int local_stride = get_local_size(0);
  int local_height = get_local_size(1);

  SETUP_SUBPEL_CHECKS

  FIRST_LEVEL_CHECKS

  tr = br;
  tc = bc;

  hstep >>= 1;
  FIRST_LEVEL_CHECKS;

  best_mv.row = br;
  best_mv.col = bc;

  if ((abs(best_mv.col - nearest_mv.col) > (MAX_FULL_PEL_VAL_SUBPEL << 3)) ||
      (abs(best_mv.row - nearest_mv.row) > (MAX_FULL_PEL_VAL_SUBPEL << 3)))
    besterr = CL_INT_MAX;

  return best_mv;
}

__kernel
void fullpel_search(__global uchar *ref_frame,
                    __global uchar *cur_frame,
                    int stride,
                    __global GPU_OUTPUT_STAGE1 *out_frame,
                    __global GPU_INPUT_STAGE1 *meta_data,
                    __global GPU_OUTPUT_STAGE1 *newmv,
                    __global GPU_RD_PARAMETERS *rd_parameters,
                    int mi_rows,
                    int mi_cols
                    ) {
  __local ushort intermediate_ushort[(BLOCK_SIZE_IN_PIXELS + 1) * BLOCK_SIZE_IN_PIXELS];
  __local int    *intermediate_int = (__local int*)  intermediate_ushort;
  __local uint   *intermediate_sse = (__local uint*) intermediate_ushort;
  __global int *nmvsadcost_0 = rd_parameters->nmvsadcost[0] + MV_MAX;
  __global int *nmvsadcost_1 = rd_parameters->nmvsadcost[1] + MV_MAX;
  __global int *nmvcost_0 = rd_parameters->mvcost[0] + MV_MAX;
  __global int *nmvcost_1 = rd_parameters->mvcost[1] + MV_MAX;

  MV nearest_mv, near_mv, pred_mv;
  MV this_mv, best_mv;
  MV fcenter_mv;
  int bestsad;
  int i,mi_row,mi_col, sad_per_bit, error_per_bit;
  int index_mv0,index_mv1,mvc_0,mvc_1;
  INIT x, tmp_x;
  int padding_offset;
  int global_offset;

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);

  int group_stride = get_num_groups(0);
  int group_height = get_num_groups(1);

  int local_stride = get_local_size(0);
  int local_height = get_local_size(1);

  sad_per_bit     = rd_parameters->sad_per_bit;
  error_per_bit     = rd_parameters->error_per_bit;

  mi_row = group_row * local_stride;
  mi_col = group_col * local_stride;

  vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows,
                              mi_cols, (BLOCK_SIZE_IN_PIXELS >> 4));

  meta_data += group_col + group_row * group_stride;
  out_frame += group_col + group_row * group_stride;

#if BLOCK_SIZE_IN_PIXELS != 32
  newmv  += group_col/2 + group_row/2 * group_stride/2;
#endif

  padding_offset = (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;
  global_offset = global_col * NUM_PIXELS_PER_WORKITEM + global_row * stride;
  ref_frame += global_offset + padding_offset;
  cur_frame += global_offset + padding_offset;

  nearest_mv = meta_data->nearest_mv;
  tmp_x = x;
  vp9_set_mv_search_range_step2(&x, &nearest_mv);

  near_mv = meta_data->near_mv;

#if BLOCK_SIZE_IN_PIXELS != 32
  pred_mv = newmv->mv;
#endif

  fcenter_mv.row = nearest_mv.row >> 3;
  fcenter_mv.col = nearest_mv.col >> 3;

  best_mv = get_best_mv(ref_frame, cur_frame, stride, intermediate_int,
                        nearest_mv, near_mv, pred_mv, &bestsad);
  bestsad += mvsad_err_cost(&best_mv, &fcenter_mv, nmvsadcost_0, nmvsadcost_1,
                            sad_per_bit);

  best_mv = full_pixel_search(ref_frame, cur_frame, stride,
                              intermediate_int, best_mv, bestsad, fcenter_mv,
                              nmvsadcost_0, nmvsadcost_1,
                              &x, sad_per_bit);


  out_frame->rate_mv = vp9_mv_bit_cost(&best_mv, &nearest_mv,
                                       nmvjointcost, nmvcost_0, nmvcost_1, 
                                       MV_COST_WEIGHT);

  x = tmp_x;

  out_frame->mv = vp9_find_best_sub_pixel_tree(ref_frame, cur_frame, stride,
                                          intermediate_ushort, best_mv, nearest_mv,
                                          fcenter_mv, nmvcost_0, nmvcost_1,
                                          &x, error_per_bit);

  return;
}

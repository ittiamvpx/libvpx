/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

typedef enum BLOCK_SIZE {
  BLOCK_4X4,
  BLOCK_4X8,
  BLOCK_8X4,
  BLOCK_8X8,
  BLOCK_8X16,
  BLOCK_16X8,
  BLOCK_16X16,
  BLOCK_16X32,
  BLOCK_32X16,
  BLOCK_32X32,
  BLOCK_32X64,
  BLOCK_64X32,
  BLOCK_64X64,
  BLOCK_SIZES,
  BLOCK_INVALID = BLOCK_SIZES
} BLOCK_SIZE;

// This enumerator type needs to be kept aligned with the mode order in
// const MODE_DEFINITION vp9_mode_order[MAX_MODES] used in the rd code.
typedef enum {
  THR_NEARESTMV,
  THR_NEARESTA,
  THR_NEARESTG,

  THR_DC,

  THR_NEWMV,
  THR_NEWA,
  THR_NEWG,

  THR_NEARMV,
  THR_NEARA,
  THR_COMP_NEARESTLA,
  THR_COMP_NEARESTGA,

  THR_TM,

  THR_COMP_NEARLA,
  THR_COMP_NEWLA,
  THR_NEARG,
  THR_COMP_NEARGA,
  THR_COMP_NEWGA,

  THR_ZEROMV,
  THR_ZEROG,
  THR_ZEROA,
  THR_COMP_ZEROLA,
  THR_COMP_ZEROGA,

  THR_H_PRED,
  THR_V_PRED,
  THR_D135_PRED,
  THR_D207_PRED,
  THR_D153_PRED,
  THR_D63_PRED,
  THR_D117_PRED,
  THR_D45_PRED,
} THR_MODES;

typedef enum {
  DC_PRED,         // Average of above and left pixels
  V_PRED,          // Vertical
  H_PRED,          // Horizontal
  D45_PRED,        // Directional 45  deg = round(arctan(1/1) * 180/pi)
  D135_PRED,       // Directional 135 deg = 180 - 45
  D117_PRED,       // Directional 117 deg = 180 - 63
  D153_PRED,       // Directional 153 deg = 180 - 27
  D207_PRED,       // Directional 207 deg = 180 + 27
  D63_PRED,        // Directional 63  deg = round(arctan(2/1) * 180/pi)
  TM_PRED,         // True-motion
  NEARESTMV,
  NEARMV,
  ZEROMV,
  NEWMV,
  MB_MODE_COUNT
} PREDICTION_MODE;

typedef enum {
  NONE = -1,
  INTRA_FRAME = 0,
  LAST_FRAME = 1,
  GOLDEN_FRAME = 2,
  ALTREF_FRAME = 3,
  MAX_REF_FRAMES = 4
} MV_REFERENCE_FRAME;

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

typedef enum {
  TX_4X4 = 0,                      // 4x4 transform
  TX_8X8 = 1,                      // 8x8 transform
  TX_16X16 = 2,                    // 16x16 transform
  TX_32X32 = 3,                    // 32x32 transform
  TX_SIZES
} TX_SIZE;

typedef enum {
  EIGHTTAP = 0,
  EIGHTTAP_SMOOTH = 1,
  EIGHTTAP_SHARP = 2,
  BILINEAR = 3,
  SWITCHABLE = 4  /* should be the last one */
} INTERP_FILTER;

typedef enum GPU_BLOCK_SIZE {
  GPU_BLOCK_32X32,
  GPU_BLOCK_16X16,
  GPU_BLOCK_SIZES,
  GPU_BLOCK_INVALID = GPU_BLOCK_SIZES
} GPU_BLOCK_SIZE;

typedef enum {
  DIAMOND = 0,
  NSTEP = 1,
  HEX = 2,
  BIGDIA = 3,
  SQUARE = 4,
  FAST_HEX = 5,
  FAST_DIAMOND = 6
} SEARCH_METHODS;

#define NUM_PIXELS_PER_WORKITEM 8
#define VP9_ENC_BORDER_IN_PIXELS    160
#define SUBPEL_BITS 3
#define SUBPEL_MASK ((1 << SUBPEL_BITS) - 1)
#define INT32_MAX 2147483647
#define INT64_MAX 9223372036854775807LL
#define INTER_MODE_CONTEXTS 7
#define GPU_INTER_MODES 2 // ZEROMV and NEWMV
#define SWITCHABLE_FILTERS 3   // number of switchable filters
#define MAX_MODES 30
#define MV_CLASSES     11
#define CLASS0_BITS    1  /* bits at integer precision for class 0 */
#define MV_MAX_BITS    (MV_CLASSES + CLASS0_BITS + 2)
#define MV_MAX         ((1 << MV_MAX_BITS) - 1)
#define MV_VALS        ((MV_MAX << 1) + 1)

#define INTER_MODES (1 + NEWMV - NEARESTMV)
#define INTER_OFFSET(mode) ((mode) - NEARESTMV)
#define GPU_INTER_OFFSET(mode) ((mode) - ZEROMV)
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#define CL_INT_MAX          2147483647
#define VP9_ENC_BORDER_IN_PIXELS    160
#define MAX_PATTERN_SCALES 2
#define NUM_PIXELS_PER_WORKITEM 8
#define PATTERN_CANDIDATES_REF 3
#define MAX_PATTERN_CANDIDATES 8
#define MV_JOINTS 4
#define MI_SIZE 8
#define VP9_INTERP_EXTEND 4
#define FILTER_BITS 7
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
#define LOCAL_STRIDE (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM)
#define LOCAL_HEIGHT  BLOCK_SIZE_IN_PIXELS

#define GPU_SEARCH_METHODS 2
#define GPU_SEARCH_OFFSET(search_method) ((search_method) - FAST_HEX)


/* The maximum number of steps in a step search given the largest allowed
 * initial step
 */
#define MAX_MVSEARCH_STEPS_SUBPEL 8

/* Max full pel mv specified in 1 pel units */
#define MAX_FULL_PEL_VAL_SUBPEL ((1 << (MAX_MVSEARCH_STEPS_SUBPEL)) - 1)

#define MIN(x, y) (((x) < (y)) ? (x) : (y))
#define MAX(x, y) (((x) > (y)) ? (x) : (y))

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

#define CLAMP_IT(value, low, high)  \
    (value < low ? low : (value > high ? high : value))

#define CHECK_BETTER_SUBPEL(r, c, idx)                                \
      sum = intermediate_sum_sse[idx];                                \
      sse = intermediate_sum_sse[idx + 1];                            \
                                                                      \
      thiserr  = (sse - (((long int)sum * sum)                        \
              / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS)));      \
                                                                      \
      if (thiserr < besterr) {                                        \
        besterr = thiserr;                                            \
        best_mv.row = r;                                              \
        best_mv.col = c;                                              \
      }


// The VP9_BILINEAR_FILTERS_2TAP macro returns a pointer to the bilinear
// filter kernel as a 2 tap filter.
#define BILINEAR_FILTERS_2TAP(x) \
  (vp9_bilinear_filters[(x)])

#define ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)                         \
    sum.s0123 = sum.s0123 + sum.s4567;                                  \
    sum.s01   = sum.s01   + sum.s23;                                    \
    sum.s0    = sum.s0    + sum.s1;                                     \
    atomic_add(psum, sum.s0);                                           \
    sse.s01   = sse.s01   + sse.s23;                                    \
    sse.s0    = sse.s0    + sse.s1;                                     \
    atomic_add(psse, sse.s0);

#define CALCULATE_RATE_DIST                                                 \
  if (tx_mode == TX_MODE_SELECT) {                                          \
    if (sse > (variance << 2))                                              \
      tx_size =    MIN(max_txsize_lookup[bsize],                            \
              tx_mode_to_biggest_tx_size[tx_mode]);                         \
    else                                                                    \
      tx_size = TX_8X8;                                                     \
  } else {                                                                  \
    tx_size =    MIN(max_txsize_lookup[bsize],                              \
            tx_mode_to_biggest_tx_size[tx_mode]);                           \
  }                                                                         \
  if (sse < dc_quant * dc_quant >> 6)                                       \
    skip_txfm = 1;                                                          \
  else if (variance < ac_quant * ac_quant >> 6)                             \
    skip_txfm = 2;                                                          \
  else                                                                      \
    skip_txfm = 0;                                                          \
  vp9_model_rd_from_var_lapndz(sse - variance, 1 << num_pels_log2_lookup[bsize], \
                               dc_quant >> 3, &rate, &dist);                \
  actual_rate = rate >> 1;                                                  \
  actual_dist = dist << 3;                                                  \
  vp9_model_rd_from_var_lapndz(variance, 1 << num_pels_log2_lookup[bsize],  \
                               ac_quant >> 3, &rate, &dist);                \
  actual_rate += rate;                                                      \
  actual_dist += dist << 4;

#define RDCOST(RM, DM, R, D) (((128 + ((int64_t)R) * (RM)) >> 8) + (D << DM))


typedef short int16_t;
typedef long  int64_t;
typedef uint  uint32_t;
typedef ulong uint64_t;

typedef unsigned short uint16_t;

// Enabling this piece of code causes a crash in Intel HD graphics. But it works
// fine in Mali GPU. Must be an issue with Intel's driver
#if !INTEL_HD_GRAPHICS

typedef short2 MV;
#define row s0
#define col s1

typedef short4 INIT;
#define mv_row_min s0
#define mv_row_max s1
#define mv_col_min s2
#define mv_col_max s3

#else

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

#endif

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

typedef struct {
  int sum;
  unsigned int sse;
} SUM_SSE;

typedef struct {
  SUM_SSE sum_sse[5];
} subpel_sum_sse;

typedef struct {
  unsigned int sse[EIGHTTAP_SHARP + 1];
  int sum[EIGHTTAP_SHARP + 1];
} rd_calc_buffers;

__constant MV sub_pel_offset[4] = {{0, -1}, {0, 1}, {-1, 0}, {1, 0}};

__constant INTERP_FILTER interp_filter[4] =
  {EIGHTTAP, EIGHTTAP_SMOOTH, EIGHTTAP, EIGHTTAP_SHARP};

__constant int nmvjointsadcost[MV_JOINTS] = {600,300,300,300};

__constant int num_candidates_arr[GPU_SEARCH_METHODS][MAX_PATTERN_SCALES] = {
    {8, 6},
    {4}
 };

__constant MV candidates_arr[GPU_SEARCH_METHODS][MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
    {
        {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, { 0, 1}, { -1, 1}, {-1, 0}},
        {{-1, -2}, {1, -2}, {2, 0}, {1, 2}, { -1, 2}, { -2, 0}},
    },
    {
        {{0, -1}, {1, 0}, { 0, 1}, {-1, 0}}
    },
  };

__constant ushort2 vp9_bilinear_filters[16] = {
  {128,   0},
  {120,   8},
  {112,  16},
  {104,  24},
  { 96,  32},
  { 88,  40},
  { 80,  48},
  { 72,  56},
  { 64,  64},
  { 56,  72},
  { 48,  80},
  { 40,  88},
  { 32,  96},
  { 24, 104},
  { 16, 112},
  {  8, 120}
};


__constant int num_8x8_blocks_wide_lookup[3] = {1, 2, 4};
__constant int num_8x8_blocks_high_lookup[3] = {1, 2, 4};

__constant int num_pels_log2_lookup[BLOCK_SIZES] =
  {4, 5, 5, 6, 7, 7, 8, 9, 9, 10, 11, 11, 12};

__constant THR_MODES mode_idx[MAX_REF_FRAMES - 1][INTER_MODES] = {
  {THR_NEARESTMV, THR_NEARMV, THR_ZEROMV, THR_NEWMV},
  {THR_NEARESTG, THR_NEARG, THR_ZEROG, THR_NEWG},
  {THR_NEARESTA, THR_NEARA, THR_ZEROA, THR_NEWA},
};

__constant TX_SIZE max_txsize_lookup[BLOCK_SIZES] = {
  TX_4X4,   TX_4X4,   TX_4X4,
  TX_8X8,   TX_8X8,   TX_8X8,
  TX_16X16, TX_16X16, TX_16X16,
  TX_32X32, TX_32X32, TX_32X32, TX_32X32
};

__constant TX_SIZE tx_mode_to_biggest_tx_size[TX_MODES] = {
  TX_4X4,  // ONLY_4X4
  TX_8X8,  // ALLOW_8X8
  TX_16X16,  // ALLOW_16X16
  TX_32X32,  // ALLOW_32X32
  TX_32X32,  // TX_MODE_SELECT
};

__constant char8 filter[4][16] =  {

                { { 0,  0,   0,  64,   0,   0,   0,  0},
                  { 0,  1,  -5, 126,   8,  -3,   1,  0},
                  {-1,  3, -10, 122,  18,  -6,   2,  0},
                  {-1,  4, -13, 118,  27,  -9,   3, -1},
                  {-1,  4, -16, 112,  37, -11,   4, -1},
                  {-1,  5, -18, 105,  48, -14,   4, -1},
                  {-1,  5, -19,  97,  58, -16,   5, -1},
                  {-1,  6, -19,  88,  68, -18,   5, -1},
                  {-1,  6, -19,  78,  78, -19,   6, -1},
                  {-1,  5, -18,  68,  88, -19,   6, -1},
                  {-1,  5, -16,  58,  97, -19,   5, -1},
                  {-1,  4, -14,  48, 105, -18,   5, -1},
                  {-1,  4, -11,  37, 112, -16,   4, -1},
                  {-1,  3,  -9,  27, 118, -13,   4, -1},
                  { 0,  2,  -6,  18, 122, -10,   3, -1},
                  { 0,  1,  -3,   8, 126,  -5,   1,  0}},

                { { 0,  0,  0,  64,  0,  0,  0,  0},
                  {-3, -1, 32,  64, 38,  1, -3,  0},
                  {-2, -2, 29,  63, 41,  2, -3,  0},
                  {-2, -2, 26,  63, 43,  4, -4,  0},
                  {-2, -3, 24,  62, 46,  5, -4,  0},
                  {-2, -3, 21,  60, 49,  7, -4,  0},
                  {-1, -4, 18,  59, 51,  9, -4,  0},
                  {-1, -4, 16,  57, 53, 12, -4, -1},
                  {-1, -4, 14,  55, 55, 14, -4, -1},
                  {-1, -4, 12,  53, 57, 16, -4, -1},
                  { 0, -4,  9,  51, 59, 18, -4, -1},
                  { 0, -4,  7,  49, 60, 21, -3, -2},
                  { 0, -4,  5,  46, 62, 24, -3, -2},
                  { 0, -4,  4,  43, 63, 26, -2, -2},
                  { 0, -3,  2,  41, 63, 29, -2, -2},
                  { 0, -3,  1,  38, 64, 32, -1, -3}},

                { { 0,   0,   0,  64,   0,   0,   0,  0},
                  {-1,   3,  -7, 127,   8,  -3,   1,  0},
                  {-2,   5, -13, 125,  17,  -6,   3, -1},
                  {-3,   7, -17, 121,  27, -10,   5, -2},
                  {-4,   9, -20, 115,  37, -13,   6, -2},
                  {-4,  10, -23, 108,  48, -16,   8, -3},
                  {-4,  10, -24, 100,  59, -19,   9, -3},
                  {-4,  11, -24,  90,  70, -21,  10, -4},
                  {-4,  11, -23,  80,  80, -23,  11, -4},
                  {-4,  10, -21,  70,  90, -24,  11, -4},
                  {-3,   9, -19,  59, 100, -24,  10, -4},
                  {-3,   8, -16,  48, 108, -23,  10, -4},
                  {-2,   6, -13,  37, 115, -20,   9, -4},
                  {-2,   5, -10,  27, 121, -17,   7, -3},
                  {-1,   3,  -6,  17, 125, -13,   5, -2},
                  { 0,   1,  -3,   8, 127,  -7,   3, -1}},

                { { 0, 0, 0,  64,   0, 0, 0, 0 },
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
                  { 0, 0, 0,   8, 120, 0, 0, 0 }}
                };

// NOTE: The tables below must be of the same size.

// The functions described below are sampled at the four most significant
// bits of x^2 + 8 / 256.

// Normalized rate:
// This table models the rate for a Laplacian source with given variance
// when quantized with a uniform quantizer with given stepsize. The
// closed form expression is:
// Rn(x) = H(sqrt(r)) + sqrt(r)*[1 + H(r)/(1 - r)],
// where r = exp(-sqrt(2) * x) and x = qpstep / sqrt(variance),
// and H(x) is the binary entropy function.
__constant int rate_tab_q10[] = {
  65536,  6086,  5574,  5275,  5063,  4899,  4764,  4651,
   4553,  4389,  4255,  4142,  4044,  3958,  3881,  3811,
   3748,  3635,  3538,  3453,  3376,  3307,  3244,  3186,
   3133,  3037,  2952,  2877,  2809,  2747,  2690,  2638,
   2589,  2501,  2423,  2353,  2290,  2232,  2179,  2130,
   2084,  2001,  1928,  1862,  1802,  1748,  1698,  1651,
   1608,  1530,  1460,  1398,  1342,  1290,  1243,  1199,
   1159,  1086,  1021,   963,   911,   864,   821,   781,
    745,   680,   623,   574,   530,   490,   455,   424,
    395,   345,   304,   269,   239,   213,   190,   171,
    154,   126,   104,    87,    73,    61,    52,    44,
     38,    28,    21,    16,    12,    10,     8,     6,
      5,     3,     2,     1,     1,     1,     0,     0,
};
// Normalized distortion:
// This table models the normalized distortion for a Laplacian source
// with given variance when quantized with a uniform quantizer
// with given stepsize. The closed form expression is:
// Dn(x) = 1 - 1/sqrt(2) * x / sinh(x/sqrt(2))
// where x = qpstep / sqrt(variance).
// Note the actual distortion is Dn * variance.
__constant int dist_tab_q10[] = {
     0,     0,     1,     1,     1,     2,     2,     2,
     3,     3,     4,     5,     5,     6,     7,     7,
     8,     9,    11,    12,    13,    15,    16,    17,
    18,    21,    24,    26,    29,    31,    34,    36,
    39,    44,    49,    54,    59,    64,    69,    73,
    78,    88,    97,   106,   115,   124,   133,   142,
   151,   167,   184,   200,   215,   231,   245,   260,
   274,   301,   327,   351,   375,   397,   418,   439,
   458,   495,   528,   559,   587,   613,   637,   659,
   680,   717,   749,   777,   801,   823,   842,   859,
   874,   899,   919,   936,   949,   960,   969,   977,
   983,   994,  1001,  1006,  1010,  1013,  1015,  1017,
  1018,  1020,  1022,  1022,  1023,  1023,  1023,  1024,
};
__constant int xsq_iq_q10[] = {
       0,      4,      8,     12,     16,     20,     24,     28,
      32,     40,     48,     56,     64,     72,     80,     88,
      96,    112,    128,    144,    160,    176,    192,    208,
     224,    256,    288,    320,    352,    384,    416,    448,
     480,    544,    608,    672,    736,    800,    864,    928,
     992,   1120,   1248,   1376,   1504,   1632,   1760,   1888,
    2016,   2272,   2528,   2784,   3040,   3296,   3552,   3808,
    4064,   4576,   5088,   5600,   6112,   6624,   7136,   7648,
    8160,   9184,  10208,  11232,  12256,  13280,  14304,  15328,
   16352,  18400,  20448,  22496,  24544,  26592,  28640,  30688,
   32736,  36832,  40928,  45024,  49120,  53216,  57312,  61408,
   65504,  73696,  81888,  90080,  98272, 106464, 114656, 122848,
  131040, 147424, 163808, 180192, 196576, 212960, 229344, 245728,
};

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
                   __global int *joint_cost, __global int *comp_cost_0,
                   __global int *comp_cost_1) {
  return joint_cost[vp9_get_mv_joint(mv)] +
                    comp_cost_0[mv->row] + comp_cost_1[mv->col];
}

inline int mv_cost_constant(MV *mv,
                   __constant int *joint_cost, __global int *comp_cost_0,
                   __global int *comp_cost_1) {
  return joint_cost[vp9_get_mv_joint(mv)] +
                    comp_cost_0[mv->row] + comp_cost_1[mv->col];
}


short mvsad_err_cost(MV *mv,
                     MV *ref,
                     __global int *nmvsadcost_0,
                     __global int *nmvsadcost_1,
                     int sad_per_bit){
  MV diff;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  return ROUND_POWER_OF_TWO(
    mv_cost_constant(&diff, nmvjointsadcost, nmvsadcost_0,
                                    nmvsadcost_1) * sad_per_bit, 8);
}

int mv_err_cost(MV *mv,
                  MV *ref,
                  __global int *nmvcost_0,
                  __global int *nmvcost_1,
                  __global int *nmvjointcost,
                  int error_per_bit){
  MV diff;

  diff.row = mv->row - ref->row;
  diff.col = mv->col - ref->col;

  return ROUND_POWER_OF_TWO(mv_cost(&diff, nmvjointcost, nmvcost_0,
                                    nmvcost_1) * error_per_bit, 13);
}

inline int gpu_check_bounds(INIT *x,
                     int row,
                     int col,
                     int range) {
  return ((row - range) >= x->mv_row_min) &
         ((row + range) <= x->mv_row_max) &
         ((col - range) >= x->mv_col_min) &
         ((col + range) <= x->mv_col_max);
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

inline int vp9_mv_bit_cost(MV *mv, MV *ref,
                    __global int *mvjcost,
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

ushort calculate_sad(MV *currentmv,
                     __global uchar *ref_frame,
                     __global uchar *cur_frame,
                     int stride)
{
  uchar8 ref,cur;
  ushort8 sad = 0;
  int buffer_offset;
  __global uchar *tmp_ref, *tmp_cur;
  int row;

  buffer_offset = (currentmv->row * stride) + currentmv->col;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    ref = vload8(0,tmp_ref);
    cur = vload8(0,tmp_cur);

    sad += abs_diff(convert_ushort8(ref), convert_ushort8(cur));

    tmp_ref += stride;
    tmp_cur += stride;
  }

  ushort4 final_sad = convert_ushort4(sad.s0123) + convert_ushort4(sad.s4567);
  final_sad.s01  = final_sad.s01 + final_sad.s23;
  return (final_sad.s0 + final_sad.s1);
}

void calculate_fullpel_variance(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                unsigned int *sse,
                                int *sum,
                                int stride) {
  uchar8 output;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  short row;

  *sse = 0;
  *sum = 0;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {

    output = vload8(0, ref_frame);
    ref_frame += stride;

    uchar8 cur = vload8(0, cur_frame);
    cur_frame += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));
  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;
}

inline void var_filter_block2d_bil_both(__global uchar *ref_data,
                                 __global uchar *cur_data,
                                 int stride,
                                 ushort2 horz_filter,
                                 ushort2 vert_filter,
                                 unsigned int *sse,
                                 int *sum) {
  uchar8 output;
  uchar16 src;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;
  uchar8 tmp_out1, tmp_out2;
  uchar8 cur;

  src = vload16(0, ref_data);
  ref_data += stride;

  tmp_out1 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
      convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row += 2) {

    // Iteration 1
    src = vload16(0, ref_data);
    ref_data += stride;

    tmp_out2 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
        convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

    output = convert_uchar8((convert_ushort8(tmp_out1) * vert_filter.s0 +
        convert_ushort8(tmp_out2) * vert_filter.s1 + round_factor) >> filter_shift);

    cur = vload8(0, cur_data);
    cur_data += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    // Iteration 2
    src = vload16(0, ref_data);
    ref_data += stride;

    tmp_out1 = convert_uchar8((convert_ushort8(src.s01234567) * horz_filter.s0 +
        convert_ushort8(src.s12345678) * horz_filter.s1 + round_factor) >> filter_shift);

    output = convert_uchar8((convert_ushort8(tmp_out2) * vert_filter.s0 +
        convert_ushort8(tmp_out1) * vert_filter.s1 + round_factor) >> filter_shift);

    cur = vload8(0, cur_data);
    cur_data += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}

void var_filter_block2d_bil_horizontal(__global uchar *ref_frame,
                                       ushort2 vp9_filter,
                                       __global uchar *cur_frame,
                                       unsigned int *sse,
                                       int *sum,
                                       int stride) {
  uchar8 output;
  uchar16 src_0;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {

    src_0 = vload16(0, ref_frame);
    ref_frame += stride;
    output = convert_uchar8((convert_ushort8(src_0.s01234567) * vp9_filter.s0 +
        convert_ushort8(src_0.s12345678) * vp9_filter.s1 + round_factor) >> filter_shift);

    uchar8 cur = vload8(0, cur_frame);
    cur_frame += stride;

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));
  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}


void var_filter_block2d_bil_vertical(__global uchar *ref_frame,
                                     ushort2 vp9_filter,
                                     __global uchar *cur_frame,
                                     unsigned int *sse,
                                     int *sum,
                                     int stride) {

  uchar8 output;
  uchar8 src_0;
  uchar8 src_1;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;
  short stride_by_8 = stride / 8;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {

    src_0 = vload8(0, ref_frame);
    src_1 = vload8(stride_by_8, ref_frame);
    output = convert_uchar8((convert_ushort8(src_0) * vp9_filter.s0 +
        convert_ushort8(src_1) * vp9_filter.s1 + round_factor) >> filter_shift);

    uchar8 cur = vload8(0, cur_frame);

    diff = convert_short8(output) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    cur_frame += stride;
    ref_frame += stride;

  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

  return;
}

int get_sad(__global uchar *ref_frame, __global uchar *cur_frame,
            int stride, __local int* intermediate_sad, MV this_mv)
{
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
  int sad;

  barrier(CLK_LOCAL_MEM_FENCE);
  intermediate_sad[0] = 0;

  sad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);

  barrier(CLK_LOCAL_MEM_FENCE);
  atomic_add(intermediate_sad, sad);

  barrier(CLK_LOCAL_MEM_FENCE);
  return intermediate_sad[0];

}

MV full_pixel_pattern_search(__global uchar *ref_frame,
                             __global uchar *cur_frame, int stride,
                             __local int* intermediate_sad,
                             MV best_mv, MV fcenter_mv,
                             __global int *nmvsadcost_0,
                             __global int *nmvsadcost_1,
                             INIT *x, int sad_per_bit,
                             int *pbestsad, int pattern,
                             SEARCH_METHODS search_method)
{
  MV this_mv;
  char best_site = -1;
  short br, bc;
  int thissad, bestsad;
  char i, k;
  char next_chkpts_indices[PATTERN_CANDIDATES_REF];

  if (search_method == FAST_DIAMOND && pattern != 0)
    return best_mv;

  int num_candidates = num_candidates_arr[GPU_SEARCH_OFFSET(search_method)][pattern];
  __constant MV (*candidates)[MAX_PATTERN_CANDIDATES] = candidates_arr[GPU_SEARCH_OFFSET(search_method)];

  br = best_mv.row;
  bc = best_mv.col;
  bestsad = *pbestsad;
  best_site = -1;
  if (gpu_check_bounds(x, br, bc, 1 << pattern)) {
    for (i = 0; i < num_candidates; i++) {
      this_mv.row = br + candidates[pattern][i].row;
      this_mv.col = bc + candidates[pattern][i].col;

      thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);

      CHECK_BETTER
    }
  }

  if (best_site == -1) {
    goto exit;
  } else {
    br += candidates[pattern][best_site].row;
    bc += candidates[pattern][best_site].col;
    k = best_site;
  }

#if BLOCK_SIZE_IN_PIXELS != 32
  if (pattern != 0) {
    do {
      best_site = -1;
      next_chkpts_indices[0] = (k == 0) ? num_candidates - 1 : k - 1;
      next_chkpts_indices[1] = k;
      next_chkpts_indices[2] = (k == num_candidates - 1) ? 0 : k + 1;

      if (gpu_check_bounds(x, br, bc, 1 << pattern)) {
        for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
          this_mv.row = br + candidates[pattern][next_chkpts_indices[i]].row;
          this_mv.col = bc + candidates[pattern][next_chkpts_indices[i]].col;

          thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
          CHECK_BETTER
        }
      }

      if (best_site != -1) {
        k = next_chkpts_indices[best_site];
        br += candidates[pattern][k].row;
        bc += candidates[pattern][k].col;
      }
    } while(best_site != -1);
  }
#endif
exit:
  *pbestsad = bestsad;
  best_mv.row = br;
  best_mv.col = bc;

  return best_mv;
}

MV combined_motion_search(__global uchar *ref_frame,
                          __global uchar *cur_frame,
                          __global GPU_INPUT *input_mv,
                          __global GPU_RD_PARAMETERS *rd_parameters,
                          int stride,
                          int mi_rows,
                          int mi_cols,
                          __local int *intermediate_int,
                          SEARCH_METHODS search_method) {
  __global int   *nmvsadcost_0 = rd_parameters->nmvsadcost[0] + MV_MAX;
  __global int   *nmvsadcost_1 = rd_parameters->nmvsadcost[1] + MV_MAX;

  MV nearest_mv, near_mv;
  MV best_mv;
  MV fcenter_mv;
  MV this_mv;
  int num_candidates[MAX_PATTERN_SCALES];
  MV candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES];

  int sad_per_bit = rd_parameters->sad_per_bit;
  INIT x;

  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  int thissad, bestsad = CL_INT_MAX;

  int mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  int mi_col = global_col;

#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows,
                              mi_cols, (BLOCK_SIZE_IN_PIXELS >> 4));

  // compute best mv and best sad among nearest & near mv
  nearest_mv = input_mv->nearest_mv;
  near_mv = input_mv->near_mv;

  vp9_set_mv_search_range_step2(&x, &nearest_mv);

  fcenter_mv.row = nearest_mv.row >> 3;
  fcenter_mv.col = nearest_mv.col >> 3;

  this_mv = fcenter_mv;

  this_mv.col = CLAMP_IT(this_mv.col, x.mv_col_min, x.mv_col_max);
  this_mv.row = CLAMP_IT(this_mv.row, x.mv_row_min, x.mv_row_max);

  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_int, this_mv);
  CHECK_BETTER_CENTER

  this_mv.row = near_mv.row >> 3;
  this_mv.col = near_mv.col >> 3;

  this_mv.col = CLAMP_IT(this_mv.col, x.mv_col_min, x.mv_col_max);
  this_mv.row = CLAMP_IT(this_mv.row, x.mv_row_min, x.mv_row_max);

  thissad = get_sad(ref_frame, cur_frame, stride, intermediate_int, this_mv);
  CHECK_BETTER_CENTER

  // full pel search around best mv
  bestsad += mvsad_err_cost(&best_mv, &fcenter_mv, nmvsadcost_0, nmvsadcost_1,
                            sad_per_bit);

  // Search with pattern = 1
  best_mv = full_pixel_pattern_search(ref_frame, cur_frame, stride,
                                      intermediate_int, best_mv, fcenter_mv,
                                      nmvsadcost_0, nmvsadcost_1,
                                      &x, sad_per_bit,
                                      &bestsad, 1,
                                      search_method);
  // Search with pattern = 0
  best_mv = full_pixel_pattern_search(ref_frame, cur_frame, stride,
                                      intermediate_int, best_mv, fcenter_mv,
                                      nmvsadcost_0, nmvsadcost_1,
                                      &x, sad_per_bit,
                                      &bestsad, 0,
                                      search_method);

  best_mv.row = best_mv.row * 8;
  best_mv.col = best_mv.col * 8;

  return best_mv;
}


inline int get_msb(unsigned int n) {
  return 31 ^ clz(n);
}

inline int rd_less_than_thresh(int64_t best_rd, int thresh, int thresh_fact) {
    return best_rd < ((int64_t)thresh * thresh_fact >> 5) || thresh == INT_MAX;
}

void model_rd_norm(int xsq_q10, int *r_q10, int *d_q10) {
  const int tmp = (xsq_q10 >> 2) + 8;
  const int k = get_msb(tmp) - 3;
  const int xq = (k << 3) + ((tmp >> k) & 0x7);
  const int one_q10 = 1 << 10;
  const int a_q10 = ((xsq_q10 - xsq_iq_q10[xq]) << 10) >> (2 + k);
  const int b_q10 = one_q10 - a_q10;
  *r_q10 = (rate_tab_q10[xq] * b_q10 + rate_tab_q10[xq + 1] * a_q10) >> 10;
  *d_q10 = (dist_tab_q10[xq] * b_q10 + dist_tab_q10[xq + 1] * a_q10) >> 10;
}

void vp9_model_rd_from_var_lapndz(unsigned int var, unsigned int n,
                                  unsigned int qstep, int *rate,
                                  int64_t *dist) {
  // This function models the rate and distortion for a Laplacian
  // source with given variance when quantized with a uniform quantizer
  // with given stepsize. The closed form expressions are in:
  // Hang and Chen, "Source Model for transform video coder and its
  // application - Part I: Fundamental Theory", IEEE Trans. Circ.
  // Sys. for Video Tech., April 1997.
  if (var == 0) {
    *rate = 0;
    *dist = 0;
  } else {
    int d_q10, r_q10;
    const uint32_t MAX_XSQ_Q10 = 245727;
    const uint64_t xsq_q10_64 =
        ((((uint64_t)qstep * qstep * n) << 10) + (var >> 1)) / var;
    const int xsq_q10 = (int)MIN(xsq_q10_64, MAX_XSQ_Q10);
    model_rd_norm(xsq_q10, &r_q10, &d_q10);
    *rate = (n * r_q10 + 2) >> 2;
    *dist = (var * (int64_t)d_q10 + 512) >> 10;
  }
}

void inter_prediction(__global uchar *ref_data,
                      __global uchar *cur_frame,
                      int horz_subpel,
                      int vert_subpel,
                      int filter_type,
                      __local uchar8 *intermediate,
                      __global int *psum,
                      __global uint *psse) {
  __local uchar8 *intermediate_uchar8;
  __local int *intermediate_int = (__local int *)intermediate;
  uchar8 curr_data = vload8(0, cur_frame);

  uchar16 ref;
  uchar8 ref_u8;

  short8 inter;
  int4 inter_out1;
  short8 sum = (short8)(0, 0, 0, 0, 0, 0, 0, 0);
  uint4 sse = (uint4)(0, 0, 0, 0);
  short8 c;

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int inter_offset = (local_row * LOCAL_STRIDE * PIXEL_ROWS_PER_WORKITEM) + local_col;

  short8 tmp;
  int4 shift_val = (int4)(1 << 14);
  int4 tmp1;
  uchar8 temp_out;
  uchar8 out_uni;
  uchar8 out_bi;
  int i;

  if (!vert_subpel) {
    /* L0 only x_frac */
    char8 filt = filter[filter_type][horz_subpel];

    for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {
      ref = vload16(0, ref_data - 3);

      inter = (short8)(-1 << 14);

      tmp = filt.s0;
      inter += convert_short8(ref.s01234567) * tmp;
      tmp = filt.s1;
      inter += convert_short8(ref.s12345678) * tmp;
      tmp = filt.s2;
      inter += convert_short8(ref.s23456789) * tmp;
      tmp = filt.s3;
      inter += convert_short8(ref.s3456789a) * tmp;
      tmp = filt.s4;
      inter += convert_short8(ref.s456789ab) * tmp;
      tmp = filt.s5;
      inter += convert_short8(ref.s56789abc) * tmp;
      tmp = filt.s6;
      inter += convert_short8(ref.s6789abcd) * tmp;
      tmp = filt.s7;
      inter += convert_short8(ref.s789abcde) * tmp;

      if (horz_subpel == 0) {
        tmp1 = (1 << 5) + shift_val;
        inter_out1 = convert_int4(inter.s0123);

        out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 6);
        inter_out1 = convert_int4(inter.s4567);

        out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 6);
      } else {
        tmp1 = (1 << 6) + shift_val;
        inter_out1 = convert_int4(inter.s0123);

        out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
        inter_out1 =convert_int4(inter.s4567);

        out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      }

      curr_data = vload8(0, cur_frame);
      short8 diff = convert_short8(curr_data) - convert_short8(out_uni);
      sum += diff;
      sse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
      sse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

      ref_data += STRIDE;
      cur_frame += STRIDE;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  } else if(!horz_subpel) {
    /* L0 only y_frac */
    char8 filt = filter[filter_type][vert_subpel];
    ref_data -= (3 * STRIDE);
    for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {

      inter = (short8)(-1 << 14);
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s0;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s1;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s2;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s3;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s4;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s5;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s6;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += STRIDE;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s7;
      inter += convert_short8(ref_u8) * tmp;

      tmp1 = (1 << 6) + shift_val;
      inter_out1 = convert_int4(inter.s0123);

      out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      inter_out1 = convert_int4(inter.s4567);

      out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);

      curr_data = vload8(0, cur_frame);
      short8 diff = convert_short8(curr_data) - convert_short8(out_uni);
      sum += diff;
      sse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
      sse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

      ref_data  -= 6 * STRIDE;
      cur_frame += STRIDE;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  } else {
    char8 filt = filter[filter_type][horz_subpel];
    ref_data -= (3 * STRIDE);

    for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {
      inter = (short8)(-1 << 14);
      ref = vload16(0, ref_data - 3);

      tmp = filt.s0;
      inter += convert_short8(ref.s01234567) * tmp;
      tmp = filt.s1;
      inter += convert_short8(ref.s12345678) * tmp;
      tmp = filt.s2;
      inter += convert_short8(ref.s23456789) * tmp;
      tmp = filt.s3;
      inter += convert_short8(ref.s3456789a) * tmp;
      tmp = filt.s4;
      inter += convert_short8(ref.s456789ab) * tmp;
      tmp = filt.s5;
      inter += convert_short8(ref.s56789abc) * tmp;
      tmp = filt.s6;
      inter += convert_short8(ref.s6789abcd) * tmp;
      tmp = filt.s7;
      inter += convert_short8(ref.s789abcde) * tmp;
      tmp1 = (1 << 6) + shift_val;
      inter_out1 = convert_int4(inter.s0123);
      temp_out.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      inter_out1 = convert_int4(inter.s4567);
      temp_out.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      intermediate[inter_offset] = temp_out;

      ref_data += STRIDE;
      inter_offset += LOCAL_STRIDE;
    }

    if (local_row < 8 / PIXEL_ROWS_PER_WORKITEM) {
      ref_data += (BLOCK_SIZE_IN_PIXELS - PIXEL_ROWS_PER_WORKITEM) * STRIDE;
      inter_offset += (BLOCK_SIZE_IN_PIXELS - PIXEL_ROWS_PER_WORKITEM) * LOCAL_STRIDE;

      for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {
        ref = vload16(0, ref_data - 3);
        inter = (short8)(-1 << 14);
        tmp = filt.s0;
        inter += convert_short8(ref.s01234567) * tmp;
        tmp = filt.s1;
        inter += convert_short8(ref.s12345678) * tmp;
        tmp = filt.s2;
        inter += convert_short8(ref.s23456789) * tmp;
        tmp = filt.s3;
        inter += convert_short8(ref.s3456789a) * tmp;
        tmp = filt.s4;
        inter += convert_short8(ref.s456789ab) * tmp;
        tmp = filt.s5;
        inter += convert_short8(ref.s56789abc) * tmp;
        tmp = filt.s6;
        inter += convert_short8(ref.s6789abcd) * tmp;
        tmp = filt.s7;
        inter += convert_short8(ref.s789abcde) * tmp;
        tmp1 = (1 << 6) + shift_val;
        inter_out1 = convert_int4(inter.s0123);
        temp_out.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
        inter_out1 = convert_int4(inter.s4567);
        temp_out.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
        intermediate[inter_offset] = temp_out;

        ref_data += STRIDE;
        inter_offset += LOCAL_STRIDE;
      }
      inter_offset -= BLOCK_SIZE_IN_PIXELS * LOCAL_STRIDE;
    }

    inter_offset -= (PIXEL_ROWS_PER_WORKITEM) * LOCAL_STRIDE;
    intermediate_uchar8 = intermediate + inter_offset + (3 * LOCAL_STRIDE);
    filt = filter[filter_type][vert_subpel];
    barrier(CLK_LOCAL_MEM_FENCE);
    for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {
      inter = (short8)(-1 << 14);
      ref_u8 = intermediate_uchar8[-3 * LOCAL_STRIDE];
      tmp = filt.s0;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[-2 * LOCAL_STRIDE];
      tmp = filt.s1;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[-1 * LOCAL_STRIDE];
      tmp = filt.s2;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[0 * LOCAL_STRIDE];
      tmp = filt.s3;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[1 * LOCAL_STRIDE];
      tmp = filt.s4;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[2 * LOCAL_STRIDE];
      tmp = filt.s5;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[3 * LOCAL_STRIDE];
      tmp = filt.s6;
      inter += convert_short8(ref_u8) * tmp;
      ref_u8 = intermediate_uchar8[4 * LOCAL_STRIDE];
      tmp = filt.s7;
      inter += convert_short8(ref_u8) * tmp;

      tmp1 = (1 << 6) + shift_val;
      inter_out1 = convert_int4(inter.s0123);

      out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      inter_out1 = convert_int4(inter.s4567);

      out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);

      curr_data = vload8(0, cur_frame);
      short8 diff = convert_short8(curr_data) - convert_short8(out_uni);
      sum += diff;
      sse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
      sse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

      intermediate_uchar8 += LOCAL_STRIDE;
      cur_frame += STRIDE;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  }
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_full_pixel_search(__global uchar *ref_frame,
                           __global uchar *cur_frame,
                           int stride,
                           __global GPU_INPUT *mv_input,
                           __global GPU_OUTPUT *sse_variance_output,
                           __global GPU_RD_PARAMETERS *rd_parameters,
                           __global subpel_sum_sse *rd_calc_tmp_buffers,
                           int mi_rows,
                           int mi_cols,
                           SEARCH_METHODS search_method) {
  __local int intermediate_int[1];
  __global int   *nmvcost_0        = rd_parameters->mvcost[0] + MV_MAX;
  __global int   *nmvcost_1        = rd_parameters->mvcost[1] + MV_MAX;
  __global int   *nmvjointcost     = rd_parameters->nmvjointcost;

  short global_row = get_global_id(1);
  short global_col = get_global_id(0);

  short group_col = get_group_id(0);
  int group_stride = get_num_groups(0);

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (group_col * BLOCK_SIZE_IN_PIXELS) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + group_col);

  MV best_mv, nearest_mv;

  mv_input += group_offset;

  sse_variance_output += group_offset;

  cur_frame += global_offset;

  ref_frame += global_offset;

  rd_calc_tmp_buffers += group_offset;

  if (!mv_input->do_compute)
    goto exit;

  if (!mv_input->do_newmv)
    goto exit;

  int64_t best_rd = sse_variance_output->best_rd;

  best_mv = combined_motion_search(ref_frame, cur_frame,
                                   mv_input, rd_parameters,
                                   stride,
                                   mi_rows, mi_cols,
                                   intermediate_int, search_method);

  nearest_mv = mv_input->nearest_mv;

  int rate_mv = vp9_mv_bit_cost(&best_mv, &nearest_mv,
                                nmvjointcost, nmvcost_0, nmvcost_1,
                                MV_COST_WEIGHT);

  int rate_mode = rd_parameters->inter_mode_cost[mv_input->mode_context]
                                                 [GPU_INTER_OFFSET(NEWMV)];
  mv_input->do_newmv = !(RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
                                (rate_mv + rate_mode), 0) > best_rd);

  if (mv_input->do_newmv && local_col == 0 && local_row == 0) {
    unsigned int besterr;
    int error_per_bit = rd_parameters->error_per_bit;
    MV minmv, maxmv;
    INIT x;
    int tc, tr;
    int hstep_limit = 6;
    MV fcenter_mv;

    fcenter_mv.row = nearest_mv.row >> 3;
    fcenter_mv.col = nearest_mv.col >> 3;

    int mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
    int mi_col = global_col;

#if BLOCK_SIZE_IN_PIXELS == 32
    mi_row = (mi_row >> 2) << 2;
    mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
    mi_row = (mi_row >> 1) << 1;
    mi_col = (mi_col >> 1) << 1;
#endif

    vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows,
                                mi_cols, (BLOCK_SIZE_IN_PIXELS >> 4));

    minmv.col = MAX(x.mv_col_min * 8, fcenter_mv.col - MV_MAX);
    maxmv.col = MIN(x.mv_col_max * 8, fcenter_mv.col + MV_MAX);
    minmv.row = MAX(x.mv_row_min * 8, fcenter_mv.row - MV_MAX);
    maxmv.row = MIN(x.mv_row_max * 8, fcenter_mv.row + MV_MAX);

    tr = best_mv.row;
    tc = best_mv.col;

    if (!(tc - hstep_limit >= minmv.col && tc + hstep_limit <= maxmv.col
        && tr - hstep_limit >= minmv.row && tr + hstep_limit <= maxmv.row))  {
      mv_input->do_newmv = -1;
    }

    sse_variance_output->rate_mv = rate_mv;
    sse_variance_output->mv = best_mv;

    __global int *intermediate_sum_sse = (__global int *)rd_calc_tmp_buffers;
    vstore8(0, 0, intermediate_sum_sse);
    vstore2(0, 4, intermediate_sum_sse);
  }

exit:
  return;
}

__kernel
void vp9_full_pixel_search_zeromv(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    int mi_rows,
    int mi_cols
) {
  __global uchar *tmp_ref, *tmp_cur;
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  int global_offset = (global_row * stride * BLOCK_SIZE_IN_PIXELS) +
                      (global_col * BLOCK_SIZE_IN_PIXELS);
  int sum;
  uint sse, variance;
  int rate, actual_rate;
  int64_t dist, actual_dist;
  int64_t this_rd, best_rd = INT64_MAX;
  int skip_txfm;
  TX_SIZE tx_size;
  TX_MODE tx_mode = rd_parameters->tx_mode;
  int dc_quant = rd_parameters->dc_quant;
  int ac_quant = rd_parameters->ac_quant;

  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

#if BLOCK_SIZE_IN_PIXELS == 32
  int bsize = BLOCK_32X32;
  int gpu_bsize = GPU_BLOCK_32X32;
#elif BLOCK_SIZE_IN_PIXELS == 16
  int bsize = BLOCK_16X16;
  int gpu_bsize = GPU_BLOCK_16X16;
#endif

  mv_input += (global_row * global_stride + global_col);
  sse_variance_output += (global_row * global_stride + global_col);

  uchar8 curr_data, pred_data;

  cur_frame += global_offset;
  ref_frame += global_offset;

  tmp_ref = ref_frame;
  tmp_cur = cur_frame;

  if (!mv_input->do_compute)
    goto exit;

  // ZEROMV not required for BLOCK_32X32
  if (BLOCK_SIZE_IN_PIXELS != 32) {
    int row, col;
    short8 vsum = 0;
    uint4 vsse = 0;

    for(row = 0; row < BLOCK_SIZE_IN_PIXELS; row++)
    {
      curr_data = vload8(0, cur_frame);
      pred_data = vload8(0, ref_frame);
      short8 diff = convert_short8(curr_data) - convert_short8(pred_data);
      vsum += diff;
      vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
      vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

      if (BLOCK_SIZE_IN_PIXELS == 16) {
        curr_data = vload8(1, cur_frame);
        pred_data = vload8(1, ref_frame);
        short8 diff = convert_short8(curr_data) - convert_short8(pred_data);
        vsum += diff;
        vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
        vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));
      }
      cur_frame += stride;
      ref_frame += stride;
    }

    vsum.s0123 = vsum.s0123 + vsum.s4567;
    vsum.s01   = vsum.s01   + vsum.s23;
    sum        = (int)vsum.s0 + vsum.s1;

    vsse.s01   = vsse.s01   + vsse.s23;
    vsse.s0    = vsse.s0    + vsse.s1;
    sse = vsse.s0;
    variance = sse - ((long)sum * sum) / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS);

    CALCULATE_RATE_DIST
    actual_rate += rd_parameters->inter_mode_cost[mv_input->mode_context]
                                                  [GPU_INTER_OFFSET(ZEROMV)];
    this_rd = RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
                     actual_rate, actual_dist);

    best_rd = this_rd;
    sse_variance_output->returnrate = actual_rate;
    sse_variance_output->returndistortion = actual_dist;
    sse_variance_output->best_rd = this_rd;
    sse_variance_output->best_mode = ZEROMV;
    sse_variance_output->best_pred_filter = EIGHTTAP;
    sse_variance_output->skip_txfm = skip_txfm;
    sse_variance_output->tx_size = tx_size;

    if (this_rd < (int64_t)(1 << num_pels_log2_lookup[bsize])) {
      mv_input->do_newmv = 0;
      goto exit;
    }

    int mode_rd_thresh =
        rd_parameters->threshes[gpu_bsize][mode_idx[0][INTER_OFFSET(NEWMV)]];

    if (rd_less_than_thresh(best_rd, mode_rd_thresh,
            rd_parameters->thresh_fact_newmv[gpu_bsize])) {
      mv_input->do_newmv = 0;
      goto exit;
    }

#if BLOCK_SIZE_IN_PIXELS == 16
    if (skip_txfm) {
      mv_input->do_newmv = 0;
    }
#endif

  } else {
    sse_variance_output->best_rd = INT64_MAX;
    sse_variance_output->best_mode = ZEROMV;
  }
exit:
  return;
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_sub_pixel_search_halfpel_filtering(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global subpel_sum_sse *rd_calc_tmp_buffers
) {
  short global_row = get_global_id(1);

  short group_col = get_group_id(0);
  int group_stride = get_num_groups(0) >> 2;

  int local_col = get_local_id(0);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      ((group_col >> 2) * BLOCK_SIZE_IN_PIXELS) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + (group_col >> 2));

  mv_input += group_offset;

  if (!mv_input->do_compute)
    goto exit;

  if (mv_input->do_newmv < 1)
    goto exit;

  rd_calc_tmp_buffers += group_offset;
  sse_variance_output += group_offset;

  cur_frame += global_offset;
  ref_frame += global_offset;

  int sum;
  unsigned int sse;

  MV best_mv = sse_variance_output->mv;
  int buffer_offset;
  int local_offset;

  int idx = (group_col & 3);

  __global int *intermediate_sum_sse = (__global int *)rd_calc_tmp_buffers;

  /* Half pel */

  const char hstep = 4;
#if !INTEL_HD_GRAPHICS
  best_mv = best_mv + sub_pel_offset[idx] * hstep;
#else
  best_mv.row = best_mv.row + sub_pel_offset[idx].row * hstep;
  best_mv.col = best_mv.col + sub_pel_offset[idx].col * hstep;
#endif
  idx *= 2;
  buffer_offset = ((best_mv.row >> 3) * stride) + (best_mv.col >> 3);
  ref_frame += buffer_offset;

  if(idx == 2) {
    calculate_fullpel_variance(ref_frame, cur_frame, &sse, &sum, stride);
    atomic_add(intermediate_sum_sse + 8, sum);
    atomic_add(intermediate_sum_sse + 8 + 1, sse);
  }

  if(idx < 4) {
    var_filter_block2d_bil_horizontal(ref_frame,
                                      BILINEAR_FILTERS_2TAP(sp(best_mv.col)),
                                      cur_frame, &sse, &sum, stride);
  } else {
    var_filter_block2d_bil_vertical(ref_frame,
                                    BILINEAR_FILTERS_2TAP(sp(best_mv.row)),
                                    cur_frame, &sse, &sum, stride);
  }
  atomic_add(intermediate_sum_sse + idx, sum);
  atomic_add(intermediate_sum_sse + idx + 1, sse);

exit:
  return;
}

__kernel
void vp9_sub_pixel_search_halfpel_bestmv(__global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global subpel_sum_sse *rd_calc_tmp_buffers
) {
  short global_col = get_global_id(0);
  short global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  int group_offset = (global_row * global_stride + global_col);

  mv_input            += group_offset;
  if(!mv_input->do_compute)
    goto exit;

  if(mv_input->do_newmv < 1)
    goto exit;

  sse_variance_output += group_offset;

  int sum, tr, tc;
  unsigned int besterr, sse, thiserr;
  const char hstep = 4;
  __global int *intermediate_sum_sse = (__global int *)(rd_calc_tmp_buffers + group_offset);

  MV best_mv = sse_variance_output->mv;
  besterr = INT32_MAX;
  /*Part 1*/
  {
    tr = best_mv.row;
    tc = best_mv.col;

    CHECK_BETTER_SUBPEL(tr, tc, 8);
    CHECK_BETTER_SUBPEL(tr, (tc - hstep), 0);
    CHECK_BETTER_SUBPEL(tr, (tc + hstep), 2);
    CHECK_BETTER_SUBPEL((tr - hstep), tc, 4);
    CHECK_BETTER_SUBPEL((tr + hstep), tc, 6);
  }

  intermediate_sum_sse[8] = besterr;
  sse_variance_output->mv = best_mv;
  vstore8(0, 0, intermediate_sum_sse);

exit:
  return;
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_sub_pixel_search_quarterpel_filtering(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global subpel_sum_sse *rd_calc_tmp_buffers
) {
  short global_row = get_global_id(1);

  short group_col = get_group_id(0);
  int group_stride = get_num_groups(0) >> 2;

  int local_col = get_local_id(0);
  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      ((group_col >> 2) * BLOCK_SIZE_IN_PIXELS) +
                      (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  int group_offset = (global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + (group_col >> 2));

  mv_input += group_offset;

  if(!mv_input->do_compute)
    goto exit;

  if(mv_input->do_newmv < 1)
    goto exit;

  rd_calc_tmp_buffers += group_offset;
  sse_variance_output += group_offset;

  cur_frame += global_offset;
  ref_frame += global_offset;

  int sum;
  unsigned int sse;

  MV best_mv = sse_variance_output->mv;
  int buffer_offset;

  int idx = (group_col & 3);

  __global int *intermediate_sum_sse = (__global int *)rd_calc_tmp_buffers;

  /* Quarter pel */

  const char hstep = 2;
#if !INTEL_HD_GRAPHICS
  best_mv = best_mv + sub_pel_offset[idx] * hstep;
#else
  best_mv.row = best_mv.row + sub_pel_offset[idx].row * hstep;
  best_mv.col = best_mv.col + sub_pel_offset[idx].col * hstep;
#endif
  idx *= 2;

  buffer_offset = ((best_mv.row >> 3) * stride) + (best_mv.col >> 3);
  ref_frame += buffer_offset;

  var_filter_block2d_bil_both(ref_frame, cur_frame, stride,
                              BILINEAR_FILTERS_2TAP(sp(best_mv.col)),
                              BILINEAR_FILTERS_2TAP(sp(best_mv.row)),
                              &sse, &sum);

  atomic_add(intermediate_sum_sse + idx, sum);
  atomic_add(intermediate_sum_sse + idx + 1, sse);

exit:
  return;
}

__kernel
void vp9_sub_pixel_search_quarterpel_bestmv(__global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global subpel_sum_sse *rd_calc_tmp_buffers
) {
  short global_col = get_global_id(0);
  short global_row = get_global_id(1);
  int global_stride = get_global_size(0);

  mv_input            += (global_row * global_stride + global_col);
  rd_calc_tmp_buffers += (global_row * global_stride + global_col);
  __global int *intermediate_sum_sse = (__global int *)rd_calc_tmp_buffers;
  if(!mv_input->do_compute)
    goto exit;

  if(mv_input->do_newmv < 1)
    goto exit;

  sse_variance_output += (global_row * global_stride + global_col);

  int sum, tr, tc;
  unsigned int besterr, sse, thiserr;

  const char hstep = 2;

  MV best_mv = sse_variance_output->mv;
  besterr = intermediate_sum_sse[8];

  /*Part 2*/
  {
    tr = best_mv.row;
    tc = best_mv.col;

    CHECK_BETTER_SUBPEL(tr, (tc - hstep), 0);
    CHECK_BETTER_SUBPEL(tr, (tc + hstep), 2);
    CHECK_BETTER_SUBPEL((tr - hstep), tc, 4);
    CHECK_BETTER_SUBPEL((tr + hstep), tc, 6);
  }


  sse_variance_output->mv = best_mv;

exit:
  vstore8(0, 0, intermediate_sum_sse);
  vstore2(0, 4, intermediate_sum_sse);
  return;
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_inter_prediction_and_sse(__global uchar *ref_frame,
    __global uchar *cur_frame,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global rd_calc_buffers *rd_calc_tmp_buffers)
{
  __local uchar8 intermediate_uchar8[(BLOCK_SIZE_IN_PIXELS * (BLOCK_SIZE_IN_PIXELS + 8)) / NUM_PIXELS_PER_WORKITEM];
  __local int *intermediate_int = (__local int *)intermediate_uchar8;

  int global_row = get_global_id(1);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0) / 2;

  int sum;
  uint sse, variance;


  int group_idx = group_col % 4;
  if (group_idx == 3) {
    group_col -= 2;
  }

  int group_offset = global_row / (BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM) *
      group_stride + (group_col / 2);

  mv_input += group_offset;

  int filter_type = interp_filter[group_idx];

  if (group_col % 2 != 0 && mv_input->filter_type != SWITCHABLE) {
    group_col += 2;
    mv_input++;
    group_offset++;
  }

  if(group_col >= group_stride * 2)
    goto exit;

  if(!mv_input->do_compute)
    goto exit;

  if(!mv_input->do_newmv)
    goto exit;

  int local_col = get_local_id(0);
  int global_offset = (global_row * STRIDE * PIXEL_ROWS_PER_WORKITEM) +
      ((group_col / 2) * BLOCK_SIZE_IN_PIXELS) + (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * STRIDE) + VP9_ENC_BORDER_IN_PIXELS;

  sse_variance_output += group_offset;
  rd_calc_tmp_buffers += group_offset;

  cur_frame += global_offset;

  MV best_mv;

  best_mv = sse_variance_output->mv;
  int mv_row = best_mv.row;
  int mv_col = best_mv.col;
  int mv_offset = ((mv_row >> SUBPEL_BITS) * STRIDE) + (mv_col >> SUBPEL_BITS);
  int horz_subpel = (mv_col & SUBPEL_MASK) << 1;
  int vert_subpel = (mv_row & SUBPEL_MASK) << 1;

  ref_frame += global_offset + mv_offset;

  if(filter_type != EIGHTTAP && !horz_subpel && !vert_subpel)
    goto exit;

  inter_prediction(ref_frame, cur_frame, horz_subpel, vert_subpel,
                   filter_type, intermediate_uchar8,
                   &rd_calc_tmp_buffers->sum[filter_type],
                   &rd_calc_tmp_buffers->sse[filter_type]);

exit:
  return;
}

__kernel
void vp9_rd_calculation(__global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    __global rd_calc_buffers *rd_calc_tmp_buffers)
{
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  uchar8 curr_data;

  int sum;
  uint sse, variance;
  int rate, actual_rate, newmv_rate = INT32_MAX;
  int64_t dist, actual_dist, newmv_dist = INT64_MAX;
  int64_t this_rd, best_rd = INT64_MAX;
  int newmv_skip_txfm = 0, skip_txfm;
  TX_SIZE newmv_tx_size, tx_size;
  TX_MODE tx_mode = rd_parameters->tx_mode;
  int dc_quant = rd_parameters->dc_quant;
  int ac_quant = rd_parameters->ac_quant;

  mv_input      += (global_row * global_stride + global_col);

#if BLOCK_SIZE_IN_PIXELS == 32
  int bsize = BLOCK_32X32;
#elif BLOCK_SIZE_IN_PIXELS == 16
  int bsize = BLOCK_16X16;
#endif

  sse_variance_output += (global_row * global_stride + global_col);

  PREDICTION_MODE best_mode = ZEROMV;
  INTERP_FILTER newmv_filter, best_pred_filter = EIGHTTAP;

  if(!mv_input->do_compute)
    goto exit;

  if(!mv_input->do_newmv)
    goto exit;

  // ZEROMV not required for BLOCK_32X32
  if (BLOCK_SIZE_IN_PIXELS != 32) {
    best_rd = sse_variance_output->best_rd;
  }
  MV best_mv = sse_variance_output->mv;
  int mv_row = best_mv.row;
  int mv_col = best_mv.col;
  int horz_subpel = (mv_col & SUBPEL_MASK) << 1;
  int vert_subpel = (mv_row & SUBPEL_MASK) << 1;

  int rate_mv = sse_variance_output->rate_mv;

  int64_t cost, best_cost = INT64_MAX;
  int filter_type, end_filter_type;
  if(mv_input->filter_type == SWITCHABLE && (horz_subpel || vert_subpel))
    end_filter_type = EIGHTTAP_SHARP;
  else
    end_filter_type = EIGHTTAP;

  rd_calc_tmp_buffers += (global_row * global_stride + global_col);

  for (filter_type = EIGHTTAP; filter_type <= end_filter_type; ++filter_type) {
    sum = rd_calc_tmp_buffers->sum[filter_type];
    sse = rd_calc_tmp_buffers->sse[filter_type];
    variance = sse - ((long)sum * sum) / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS);
    CALCULATE_RATE_DIST
    cost = RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
      rd_parameters->switchable_interp_costs[filter_type] + actual_rate, actual_dist);
    if (cost < best_cost) {
      best_cost = cost;
      newmv_rate = actual_rate;
      newmv_dist = actual_dist;
      newmv_filter = filter_type;
      newmv_skip_txfm = skip_txfm;
      newmv_tx_size = tx_size;
    }
  }
  newmv_rate += rate_mv;
  newmv_rate += rd_parameters->inter_mode_cost[mv_input->mode_context]
                                              [GPU_INTER_OFFSET(NEWMV)];
  this_rd = RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
      newmv_rate, newmv_dist);
  if (this_rd < best_rd) {
    sse_variance_output->returnrate = newmv_rate;
    sse_variance_output->returndistortion = newmv_dist;
    sse_variance_output->best_rd = this_rd;
    sse_variance_output->best_mode = NEWMV;
    sse_variance_output->best_pred_filter = newmv_filter;
    sse_variance_output->skip_txfm = newmv_skip_txfm;
    sse_variance_output->tx_size = newmv_tx_size;
  }

exit:
  return;
}

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

///* estimated cost of a motion vector (r,c) */
#define MVC(v, r, c)                                         \
     (((nmvjointcost[((r) != refmv.row) * 2 + ((c) != refmv.col)]\
          + nmvcost_0[((r) - refmv.row)]                     \
                 + nmvcost_1[((c) - refmv.col)])             \
                      * error_per_bit + 4096) >> 13)

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
    atomic_add(psse, sse.s0);                                           \

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

typedef struct GPU_OUTPUT_STAGE1 {
  MV mv;
  int rate_mv;
} GPU_OUTPUT_STAGE1;

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
  int          sum[EIGHTTAP_SHARP + 1];
  uint         sse[EIGHTTAP_SHARP + 1];
  int          returnrate;
  int64_t      returndistortion;
  int64_t      best_rd;
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
  int threshes[MAX_MODES];
  int thresh_fact_newmv;
  TX_MODE tx_mode;
  int dc_quant;
  int ac_quant;
  int nmvsadcost[2][MV_VALS];
  int mvcost[2][MV_VALS];
  int sad_per_bit;
  int error_per_bit;
  int nmvjointcost[MV_JOINTS];
} GPU_RD_PARAMETERS;

__constant int nmvjointsadcost[MV_JOINTS] = {600,300,300,300};

__constant int hex_num_candidates[MAX_PATTERN_SCALES] = {8, 6};

__constant MV hex_candidates[MAX_PATTERN_SCALES][MAX_PATTERN_CANDIDATES] = {
    {{-1, -1}, {0, -1}, {1, -1}, {1, 0}, {1, 1}, { 0, 1}, { -1, 1}, {-1, 0}},
    {{-1, -2}, {1, -2}, {2, 0}, {1, 2}, { -1, 2}, { -2, 0}},
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

int clamp_it(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

void clamp_gpu_mv(MV *mv, int min_col, int max_col,
                            int min_row, int max_row) {
  mv->col = clamp_it(mv->col, min_col, max_col);
  mv->row = clamp_it(mv->row, min_row, max_row);
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
                                int stride,
                                unsigned int *sse,
                                int *sum,
                                MV *submv) {
  uchar8 ref,cur;
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;

  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;
  buffer_offset = ((submv->row >> 3) * stride) + (submv->col >> 3);
  *sum = 0;
  *sse = 0;

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {
    ref = vload8(0,tmp_ref);
    cur = vload8(0,tmp_cur);

    diff = convert_short8(ref) - convert_short8(cur);
    vsum += diff;
    vsse += convert_uint4(convert_int4(diff.s0123) * convert_int4(diff.s0123));
    vsse += convert_uint4(convert_int4(diff.s4567) * convert_int4(diff.s4567));

    tmp_ref += stride;
    tmp_cur += stride;
  }
  vsum.s0123 = vsum.s0123 + vsum.s4567;
  vsum.s01 = vsum.s01 + vsum.s23;
  *sum = vsum.s0 + vsum.s1;

  vsse.s01 = vsse.s01 + vsse.s23;
  *sse = vsse.s0 + vsse.s1;

}

void var_filter_block2d_bil_both(__global uchar *ref_data,
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
  uchar8 src_0;
  uchar8 src_1;
  ushort8 round_factor = 1 << (FILTER_BITS - 1);
  ushort8 filter_shift = FILTER_BITS;
  short8 diff;
  short8 vsum = 0;
  uint4 vsse = 0;
  int row;

  for(row = 0; row < PIXEL_ROWS_PER_WORKITEM; row++) {

    src_0 = vload8(0, ref_frame);
    src_1 = vload8(0, ref_frame + 1);
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
  int stride_by_8 = stride / 8;

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

void calculate_subpel_variance(__global uchar *ref_frame,
                               __global uchar *cur_frame,
                               int stride,
                               int xoffset,
                               int yoffset,
                               int row,
                               int col,
                               unsigned int *sse,
                               int *sum) {
  int buffer_offset;
  __global uchar *tmp_ref,*tmp_cur;

  buffer_offset = ((row >> 3) * stride) + (col >> 3);

  tmp_ref = ref_frame + buffer_offset;
  tmp_cur = cur_frame;

// Enabling this piece of code causes a crash in Intel HD graphics. But it works
// fine in Mali GPU and AMD GPU. Must be an issue with Intel's driver
#if !INTEL_HD_GRAPHICS
  if(!yoffset) {
    var_filter_block2d_bil_horizontal(tmp_ref,
                                      BILINEAR_FILTERS_2TAP(xoffset),
                                      tmp_cur, sse, sum, stride);
  } else if(!xoffset) {
    var_filter_block2d_bil_vertical(tmp_ref,
                                    BILINEAR_FILTERS_2TAP(yoffset),
                                    tmp_cur, sse, sum, stride);

  } else
#endif
  {
    var_filter_block2d_bil_both(tmp_ref, tmp_cur, stride,
                                BILINEAR_FILTERS_2TAP(xoffset),
                                BILINEAR_FILTERS_2TAP(yoffset),
                                sse, sum);
  }
}

inline int get_sad(__global uchar *ref_frame, __global uchar *cur_frame,
                   int stride, __local int* intermediate_sad, MV this_mv)
{
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
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
#else
  int thissad = calculate_sad(&this_mv, ref_frame, cur_frame, stride);
  return thissad;
#endif

}

inline MV get_best_mv(__global uchar *ref_frame, __global uchar *cur_frame,
                      int stride, __local int* intermediate_sad,
                      MV nearest_mv, MV near_mv, MV pred_mv)
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

  return best_mv;
}

MV full_pixel_search(__global uchar *ref_frame,
                            __global uchar *cur_frame, int stride,
                            __local int* intermediate_sad,
                            MV best_mv, MV fcenter_mv,
                            __global int *nmvsadcost_0,
                            __global int *nmvsadcost_1,
                            INIT *x, int sad_per_bit)
{
  MV this_mv;
  int best_site = -1;
  int i, k;
  short br, bc;
  int pattern = 1;
  int next_chkpts_indices[PATTERN_CANDIDATES_REF];
  int thissad, bestsad;

  clamp_gpu_mv(&best_mv, x->mv_col_min, x->mv_col_max,
                            x->mv_row_min , x->mv_row_max);

  bestsad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, best_mv);

  bestsad += mvsad_err_cost(&best_mv, &fcenter_mv, nmvsadcost_0, nmvsadcost_1,
                            sad_per_bit);
  br = best_mv.row;
  bc = best_mv.col;

  do {
    best_site = -1;
    if (gpu_check_bounds(x,br,bc,1 << pattern)) {
      for (i = 0; i < hex_num_candidates[pattern]; i++) {
        this_mv.row = br + hex_candidates[pattern][i].row;
        this_mv.col = bc + hex_candidates[pattern][i].col;

        thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);

        CHECK_BETTER
      }
    } else {
      for (i = 0; i < hex_num_candidates[pattern]; i++) {
        this_mv.row = br + hex_candidates[pattern][i].row;
        this_mv.col = bc + hex_candidates[pattern][i].col;

        if (!is_mv_in(x,&this_mv)) {
          continue;
        }

        thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
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

          thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
          CHECK_BETTER
        }
      } else {
        for (i = 0; i < PATTERN_CANDIDATES_REF; i++) {
          this_mv.row = br + hex_candidates[pattern][next_chkpts_indices[i]].row;
          this_mv.col = bc + hex_candidates[pattern][next_chkpts_indices[i]].col;

          if (!is_mv_in(x,&this_mv)) {
            continue;
          }

          thissad = get_sad(ref_frame, cur_frame, stride, intermediate_sad, this_mv);
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

MV check_better_subpel(__global uchar *ref_frame,
                            __global uchar *cur_frame,
                            __global int *nmvcost_0,
                            __global int *nmvcost_1,
                            __global int *nmvjointcost,
                            int stride,
                            unsigned int *v,
                            int r,
                            int c,
                            MV best_mv,
                            MV refmv,
                            MV minmv,
                            MV maxmv,
                            unsigned int *pbesterr,
                            int error_per_bit,
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
                            __local int *intermediate_int) {

#else
                            int *intermediate_int) {
#endif

  int sum, thismse;
  unsigned int sse;
  int distortion;

  if (c >= minmv.col && c <= maxmv.col && r >= minmv.row && r <= maxmv.row) {
    calculate_subpel_variance(ref_frame, cur_frame, stride,
       sp(c), sp(r), r, c, &sse, &sum);

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
    barrier(CLK_LOCAL_MEM_FENCE);
    intermediate_int[0] = 0;
    intermediate_int[1] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(intermediate_int, sum);
    atomic_add(intermediate_int + 1, sse);

    barrier(CLK_LOCAL_MEM_FENCE);
    sum = intermediate_int[0];
    sse = intermediate_int[1];
#endif

    thismse = sse - (((long int)sum * sum)
            / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));

    if ((*v = MVC(*v, r, c) + thismse) < *pbesterr) {
      *pbesterr = *v;
      best_mv.row = r;
      best_mv.col = c;
      distortion = thismse;
    }
  } else {
    *v = CL_INT_MAX;
  }
  return best_mv;
}


MV first_level_checks(__global uchar *ref_frame,
                      __global uchar *cur_frame,
                      __global int *nmvcost_0,
                      __global int *nmvcost_1,
                      __global int *nmvjointcost,
                      int stride,
                      int hstep,
                      MV best_mv,
                      MV refmv,
                      MV minmv,
                      MV maxmv,
                      unsigned int *pbesterr,
                      int error_per_bit,
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
                      __local int *intermediate_int) {
#else
                      int *intermediate_int) {
#endif


  unsigned int left, right, up, down, diag, whichdir;
  int distortion;
  int sum, thismse, tr, tc;
  unsigned int besterr, sse;

  tr = best_mv.row;
  tc = best_mv.col;

  besterr = *pbesterr;

  best_mv = check_better_subpel(ref_frame, cur_frame,
                              nmvcost_0, nmvcost_1, nmvjointcost, stride,
                              &left, tr, (tc - hstep),
                              best_mv, refmv, minmv, maxmv,
                              &besterr, error_per_bit,
                              intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                              nmvcost_0, nmvcost_1, nmvjointcost, stride,
                              &right, tr, (tc + hstep),
                              best_mv, refmv, minmv, maxmv,
                              &besterr, error_per_bit,
                              intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                              nmvcost_0, nmvcost_1, nmvjointcost, stride,
                              &up, (tr - hstep), tc,
                              best_mv, refmv, minmv, maxmv,
                              &besterr, error_per_bit,
                              intermediate_int);

  best_mv = check_better_subpel(ref_frame, cur_frame,
                              nmvcost_0, nmvcost_1, nmvjointcost, stride,
                              &down, (tr + hstep), tc,
                              best_mv, refmv, minmv, maxmv,
                              &besterr, error_per_bit,
                              intermediate_int);


  whichdir = (left < right ? 0 : 1) + (up < down ? 0 : 2);

  switch (whichdir) {
    case 0:
      tr = tr - hstep;
      tc = tc - hstep;
      break;
    case 1:
      tr = tr - hstep;
      tc = tc + hstep;
      break;
    case 2:
      tr = tr + hstep;
      tc = tc - hstep;
      break;
    case 3:
      tr = tr + hstep;
      tc = tc + hstep;
      break;
  }

  best_mv = check_better_subpel(ref_frame, cur_frame,
                              nmvcost_0, nmvcost_1, nmvjointcost, stride,
                              &diag, tr, tc,
                              best_mv, refmv, minmv, maxmv,
                              &besterr, error_per_bit,
                              intermediate_int);

  *pbesterr = besterr;
  return best_mv;
}

MV vp9_find_best_sub_pixel_tree(__global uchar *ref_frame,
                                __global uchar *cur_frame,
                                __global int *nmvcost_0,
                                __global int *nmvcost_1,
                                __global int *nmvjointcost,
                                int stride,
                                MV best_mv,
                                MV nearest_mv,
                                MV fcenter_mv,
                                INIT *x,
                                int error_per_bit,
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
                                __local int *intermediate_int) {
#else
                                int *intermediate_int) {
#endif
  int sum, thismse;
  int hstep;
  unsigned int sse, besterr;
  MV minmv,maxmv;

  hstep = 4;
  besterr = CL_INT_MAX;

  calculate_fullpel_variance(ref_frame, cur_frame, stride,
                    &sse, &sum, &best_mv);

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
   barrier(CLK_LOCAL_MEM_FENCE);
   intermediate_int[0] = 0;
   intermediate_int[1] = 0;

   barrier(CLK_LOCAL_MEM_FENCE);
   atomic_add(intermediate_int, sum);
   atomic_add(intermediate_int + 1, sse);
   barrier(CLK_LOCAL_MEM_FENCE);
   sum = intermediate_int[0];
   sse = intermediate_int[1];
#endif

  besterr = sse - (((long int)sum * sum)
                    / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS));

  besterr += mv_err_cost(&best_mv, &nearest_mv,
                         nmvcost_0, nmvcost_1, nmvjointcost,
                         error_per_bit);

  minmv.col = MAX(x->mv_col_min * 8, fcenter_mv.col - MV_MAX);
  maxmv.col = MIN(x->mv_col_max * 8, fcenter_mv.col + MV_MAX);
  minmv.row = MAX(x->mv_row_min * 8, fcenter_mv.row - MV_MAX);
  maxmv.row = MIN(x->mv_row_max * 8, fcenter_mv.row + MV_MAX);

  best_mv = first_level_checks(ref_frame, cur_frame,
                               nmvcost_0, nmvcost_1, nmvjointcost,
                               stride, hstep,
                               best_mv, nearest_mv, minmv, maxmv,
                               &besterr, error_per_bit,
                               intermediate_int);

  hstep >>= 1;
  best_mv = first_level_checks(ref_frame, cur_frame,
                               nmvcost_0, nmvcost_1, nmvjointcost,
                               stride, hstep,
                               best_mv, nearest_mv, minmv, maxmv,
                               &besterr, error_per_bit,
                               intermediate_int);

  return best_mv;
}


MV combined_motion_search(__global uchar *ref_frame,
                    __global uchar *cur_frame,
                    __global GPU_INPUT *input_mv,
                    __global GPU_RD_PARAMETERS *rd_parameters,
                    int stride,
                    MV pred_mv,
                    int mi_rows,
                    int mi_cols,
                    __local int *intermediate_int
                    ) {
  __global int   *nmvsadcost_0     = rd_parameters->nmvsadcost[0] + MV_MAX;
  __global int   *nmvsadcost_1     = rd_parameters->nmvsadcost[1] + MV_MAX;

  MV nearest_mv, near_mv;
  MV best_mv;
  MV fcenter_mv;
  int mi_row, mi_col, sad_per_bit;
  INIT x;

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
#else
  int local_col  = 0;
  int local_row  = 0;
#endif

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);

  sad_per_bit   = rd_parameters->sad_per_bit;

  mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  mi_col = global_col;

#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows,
                              mi_cols, (BLOCK_SIZE_IN_PIXELS >> 4));

  nearest_mv = input_mv->nearest_mv;
  vp9_set_mv_search_range_step2(&x, &nearest_mv);

  near_mv = input_mv->near_mv;

  fcenter_mv.row = nearest_mv.row >> 3;
  fcenter_mv.col = nearest_mv.col >> 3;

  best_mv = get_best_mv(ref_frame, cur_frame, stride, intermediate_int,
                        nearest_mv, near_mv, pred_mv);

  best_mv = full_pixel_search(ref_frame, cur_frame, stride,
                              intermediate_int, best_mv, fcenter_mv,
                              nmvsadcost_0, nmvsadcost_1,
                              &x, sad_per_bit);

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

inline void inter_prediction(__global uchar *ref_data,
                               __global uchar *cur_frame,
    int stride,
    int horz_subpel,
    int vert_subpel,
    int filter_type,
    __local uchar8 *intermediate,
    __global int *psum,
    __global uint *psse
) {
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

  *psum = 0;
  *psse = 0;
  barrier(CLK_LOCAL_MEM_FENCE);

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

      ref_data += stride;
      cur_frame += stride;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  } else if(!horz_subpel) {
    /* L0 only y_frac */
    char8 filt = filter[filter_type][vert_subpel];
    ref_data -= (3 * stride);
    for(i = 0; i < PIXEL_ROWS_PER_WORKITEM; i++) {

      inter = (short8)(-1 << 14);
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s0;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s1;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s2;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s3;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s4;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s5;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
      ref_u8 = vload8(0, ref_data);
      tmp = filt.s6;
      inter += convert_short8(ref_u8) * tmp;

      ref_data += stride;
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

      ref_data  -= 6 * stride;
      cur_frame += stride;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  } else {
    char8 filt = filter[filter_type][horz_subpel];
    ref_data -= (3 * stride);

    barrier(CLK_LOCAL_MEM_FENCE);

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

      ref_data += stride;
      inter_offset += LOCAL_STRIDE;
    }

    if (local_row < 8 / PIXEL_ROWS_PER_WORKITEM) {
      ref_data += (BLOCK_SIZE_IN_PIXELS - PIXEL_ROWS_PER_WORKITEM) * stride;
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

        ref_data += stride;
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
      cur_frame += stride;
    }
    ACCUMULATE_SUM_SSE_INTER_PRED(sum, sse)
  }
}


__kernel
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
#endif
void vp9_full_pixel_search(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    int mi_rows,
    int mi_cols,
    __global GPU_OUTPUT *pred_mv
) {
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  __local int intermediate_int[1];
#else
  __local int *intermediate_int;
#endif
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);

  int global_offset = ((global_row * PIXEL_ROWS_PER_WORKITEM) * stride) + (global_col * NUM_PIXELS_PER_WORKITEM);

  int group_col    = get_group_id(0);
  int group_row    = get_group_id(1);
  int group_stride = get_num_groups(0);

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
#else
  int local_col  = 0;
  int local_row  = 0;
#endif

  int mi_row, mi_col;
  MV best_mv,nearest_mv;

  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  mv_input            += (group_row * group_stride + group_col);
  sse_variance_output += (group_row * group_stride + group_col);
#else
  mv_input            += (global_row * global_stride + global_col);
  sse_variance_output += (global_row * global_stride + global_col);
#endif

#if BLOCK_SIZE_IN_PIXELS == 16
  pred_mv += (group_row/2 * group_stride/2 + group_col/2);
#elif BLOCK_SIZE_IN_PIXELS == 8
  pred_mv += (global_row/2 * global_stride/2 + global_col/2);
#endif

  cur_frame += global_offset;

  ref_frame += global_offset;

  if(!mv_input->do_compute)
    goto exit;

  mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  mi_col = global_col;

#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  int mi_step = ((BLOCK_SIZE_IN_PIXELS / 8) / 2);
  if (mi_col + mi_step >= mi_cols) {
    goto exit;
  }
  if (mi_row + mi_step >= mi_rows) {
    goto exit;
  }

  best_mv = combined_motion_search(ref_frame,
                  cur_frame,
                  mv_input,
                  rd_parameters,
                  stride,
                  pred_mv->mv,
                  mi_rows,
                  mi_cols,
                  intermediate_int);
  sse_variance_output->mv = best_mv;
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
    int mi_cols,
    __global GPU_OUTPUT *pred_mv
) {
  __global int   *nmvcost_0        = rd_parameters->mvcost[0] + MV_MAX;
  __global int   *nmvcost_1        = rd_parameters->mvcost[1] + MV_MAX;
  __global int   *nmvjointcost     = rd_parameters->nmvjointcost;

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  int global_offset = (global_row * stride * BLOCK_SIZE_IN_PIXELS) +
                      (global_col * BLOCK_SIZE_IN_PIXELS);

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int mi_row, mi_col;
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
  MV  best_mv, nearest_mv;
  int8 c_squared;
  short8 c;

  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

#if BLOCK_SIZE_IN_PIXELS == 32
  int bsize = BLOCK_32X32;
#elif BLOCK_SIZE_IN_PIXELS == 16
  int bsize = BLOCK_16X16;
#elif BLOCK_SIZE_IN_PIXELS == 8
  int bsize = BLOCK_8X8;
#endif

  mv_input            += (global_row * global_stride + global_col);
  sse_variance_output += (global_row * global_stride + global_col);
  pred_mv             += (global_row/2 * global_stride/2 + global_col/2);

  cur_frame += global_offset;
  uchar8 curr_data;
  uchar8 pred_data;
  PREDICTION_MODE best_mode = ZEROMV;
  INTERP_FILTER newmv_filter, best_pred_filter = EIGHTTAP;

  ref_frame += global_offset;

  GPU_OUTPUT_STAGE1 out_mv;

#if BLOCK_SIZE_IN_PIXELS != 32
  out_mv.mv = pred_mv->mv;
#else
  out_mv.mv.row = 0;
  out_mv.mv.col = 0;
#endif

  if(!mv_input->do_compute)
    goto exit;

  mi_row = global_row * (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM);
  mi_col = global_col * (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM);
#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  int mi_step = ((BLOCK_SIZE_IN_PIXELS / 8) / 2);
  if (mi_col + mi_step >= mi_cols) {
    goto exit;
  }
  if (mi_row + mi_step >= mi_rows) {
    goto exit;
  }

  best_mv = sse_variance_output->mv;

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

    if (this_rd < (int64_t)(1 << num_pels_log2_lookup[bsize]))
    {
      mv_input->do_newmv = 0;
      goto exit;
    }
    int mode_rd_thresh =
        rd_parameters->threshes[mode_idx[0][INTER_OFFSET(NEWMV)]];
    if (rd_less_than_thresh(best_rd, mode_rd_thresh,
            rd_parameters->thresh_fact_newmv))
    {
      mv_input->do_newmv = 0;
      goto exit;
    }
  }

  nearest_mv = mv_input->nearest_mv;

  out_mv.rate_mv = vp9_mv_bit_cost(&best_mv, &nearest_mv,
                                       nmvjointcost, nmvcost_0, nmvcost_1,
                                       MV_COST_WEIGHT);

  int rate_mode = rd_parameters->inter_mode_cost[mv_input->mode_context]
                                                [GPU_INTER_OFFSET(NEWMV)];
  mv_input->do_newmv = !(RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
                         (out_mv.rate_mv + rate_mode), 0) > best_rd);

  if(mv_input->do_newmv)
    out_mv.mv = best_mv;

  sse_variance_output->rate_mv = out_mv.rate_mv;

exit:
  sse_variance_output->mv = out_mv.mv;
  return;
}

__kernel
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
#endif
void vp9_sub_pixel_search(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    int mi_rows,
    int mi_cols,
    __global GPU_OUTPUT *pred_mv
) {
#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  __local int intermediate_int[2];
#else
  int intermediate_int[2];
#endif

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  int local_col  = get_local_id(0);
  int local_row  = get_local_id(1);
#else
  int local_col  = 0;
  int local_row  = 0;
#endif

  int mi_row, mi_col;

  int global_offset = (global_row * PIXEL_ROWS_PER_WORKITEM * stride) +
                      (global_col * NUM_PIXELS_PER_WORKITEM);

#if BLOCK_SIZE_IN_PIXELS > PIXEL_ROWS_PER_WORKITEM || BLOCK_SIZE_IN_PIXELS > NUM_PIXELS_PER_WORKITEM
  mv_input            += (group_row * group_stride + group_col);
  sse_variance_output += (group_row * group_stride + group_col);
#else
  mv_input            += (global_row * global_stride + global_col);
  sse_variance_output += (global_row * global_stride + global_col);
#endif

  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  cur_frame += global_offset;
  ref_frame += global_offset;

  if(!mv_input->do_compute)
    goto exit;

  mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  mi_col = global_col;
#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  int mi_step = ((BLOCK_SIZE_IN_PIXELS / 8) / 2);
  if (mi_col + mi_step >= mi_cols) {
    goto exit;
  }
  if (mi_row + mi_step >= mi_rows) {
    goto exit;
  }

  if (mv_input->do_newmv)
  {
    MV best_mv = sse_variance_output->mv;
    MV nearest_mv = mv_input->nearest_mv;
    MV fcenter_mv;
    INIT x;
    int error_per_bit = rd_parameters->error_per_bit;
    __global int   *nmvcost_0        = rd_parameters->mvcost[0] + MV_MAX;
    __global int   *nmvcost_1        = rd_parameters->mvcost[1] + MV_MAX;
    __global int   *nmvjointcost     = rd_parameters->nmvjointcost;

    fcenter_mv.row = nearest_mv.row >> 3;
    fcenter_mv.col = nearest_mv.col >> 3;

    vp9_gpu_set_mv_search_range(&x, mi_row, mi_col, mi_rows,
                                mi_cols, (BLOCK_SIZE_IN_PIXELS >> 4));

    sse_variance_output->mv  = vp9_find_best_sub_pixel_tree(ref_frame, cur_frame,
                                             nmvcost_0, nmvcost_1, nmvjointcost,
                                             stride,
                                             best_mv, nearest_mv, fcenter_mv,
                                             &x, error_per_bit,
                                             intermediate_int);
  }
exit:
  return;
}

__kernel
__attribute__((reqd_work_group_size(BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM,
                                    BLOCK_SIZE_IN_PIXELS / PIXEL_ROWS_PER_WORKITEM,
                                    1)))
void vp9_inter_prediction_and_sse(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    int mi_rows,
    int mi_cols,
    __global GPU_OUTPUT *pred_mv
) {

  __local uchar8 intermediate_uchar8[(BLOCK_SIZE_IN_PIXELS*(BLOCK_SIZE_IN_PIXELS + 8))/NUM_PIXELS_PER_WORKITEM];
  __local int *intermediate_int = (__local int *)intermediate_uchar8;

  int global_row = get_global_id(1);

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0) / 2;

  int mi_row, mi_col;
  int sum;
  uint sse, variance;


  int is_fourth_group = 0;
  if(group_col % 4 == 3) {
    group_col -= 2;
    is_fourth_group = 1;
  }

  int group_offset = group_row * group_stride + (group_col / 2);
  mv_input      += group_offset;

  int filter_type;

  if (group_col % 2 == 0) {
    filter_type = EIGHTTAP;
  } else {
    if(is_fourth_group == 0)
      filter_type = EIGHTTAP_SMOOTH;
    else
      filter_type = EIGHTTAP_SHARP;

    if(mv_input->filter_type != SWITCHABLE) {
        group_col += 2;
        mv_input++;
    }
  }

  if(!mv_input->do_compute)
    goto exit;

  if(!mv_input->do_newmv)
    goto exit;

  int local_col = get_local_id(0);
  int global_offset = (global_row * stride * PIXEL_ROWS_PER_WORKITEM) +
      ((group_col / 2) * BLOCK_SIZE_IN_PIXELS) + (local_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  group_offset = group_row * group_stride + (group_col / 2);
  sse_variance_output += group_offset;

  cur_frame += global_offset;
  uchar8 pred_data;
  PREDICTION_MODE best_mode = ZEROMV;
  INTERP_FILTER newmv_filter, best_pred_filter = EIGHTTAP;

  GPU_OUTPUT_STAGE1 out_mv;


  mi_row = (global_row * PIXEL_ROWS_PER_WORKITEM) / MI_SIZE;
  mi_col = (group_col / 2) * (BLOCK_SIZE_IN_PIXELS / MI_SIZE);
#if BLOCK_SIZE_IN_PIXELS == 32
  mi_row = (mi_row >> 2) << 2;
  mi_col = (mi_col >> 2) << 2;
#elif BLOCK_SIZE_IN_PIXELS == 16
  mi_row = (mi_row >> 1) << 1;
  mi_col = (mi_col >> 1) << 1;
#endif

  int mi_step = ((BLOCK_SIZE_IN_PIXELS / 8) / 2);
  if (mi_col + mi_step >= mi_cols) {
    goto exit;
  }
  if (mi_row + mi_step >= mi_rows) {
    goto exit;
  }

  out_mv.mv = sse_variance_output->mv;
  int mv_row = out_mv.mv.row;
  int mv_col = out_mv.mv.col;
  int mv_offset = ((mv_row >> SUBPEL_BITS) * stride) + (mv_col >> SUBPEL_BITS);
  int horz_subpel = (mv_col & SUBPEL_MASK) << 1;
  int vert_subpel = (mv_row & SUBPEL_MASK) << 1;

  ref_frame += global_offset + mv_offset;

  if(filter_type != EIGHTTAP && !horz_subpel && !vert_subpel)
    goto exit;

  inter_prediction(ref_frame, cur_frame, stride, horz_subpel, vert_subpel,
                          filter_type, intermediate_uchar8,
                          &sse_variance_output->sum[filter_type],
                          &sse_variance_output->sse[filter_type]);

exit:
  return;
}

__kernel
void vp9_rd_calculation(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_PARAMETERS *rd_parameters,
    int mi_rows,
    int mi_cols,
    __global GPU_OUTPUT *pred_mv
) {


  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);
  uchar8 curr_data;

  int mi_row, mi_col;
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
#elif BLOCK_SIZE_IN_PIXELS == 8
  int bsize = BLOCK_8X8;
#endif

  sse_variance_output += (global_row * global_stride + global_col);

  PREDICTION_MODE best_mode = ZEROMV;
  INTERP_FILTER newmv_filter, best_pred_filter = EIGHTTAP;

  GPU_OUTPUT_STAGE1 out_mv;

  if(!mv_input->do_compute)
    goto exit;

  if(!mv_input->do_newmv)
    goto exit;

  mi_row = global_row * (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM);
  mi_col = global_col * (BLOCK_SIZE_IN_PIXELS / NUM_PIXELS_PER_WORKITEM);

  int mi_step = ((BLOCK_SIZE_IN_PIXELS / 8) / 2);
  if (mi_col + mi_step >= mi_cols) {
    goto exit;
  }
  if (mi_row + mi_step >= mi_rows) {
    goto exit;
  }

  // ZEROMV not required for BLOCK_32X32
  if (BLOCK_SIZE_IN_PIXELS != 32) {
    best_rd = sse_variance_output->best_rd;
  }
  out_mv.mv = sse_variance_output->mv;
  int mv_row = out_mv.mv.row;
  int mv_col = out_mv.mv.col;
  int horz_subpel = (mv_col & SUBPEL_MASK) << 1;
  int vert_subpel = (mv_row & SUBPEL_MASK) << 1;

  int rate_mv = sse_variance_output->rate_mv;

  int64_t cost, best_cost = INT64_MAX;
  int filter_type, end_filter_type;
  if(mv_input->filter_type == SWITCHABLE && (horz_subpel || vert_subpel))
    end_filter_type = EIGHTTAP_SHARP;
  else
    end_filter_type = EIGHTTAP;

  for (filter_type = EIGHTTAP; filter_type <= end_filter_type; ++filter_type) {
    sum = sse_variance_output->sum[filter_type];
    sse = sse_variance_output->sse[filter_type];
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

__kernel
void vp9_is_8x8_required(
    __global GPU_INPUT  *mv_32x32,
    __global GPU_OUTPUT *rd_32x32,
    __global GPU_INPUT  *mv_16x16,
    __global GPU_OUTPUT *rd_16x16,
    __global GPU_INPUT  *mv_8x8,
    __global GPU_OUTPUT *rd_8x8,
    __global GPU_RD_PARAMETERS *rd_parameters)
{
  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_stride = get_global_size(0);

  mv_32x32 += global_row * global_stride + global_col;
  rd_32x32 += global_row * global_stride + global_col;

  global_row *= 2;
  global_col *= 2;
  global_stride *= 2;
  mv_16x16 += global_row * global_stride + global_col;
  rd_16x16 += global_row * global_stride + global_col;
  if(mv_32x32->do_compute == 0 || mv_16x16->do_compute == 0)
    return;


  int total_rate_16x16 = rd_16x16[0].returnrate +
                         rd_16x16[1].returnrate +
                         rd_16x16[global_stride].returnrate +
                         rd_16x16[global_stride + 1].returnrate;
  int64_t total_dist_16x16 = rd_16x16[0].returndistortion +
                             rd_16x16[1].returndistortion +
                             rd_16x16[global_stride].returndistortion +
                             rd_16x16[global_stride + 1].returndistortion;


  int64_t total_rd_16x16 = RDCOST(rd_parameters->rd_mult, rd_parameters->rd_div,
                          total_rate_16x16, total_dist_16x16);

  int64_t total_rd_16x16_minus_12_point_5_percent = total_rd_16x16 -
      (total_rd_16x16 / 8);

  if(rd_32x32[0].best_rd < total_rd_16x16_minus_12_point_5_percent)
  {
    global_row *= 2;
    global_col *= 2;
    global_stride *= 2;
    rd_8x8 += global_row * global_stride + global_col;
    mv_8x8 += global_row * global_stride + global_col;
    int i;
    for(i = 0; i < 4; i++)
    {
      int j;
      for(j = 0; j < 4; j++)
      {
        mv_8x8[j].do_compute = 0;
        rd_8x8[j].best_mode = ZEROMV;
        rd_8x8[j].best_rd = INT64_MAX;
      }
      mv_8x8 += global_stride;
      rd_8x8 += global_stride;
    }
  }
}
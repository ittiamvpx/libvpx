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
    
#define INTER_MODES (1 + NEWMV - NEARESTMV)                  
#define INTER_OFFSET(mode) ((mode) - NEARESTMV)
#define MIN(x, y) (((x) < (y)) ? (x) : (y))
    
#define INLINE inline;

typedef short int16_t;
typedef long  int64_t;
typedef uint  uint32_t;
typedef ulong uint64_t;

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
    
typedef struct mv {
  int16_t row;
  int16_t col;
} MV;


typedef struct GPU_INPUT {
  MV mv;
  INTERP_FILTER filter_type;
  int mode_context;
  int rate_mv;  
    int do_newmv;
} GPU_INPUT;

typedef struct GPU_OUTPUT {
  int          returnrate;
  int64_t      returndistortion;
  int64_t      best_rd;
  PREDICTION_MODE best_mode;
  INTERP_FILTER best_pred_filter;
  int skip_txfm;
  TX_SIZE tx_size;
} GPU_OUTPUT;

typedef struct GPU_RD_CONSTANTS {
  int rd_mult;
  int rd_div;
  int switchable_interp_costs[SWITCHABLE_FILTERS];
  unsigned int inter_mode_cost[INTER_MODE_CONTEXTS][GPU_INTER_MODES];  
  int threshes[MAX_MODES];
    int thresh_fact_newmv;
  TX_MODE tx_mode;    
  int dc_quant;
  int ac_quant;
} GPU_RD_CONSTANTS;


uchar8 motion_comp(__global uchar *ref_data,
    int stride,
    int horz_subpel,
    int vert_subpel,
    int filter_type,
    __local uchar8 *intermediate);

#define SQUARE(a) ((a) * (a))

#define ACCUMULATE(block_size_in_pixels)            \
  if(block_size_in_pixels >= 32) {                  \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if(local_col < 2)                               \
      intermediate_int[(local_row * local_stride) + local_col] += intermediate_int[(local_row * local_stride) + local_col + 2]; \
  }                                                 \
  if(block_size_in_pixels >= 16) {                  \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if(local_col < 1)                               \
      intermediate_int[(local_row * local_stride) + local_col] += intermediate_int[(local_row * local_stride) + local_col + 1]; \
  }                                                 \
  if(block_size_in_pixels >= 32) {                  \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if(local_row < 16)                              \
      intermediate_int[(local_row * local_stride)] += intermediate_int[(local_row + 16) * local_stride]; \
  }                                                 \
  if(block_size_in_pixels >= 16) {                  \
    barrier(CLK_LOCAL_MEM_FENCE);                   \
    if(local_row < 8)                               \
      intermediate_int[(local_row * local_stride)] += intermediate_int[(local_row + 8) * local_stride]; \
  }                                                 \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if(local_row < 4)                                 \
    intermediate_int[(local_row * local_stride)] += intermediate_int[(local_row + 4) * local_stride]; \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if(local_row < 2)                                 \
    intermediate_int[(local_row * local_stride)] += intermediate_int[(local_row + 2) * local_stride]; \
  barrier(CLK_LOCAL_MEM_FENCE);                     \
  if(local_row < 1)                                 \
    intermediate_int[(local_row * local_stride)] += intermediate_int[(local_row + 1) * local_stride]; 
    
#define CALCULATE_SSE_VAR(a, b)                                     \
    sum = a.s0 - b.s0 + a.s1 - b.s1 + a.s2 - b.s2 + a.s3 - b.s3 +   \
          a.s4 - b.s4 + a.s5 - b.s5 + a.s6 - b.s6 + a.s7 - b.s7;    \
    barrier(CLK_LOCAL_MEM_FENCE);                                   \
    intermediate_int[(local_row * local_stride) + local_col] = sum; \
    ACCUMULATE(BLOCK_SIZE_IN_PIXELS)                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                   \
    sum = intermediate_int[0];                                      \
    sse = SQUARE(a.s0 - b.s0) + SQUARE(a.s1 - b.s1) +               \
          SQUARE(a.s2 - b.s2) + SQUARE(a.s3 - b.s3) +               \
          SQUARE(a.s4 - b.s4) + SQUARE(a.s5 - b.s5) +               \
          SQUARE(a.s6 - b.s6) + SQUARE(a.s7 - b.s7);                \
    barrier(CLK_LOCAL_MEM_FENCE);                                   \
    intermediate_int[(local_row * local_stride) + local_col] = sse; \
    ACCUMULATE(BLOCK_SIZE_IN_PIXELS)                                \
    barrier(CLK_LOCAL_MEM_FENCE);                                   \
    sse = intermediate_int[0];                                      \
    variance = sse - ((long)sum * sum) / (BLOCK_SIZE_IN_PIXELS * BLOCK_SIZE_IN_PIXELS);
    
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
  
INLINE int get_msb(unsigned int n) {
  return 31 ^ clz(n);
}      

INLINE int rd_less_than_thresh(int64_t best_rd, int thresh, int thresh_fact) {
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

// TODO(karthick) : Move the ac_quant and dc_quant to the RD structure              
__kernel
void inter_prediction_and_rd_calc(__global uchar *ref_frame,
    __global uchar *cur_frame,
    int stride,
    __global GPU_INPUT *mv_input,
    __global GPU_OUTPUT *sse_variance_output,
    __global GPU_RD_CONSTANTS *rd_constants 
) {

  __global uchar *ref_data;
  __local uchar8 intermediate_uchar8[(BLOCK_SIZE_IN_PIXELS*(BLOCK_SIZE_IN_PIXELS + 7))/NUM_PIXELS_PER_WORKITEM];
  __local int *intermediate_int = (__local int *)intermediate_uchar8;

  int global_col = get_global_id(0);
  int global_row = get_global_id(1);
  int global_offset = (global_row * stride) + (global_col * NUM_PIXELS_PER_WORKITEM);
  global_offset += (VP9_ENC_BORDER_IN_PIXELS * stride) + VP9_ENC_BORDER_IN_PIXELS;

  int group_col = get_group_id(0);
  int group_row = get_group_id(1);
  int group_stride = get_num_groups(0);
  mv_input += (group_row * group_stride + group_col);
  int mv_row = mv_input->mv.row;
  int mv_col = mv_input->mv.col;
  int mv_offset = ((mv_row >> SUBPEL_BITS) * stride) + (mv_col >> SUBPEL_BITS);

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int local_stride = get_local_size(0);

  int horz_subpel = (mv_col & SUBPEL_MASK) << 1;
  int vert_subpel = (mv_row & SUBPEL_MASK) << 1;
  int sum;
  uint sse, variance;
  int rate, actual_rate, newmv_rate, best_rate = INT32_MAX;
  int64_t dist, actual_dist, newmv_dist, best_dist = INT64_MAX;
  int64_t this_rd, best_rd = INT64_MAX;
  int newmv_skip_txfm = 0, best_skip_txfm = 0, skip_txfm;
  TX_SIZE newmv_tx_size, best_tx_size, tx_size;
  TX_MODE tx_mode = rd_constants->tx_mode;
  int dc_quant = rd_constants->dc_quant;
  int ac_quant = rd_constants->ac_quant;

#if BLOCK_SIZE_IN_PIXELS == 32    
  int bsize = BLOCK_32X32;
#elif BLOCK_SIZE_IN_PIXELS == 16    
  int bsize = BLOCK_16X16;
#elif BLOCK_SIZE_IN_PIXELS == 8
  int bsize = BLOCK_8X8;
#endif
  best_tx_size = MIN(max_txsize_lookup[bsize],
      tx_mode_to_biggest_tx_size[tx_mode]);

  sse_variance_output += (group_row * group_stride + group_col);

  cur_frame += global_offset;
  uchar8 curr_data = vload8(0, cur_frame);
  uchar8 pred_data;
  PREDICTION_MODE best_mode = ZEROMV;
  INTERP_FILTER newmv_filter, best_pred_filter = EIGHTTAP;

  ref_data = ref_frame + global_offset;
  
  // ZEROMV not required for BLOCK_32X32
  if (BLOCK_SIZE_IN_PIXELS != 32) {
    pred_data = vload8(0, ref_data);
    CALCULATE_SSE_VAR(curr_data, pred_data)
    CALCULATE_RATE_DIST
    actual_rate += rd_constants->inter_mode_cost[mv_input->mode_context][0];
    this_rd = RDCOST(rd_constants->rd_mult, rd_constants->rd_div,
        actual_rate, actual_dist);
    if (this_rd < best_rd) {
      best_rd = this_rd;
      best_rate = actual_rate;
      best_dist = actual_dist;
      best_skip_txfm = skip_txfm;
      best_tx_size = tx_size;
    }
    int mode_rd_thresh =
    rd_constants->threshes[mode_idx[0][INTER_OFFSET(NEWMV)]];
    if (rd_less_than_thresh(best_rd, mode_rd_thresh,
            rd_constants->thresh_fact_newmv))
      goto exit;
    if (!mv_input->do_newmv)
      goto exit;
  }

  ref_data = ref_frame + global_offset + mv_offset;

  int64_t cost, best_cost = INT64_MAX;
  int filter_type, end_filter_type;
  if(mv_input->filter_type == SWITCHABLE)
    end_filter_type = EIGHTTAP_SHARP;
  else
    end_filter_type = EIGHTTAP;
  
  for (filter_type = EIGHTTAP; filter_type <= end_filter_type; ++filter_type) {
    pred_data = motion_comp(ref_data, stride, horz_subpel, vert_subpel,
                            filter_type, intermediate_uchar8);

    CALCULATE_SSE_VAR(curr_data, pred_data)
    CALCULATE_RATE_DIST
    cost = RDCOST(rd_constants->rd_mult, rd_constants->rd_div,
      rd_constants->switchable_interp_costs[filter_type] + actual_rate, actual_dist);
    if (cost < best_cost) {
      best_cost = cost;
      newmv_rate = actual_rate;
      newmv_dist = actual_dist;
      newmv_filter = filter_type;
      newmv_skip_txfm = skip_txfm;
      newmv_tx_size = tx_size;
    }
  }
  newmv_rate += mv_input->rate_mv;
  newmv_rate += rd_constants->inter_mode_cost[mv_input->mode_context][1];
  this_rd = RDCOST(rd_constants->rd_mult, rd_constants->rd_div,
      newmv_rate, newmv_dist);
  if (this_rd < best_rd) {
    best_rd = this_rd;
    best_rate = newmv_rate;
    best_dist = newmv_dist;
    best_mode = NEWMV;
    best_pred_filter = newmv_filter;
    best_skip_txfm = newmv_skip_txfm;
    best_tx_size = newmv_tx_size;
  }
exit:
  sse_variance_output->returnrate = best_rate;
  sse_variance_output->returndistortion = best_dist;
  sse_variance_output->best_rd = best_rd;
  sse_variance_output->best_mode = best_mode;
  sse_variance_output->best_pred_filter = best_pred_filter;
  sse_variance_output->skip_txfm = best_skip_txfm;
  sse_variance_output->tx_size = best_tx_size;
}



uchar8 motion_comp(__global uchar *ref_data,
    int stride,
    int horz_subpel,
    int vert_subpel,
    int filter_type,
    __local uchar8 *intermediate
) {
  __global uchar8 *out_data_vec8;
  __local uchar8 *intermediate_uchar8;

  uchar16 ref;
  uchar8 ref_u8;

  short8 inter, inter1;
  int4 inter_out1;

  int local_col = get_local_id(0);
  int local_row = get_local_id(1);
  int local_stride = get_local_size(0);
  int local_height = get_local_size(1);
  int inter_offset = (local_row * local_stride) + local_col;

  short8 tmp;
  int4 shift_val=(int4)(1<<14);
  int4 tmp1;
  uchar8 temp_out;
  uchar8 out_uni;
  uchar8 out_bi;

  if (!vert_subpel) {
    /* L0 only x_frac */
    char8 filt = filter[filter_type][horz_subpel];

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
      tmp1 = (1<<6)+shift_val;
      inter_out1 = convert_int4(inter.s0123);

      out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
      inter_out1 =convert_int4(inter.s4567);

      out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
    }
  } else if(!horz_subpel) {
    /* L0 only y_frac */
    char8 filt = filter[filter_type][vert_subpel];
    inter = (short8)(-1 << 14);
    ref_data -= (3*stride);
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
  } else {
    char8 filt = filter[filter_type][horz_subpel];
    ref_data -= (3 * stride);
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

    if (local_row < 7) {
      ref_data += (BLOCK_SIZE_IN_PIXELS * stride);

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
      intermediate[inter_offset + local_stride * local_height] = temp_out;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    intermediate_uchar8 = intermediate + inter_offset + (3 * local_stride);
    filt = filter[filter_type][vert_subpel];
    inter = (short8)(-1 << 14);
    ref_u8 = intermediate_uchar8[-3 * local_stride];
    tmp = filt.s0;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[-2 * local_stride];
    tmp = filt.s1;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[-1 * local_stride];
    tmp = filt.s2;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[0 * local_stride];
    tmp = filt.s3;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[1 * local_stride];
    tmp = filt.s4;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[2 * local_stride];
    tmp = filt.s5;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[3 * local_stride];
    tmp = filt.s6;
    inter += convert_short8(ref_u8) * tmp;
    ref_u8 = intermediate_uchar8[4 * local_stride];
    tmp = filt.s7;
    inter += convert_short8(ref_u8) * tmp;

    tmp1 = (1 << 6) + shift_val;
    inter_out1 = convert_int4(inter.s0123);

    out_uni.s0123 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
    inter_out1 = convert_int4(inter.s4567);

    out_uni.s4567 = convert_uchar4_sat((inter_out1 + tmp1) >> 7);
  }
  
  return out_uni;
}

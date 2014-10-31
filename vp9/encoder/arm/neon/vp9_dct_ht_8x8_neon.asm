;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

;TODO(cd): adjust these constant to be able to use vqdmulh for faster
;          dct_const_round_shift(a * b) within butterfly calculations.
cospi_1_64  EQU 16364
cospi_2_64  EQU 16305
cospi_3_64  EQU 16207
cospi_4_64  EQU 16069
cospi_5_64  EQU 15893
cospi_6_64  EQU 15679
cospi_7_64  EQU 15426
cospi_8_64  EQU 15137
cospi_9_64  EQU 14811
cospi_10_64 EQU 14449
cospi_11_64 EQU 14053
cospi_12_64 EQU 13623
cospi_13_64 EQU 13160
cospi_14_64 EQU 12665
cospi_15_64 EQU 12140
cospi_16_64 EQU 11585
cospi_17_64 EQU 11003
cospi_18_64 EQU 10394
cospi_19_64 EQU  9760
cospi_20_64 EQU  9102
cospi_21_64 EQU  8423
cospi_22_64 EQU  7723
cospi_23_64 EQU  7005
cospi_24_64 EQU  6270
cospi_25_64 EQU  5520
cospi_26_64 EQU  4756
cospi_27_64 EQU  3981
cospi_28_64 EQU  3196
cospi_29_64 EQU  2404
cospi_30_64 EQU  1606
cospi_31_64 EQU   804

    ; Do not export DCT8x8 use intrinsics
    ;EXPORT |vp9_fdct8x8_neon|
    EXPORT  |vp9_fht8x8_neon|

    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

    AREA     Block, CODE, READONLY

    ; --------------------------------------------------------------------------
    ; Load the named constant
    ; Load the specified constant as 16 bit signed value into neon register
    ;  $dst     : neon register to load constant into
    ;  $const   : name of the constant to load
    ;  $tmp     : arm register to use as temporary register
    MACRO
    GET_CONST       $dst,   $const,     $tmp
    ; Generate the scalar constant
    mov             $tmp,   #$const  &  0xFF00
    add             $tmp,   #$const  &  0x00FF
    ; Copy the constant to neon register
    vdup.s16        $dst,   $tmp

    MEND

    ;---------------------------------------------------------------------------
    ; Computes butterfly. Cosine/Sine coefficient multiplications are not needed
    ; Computes butterfly of two registers
    ;  $reg1 : Source register for Butterfly
    ;  $reg2 : Source & destination register for butterfly
    ;  $reg3 : destination register for butterfly
    ;
    ; $reg3 = $reg1 - $reg2
    ; $reg2 = $reg1 + $reg2
    MACRO
    DO_BUTTERFLY_NO_COEFFS  $reg1, $reg2, $reg3

    vsub.s16        $reg3,  $reg1, $reg2
    vadd.s16        $reg2,  $reg1, $reg2

    MEND

    ;---------------------------------------------------------------------------
    ; Computes butterfly.
    ; Cosine/Sine coefficients are not needed
    ; Computes butterfly of two registers
    ;  $src1 : Source register for Butterfly
    ;  $src2 : Source register for butterfly
    ;  $dst1 : destination register for butterfly
    ;  $dst2 : destination register for butterfly
    MACRO
    DO_BUTTERFLY_NO_COEFFS_DST        $src1, $src2, $src3, $src4, $dst1, $dst2, $dst3, $dst4, $qdst4

    ; Compute butterfly for $src1, $src3
    vadd.s32        $qdst4,  $src1,   $src3
    vsub.s32        $src1,   $src1,   $src3

    vrshrn.s32      $dst1,   $qdst4,  #14
    vrshrn.s32      $dst3,   $src1,   #14

    ; Compute butterfly for $src2, $src4
    vadd.s32        $src1,   $src2,   $src4
    vsub.s32        $src3,   $src2,   $src4

    vrshrn.s32      $dst2,   $src1,   #14
    vrshrn.s32      $dst4,   $src3,   #14

    MEND

    ;---------------------------------------------------------------------------
    ; Computes butterfly. Cosine/Sine coefficients are same
    ; Computes butterfly of 4 registers
    ;  $src1 : Source register for butterfly (64 bit)
    ;  $src2 : Source register for butterfly (64 bit)
    ;  $src3 : Source & destination register for Butterfly (64 bit)
    ;  $src4 : Source & destination register for butterfly (64 bit)
    ;  $scal : Scaling value                 (64 bit)
    ;  $dst1 : Source register for Butterfly (64 bit)
    ;  $dst2 : Source register for butterfly (64 bit)
    ;  $tmp2,$tmp3 : temporary registers     (128 bit)
    ;
    ; Ssrc3 = ($src1 + $src2) * scal >> 14 (rounded)
    ; Ssrc4 = ($src3 + $src4) * scal >> 14 (rounded)
    ; Sdst1 = ($src1 - $src2) * scal >> 14 (rounded)
    ; Sdst2 = ($src3 - $src4) * scal >> 14 (rounded)
    MACRO
    DO_BUTTERFLY_SYMMETRIC_COEFFS   $src1,  $src2,  $src3,  $src4,  $scal,  $dst1,  $dst2,  $tmp2,  $tmp3

    ; Butterfly for $src1 $src3
    vaddl.s16       $tmp3,  $src1,  $src3
    vsubl.s16       $tmp2,  $src1,  $src3

    vmul.s32        $tmp3,  $tmp3,  $scal
    vmul.s32        $tmp2,  $tmp2,  $scal

    vrshrn.s32      $src3,  $tmp3,  #14
    vrshrn.s32      $dst1,  $tmp2,  #14

    ; Butterfly for $src2 $src4
    vaddl.s16       $tmp3,  $src2,  $src4
    vsubl.s16       $tmp2,  $src2,  $src4

    vmul.s32        $tmp3,  $tmp3,  $scal
    vmul.s32        $tmp2,  $tmp2,  $scal

    vrshrn.s32      $src4,  $tmp3,  #14
    vrshrn.s32      $dst2,  $tmp2,  #14

    MEND

    ;---------------------------------------------------------------------------
    ; Computes butterfly. Cosine/Sine coefficients are unique
    ; Computes butterfly of 4 registers
    ;  $reg1   : Source and destination for Butterfly (64 bit)
    ;  $reg2   : Source and destination for butterfly (64 bit)
    ;  $reg3   : Source and destination for Butterfly (64 bit)
    ;  $reg4   : Source and destination for butterfly (64 bit)
    ;  $scal1  : Scaling value 1                      (64 bit)
    ;  $scal2  : Scaling value 2                      (64 bit)
    ;  $tmp1, $tmp2, $tmp3 : temporary registers      (128 bit)
    ;
    ; Sreg1 = ( $reg1 * $scal1 + $reg2 * $scal2) >> 14 (rounded)
    ; Sreg2 = ( $reg3 * $scal1 + $reg4 * $scal2) >> 14 (rounded)
    ; Sreg3 = (-$reg1 * $scal2 + $reg2 * $scal1) >> 14 (rounded)
    ; Sreg4 = (-$reg3 * $scal2 + $reg4 * $scal1) >> 14 (rounded)
    MACRO
    DO_BUTTERFLY_STD        $reg1,  $reg2,  $reg3,  $reg4,  $scal1, $scal2, $tmp1,  $tmp2,  $tmp3

    vmull.s16       $tmp1,  $reg1,  $scal1
    vmull.s16       $tmp2,  $reg3,  $scal2
    vadd.s32        $tmp3,  $tmp1,  $tmp2

    vmull.s16       $tmp1,  $reg1,  $scal2
    vmull.s16       $tmp2,  $reg3,  $scal1
    vsub.s32        $tmp2,  $tmp2,  $tmp1

    vrshrn.s32      $reg1,  $tmp3,  #14
    vrshrn.s32      $reg3,  $tmp2,  #14

    vmull.s16       $tmp1,  $reg2,  $scal1
    vmull.s16       $tmp2,  $reg4,  $scal2
    vadd.s32        $tmp3,  $tmp1,  $tmp2

    vmull.s16       $tmp1,  $reg2,  $scal2
    vmull.s16       $tmp2,  $reg4,  $scal1
    vsub.s32        $tmp2,  $tmp2,  $tmp1

    vrshrn.s32      $reg2,  $tmp3,  #14
    vrshrn.s32      $reg4,  $tmp2,  #14

    MEND

    ;---------------------------------------------------------------------------
    ; Computes butterfly for DST. Cosine/Sine coefficients are unique
    ; Computes butterfly of 4 registers
    ;  $src1   : Source register for Butterfly      (64  bit)
    ;  $src2   : Source register for Butterfly      (64  bit)
    ;  $src3   : Source register for Butterfly      (64  bit)
    ;  $src4   : Source register for Butterfly      (64  bit)
    ;  $scal1  : Scaling value 1                    (64  bit)
    ;  $scal2  : Scaling value 2                    (64  bit)
    ;  $dst1   : Destination register for Butterfly (128 bit)
    ;  $dst2   : Destination register for Butterfly (128 bit)
    ;  $dst3   : Destination register for Butterfly (128 bit)
    ;  $dst4   : Destination register for Butterfly (128 bit)
    ;  $qsrc1  : Q register corresponding to $src1 & $src2
    ;
    ; $dst1 = ( $src1 * $scal1 + $src2 * $scal2)
    ; $dst1 = ( $src3 * $scal1 + $src4 * $scal2)
    ; $dst1 = (-$src1 * $scal2 + $src2 * $scal1)
    ; $dst1 = (-$src3 * $scal2 + $src4 * $scal1)
    MACRO
    DO_BUTTERFLY_STD_DST    $src1,  $src2,  $src3,  $src4,  $qsrc1, $scal1, $scal2, $dst1, $dst2, $dst3, $dst4

    ; Butterfly for $src1 and $src3
    vmull.s16       $dst2,  $src1,  $scal1
    vmull.s16       $dst4,  $src3,  $scal2
    vadd.s32        $dst1,  $dst2,  $dst4

    vmull.s16       $dst2,  $src1,  $scal2
    vmull.s16       $dst4,  $src3,  $scal1
    vsub.s32        $dst3,  $dst4,  $dst2

    ; Butterfly for $src2 and $src4
    vmull.s16       $dst2,  $src2,  $scal1
    vmull.s16       $dst4,  $src4,  $scal2
    vadd.s32        $dst2,  $dst2,  $dst4

    vmull.s16       $dst4,  $src2,  $scal2
    vmull.s16       $qsrc1, $src4,  $scal1
    vsub.s32        $dst4,  $qsrc1, $dst4

    MEND

    ;---------------------------------------------------------------------------
    ; Transpose an 8x8 16 bit data matrix.
    ; Data is loaded in q0-q7.
    MACRO
    TRANSPOSE_8X8

    vswp            d1,     d8
    vswp            d7,     d14
    vswp            d5,     d12
    vswp            d3,     d10
    vtrn.32         q0,     q2
    vtrn.32         q1,     q3
    vtrn.32         q4,     q6
    vtrn.32         q5,     q7
    vtrn.16         q0,     q1
    vtrn.16         q2,     q3
    vtrn.16         q4,     q5
    vtrn.16         q6,     q7

    MEND


;void vp9_fht8x8_neon(const int16_t *input, tran_low_t *output, int stride,
;                       int tx_type )
;
;   r0  int16_t     *input,
;   r1  tran_low_t  *out,
;   r2  int         stride
;   r3  int         tx_type
;   Computes 2-D hybrid transform of an 8x8 block
;   This is done in two stages, stage 1 does a column transform
;   The output of stage 1 is transposed
;   The transposed output is again column transformed, hence a row transform
;       is done in effect
;   Final output is transposed and written to memory
;   The transforms to be done in each stage is decided by variable tx_type

|vp9_fht8x8_neon| PROC

    push            {r4-r11,lr}
    vpush           {d8-d15}

    ; Double the stride to accomodate 2 byte data type
    lsl             r2,     #1

    ; Load the input 8x8 block into q0-q8
    vld1.s16        {q0},   [r0],   r2
    vld1.s16        {q1},   [r0],   r2
    vld1.s16        {q2},   [r0],   r2
    vld1.s16        {q3},   [r0],   r2
    vld1.s16        {q4},   [r0],   r2
    vld1.s16        {q5},   [r0],   r2
    vld1.s16        {q6},   [r0],   r2
    vld1.s16        {q7},   [r0]

    ;temp_in[j] = input[j * stride + i] * 4;
    vshl.s16        q0,     q0,     #2
    vshl.s16        q1,     q1,     #2
    vshl.s16        q2,     q2,     #2
    vshl.s16        q3,     q3,     #2
    vshl.s16        q4,     q4,     #2
    vshl.s16        q5,     q5,     #2
    vshl.s16        q6,     q6,     #2
    vshl.s16        q7,     q7,     #2

    ; Decide stage 1 transform type
    mov             r2,     #1
    and             r0,     r3,     r2
    cmp             r0,     #0
    bleq            do_fdct_8x1
    blne            do_fadst_8x1

    ; Transpose between stages
    TRANSPOSE_8x8

    ; Decide stage 2 transform type
    lsr             r0,     r3,     #1
    cmp             r0,     #0
    bleq            do_fdct_8x1
    blne            do_fadst_8x1

    ; Transpose the 8x8 block before writing into output
    TRANSPOSE_8x8

    ; Do scaling
    ;output[j + i * 8] = (temp_out[j]) + (temp_out[j] < 0)) >> 1;
    ; Extract the sign bit to get (temp_out[j] < 0)
    vshr.u16        q8,     q0,     #15
    vshr.u16        q9,     q1,     #15
    vshr.u16        q10,    q2,     #15
    vshr.u16        q11,    q3,     #15
    vshr.u16        q12,    q4,     #15
    vshr.u16        q13,    q5,     #15
    vshr.u16        q14,    q6,     #15
    vshr.u16        q15,    q7,     #15

    ;(temp_out[j] + (temp_out[j] < 0))
    vadd.s16        q0,     q0,     q8
    vadd.s16        q1,     q1,     q9
    vadd.s16        q2,     q2,     q10
    vadd.s16        q3,     q3,     q11
    vadd.s16        q4,     q4,     q12
    vadd.s16        q5,     q5,     q13
    vadd.s16        q6,     q6,     q14
    vadd.s16        q7,     q7,     q15

    ;(temp_out[j] + (temp_out[j] < 0)) >> 1;
    vshr.s16        q0,     q0,     #1
    vshr.s16        q1,     q1,     #1
    vshr.s16        q2,     q2,     #1
    vshr.s16        q3,     q3,     #1
    vshr.s16        q4,     q4,     #1
    vshr.s16        q5,     q5,     #1
    vshr.s16        q6,     q6,     #1
    vshr.s16        q7,     q7,     #1

    ; Write block into output
    vst1.16         {q0-q1},[r1]!
    vst1.16         {q2-q3},[r1]!
    vst1.16         {q4-q5},[r1]!
    vst1.16         {q6-q7},[r1]

    vpop            {d8-d15}
    pop             {r4-r11,lr}
    bx              lr

    ; end vp9_fht8x8_neon
    ENDP


;void vp9_fdct8x8_neon(const int16_t *input, tran_low_t *output, int stride)
;
;   r0  int16_t    *input,
;   r1  tran_low_t *out,
;   r2  int        stride
;
;   Computes 2-D cosine transform of an 8x8 block
;   This is done in two stages, stage 1 does a column cosine transform
;   The output of stage 1 is transposed
;   The transposed output is again column transformed, hence a row transform
;       is done in effect
;   Final output is transposed and written to memory
|vp9_fdct8x8_neon| PROC

    push            {r4-r11,lr}
    vpush           {d8-d15}

    ; Double the stride to accommodate 2 byte data type
    lsl             r2,     #1

    ; Load the input 8x8 block into q0-q8
    vld1.s16        {q0},   [r0],   r2
    vld1.s16        {q1},   [r0],   r2
    vld1.s16        {q2},   [r0],   r2
    vld1.s16        {q3},   [r0],   r2
    vld1.s16        {q4},   [r0],   r2
    vld1.s16        {q5},   [r0],   r2
    vld1.s16        {q6},   [r0],   r2
    vld1.s16        {q7},   [r0]

    ;temp_in[j] = input[j * stride + i]* 4;
    vshl.s16        q0,     q0,     #2
    vshl.s16        q1,     q1,     #2
    vshl.s16        q2,     q2,     #2
    vshl.s16        q3,     q3,     #2
    vshl.s16        q4,     q4,     #2
    vshl.s16        q5,     q5,     #2
    vshl.s16        q6,     q6,     #2
    vshl.s16        q7,     q7,     #2

    ; Do stage 1 transform
    bl              do_fdct_8x1
    ; Transpose between stages
    TRANSPOSE_8x8
    ; Do stage 2 transform
    bl              do_fdct_8x1
    ; Transpose the block before writing into output
    TRANSPOSE_8x8

    ; Do scaling
    ;final_output[j + i * 8] /= 2;
    ; Extract the sign bit to get (temp_out[j] < 0)
    vshr.u16        q8,     q0,     #15
    vshr.u16        q9,     q1,     #15
    vshr.u16        q10,    q2,     #15
    vshr.u16        q11,    q3,     #15
    vshr.u16        q12,    q4,     #15
    vshr.u16        q13,    q5,     #15
    vshr.u16        q14,    q6,     #15
    vshr.u16        q15,    q7,     #15

    ;(temp_out[j] + (temp_out[j] < 0))
    vadd.s16        q0,     q0,     q8
    vadd.s16        q1,     q1,     q9
    vadd.s16        q2,     q2,     q10
    vadd.s16        q3,     q3,     q11
    vadd.s16        q4,     q4,     q12
    vadd.s16        q5,     q5,     q13
    vadd.s16        q6,     q6,     q14
    vadd.s16        q7,     q7,     q15

    ;(temp_out[j] + (temp_out[j] < 0)) >> 1
    ; equivalent to
    ;final_output[j + i * 8] /= 2;
    vshr.s16        q0,     q0,     #1
    vshr.s16        q1,     q1,     #1
    vshr.s16        q2,     q2,     #1
    vshr.s16        q3,     q3,     #1
    vshr.s16        q4,     q4,     #1
    vshr.s16        q5,     q5,     #1
    vshr.s16        q6,     q6,     #1
    vshr.s16        q7,     q7,     #1

    ; Write block into output
    vst1.16         {q0-q1},[r1]!
    vst1.16         {q2-q3},[r1]!
    vst1.16         {q4-q5},[r1]!
    vst1.16         {q6-q7},[r1]

    vpop            {d8-d15}
    pop             {r4-r11,lr}
    bx              lr

    ; end vp9_fdct8x8_neon
    ENDP


do_fdct_8x1
    ; Computes the 8x1 1-D forward cosine transform
    ; Input     : Input must be in q0-q7
    ; Output    : Output will be provided in q0-q7
    ; Registers q8-q15 will not be preserved
    ; Arm register r2 will not be preserved
    ;
    ; input[0]  : q0        output[0]  : q0
    ; input[1]  : q1        output[1]  : q1
    ; input[2]  : q2        output[2]  : q2
    ; input[3]  : q3        output[3]  : q3
    ; input[4]  : q4        output[4]  : q4
    ; input[5]  : q5        output[5]  : q5
    ; input[6]  : q6        output[6]  : q6
    ; input[7]  : q7        output[7]  : q7

    ; --------------------------------------------------------------------------
    ; STAGE 1
    ; --------------------------------------------------------------------------
    ;s0 = input[0] + input[7];
    ;s7 = input[0] - input[7];
    DO_BUTTERFLY_NO_COEFFS q0, q7, q8
    ;s1 = input[1] + input[6];
    ;s6 = input[1] - input[6];
    DO_BUTTERFLY_NO_COEFFS q1, q6, q9
    ;s2 = input[2] + input[5];
    ;s5 = input[2] - input[5];
    DO_BUTTERFLY_NO_COEFFS q2, q5, q10
    ;s3 = input[3] + input[4];
    ;s4 = input[3] - input[4];
    DO_BUTTERFLY_NO_COEFFS q3, q4, q11

    ; fdct4(step, step);
    ;x0 = s0 + s3;
    ;x3 = s0 - s3;
    DO_BUTTERFLY_NO_COEFFS q7, q4, q0
    ;x1 = s1 + s2;
    ;x2 = s1 - s2;
    DO_BUTTERFLY_NO_COEFFS q6, q5, q1
    ;t0 = (x0 + x1) * cospi_16_64;
    ;t1 = (x0 - x1) * cospi_16_64;
    ;output[0] = fdct_round_shift(t0);
    ;output[4] = fdct_round_shift(t1);
    GET_CONST d31, cospi_16_64, r2
    vmovl.s16       q15,    d31
    DO_BUTTERFLY_SYMMETRIC_COEFFS d8, d9, d10, d11, q15, d4, d5, q13, q14
    ;t2 =  x2 * cospi_24_64 + x3 *  cospi_8_64;
    ;t3 = -x2 * cospi_8_64  + x3 * cospi_24_64;
    ;output[2] = fdct_round_shift(t2);
    ;output[6] = fdct_round_shift(t3);
    GET_CONST d30, cospi_24_64, r2
    GET_CONST d31, cospi_8_64,  r2
    DO_BUTTERFLY_STD d2, d3, d0, d1, d30, d31, q12, q13, q14
    ; KEEP OUTPUT  : t0 : output[0] : q0
    ;              : t1 : output[4] : q4
    ;              : t2 : output[2] : q2
    ;              : t3 : output[6] : q6
    ; get t3 into q6
    vmov.s16        q6,     q0
    ; get t0 into q0
    vmov.s16        q0,     q5
    ; get t1 into q4
    vmov.s16        q4,     q2
    ; get t2 into q2
    vmov.s16        q2,     q1

    ; --------------------------------------------------------------------------
    ; STAGE 2
    ; --------------------------------------------------------------------------
    ;t0 = (s6 - s5) * cospi_16_64;
    ;t1 = (s6 + s5) * cospi_16_64;
    ;t2 = fdct_round_shift(t0);
    ;t3 = fdct_round_shift(t1);
    GET_CONST d31, cospi_16_64, r2
    vmovl.s16       q15,    d31
    ;t2->q3, t3->q10
    DO_BUTTERFLY_SYMMETRIC_COEFFS d18, d19, d20, d21, q15, d6, d7, q13, q14

    ; --------------------------------------------------------------------------
    ; STAGE 3
    ; --------------------------------------------------------------------------
    ;x0 = s4 + t2;
    ;x1 = s4 - t2;
    DO_BUTTERFLY_NO_COEFFS q11, q3, q9
    ;x2 = s7 - t3;
    ;x3 = s7 + t3;
    ;x2->q11, x3->q10
    DO_BUTTERFLY_NO_COEFFS q8, q10, q11

    ; --------------------------------------------------------------------------
    ; STAGE 4
    ; --------------------------------------------------------------------------
    ;t0 = x0 * cospi_28_64 + x3 *  cospi_4_64;
    ;t3 = x3 * cospi_28_64 + x0 * -cospi_4_64;
    ;output[1] = fdct_round_shift(t0);
    ;output[7] = fdct_round_shift(t3);
    GET_CONST d31, cospi_4_64, r2
    GET_CONST d30, cospi_28_64, r2
    DO_BUTTERFLY_STD d6, d7, d20, d21, d30, d31, q12, q13, q14

    ;t1 = x1 * cospi_12_64 + x2 *  cospi_20_64;
    ;t2 = x2 * cospi_12_64 + x1 * -cospi_20_64;
    ;output[3] = fdct_round_shift(t2);
    ;output[5] = fdct_round_shift(t1);
    GET_CONST d31, cospi_20_64, r2
    GET_CONST d30, cospi_12_64, r2
    DO_BUTTERFLY_STD d18, d19, d22, d23, d30, d31, q12, q13, q14

    ; KEEP OUTPUT  : t0 : output[1] : q1
    ;              : t1 : output[3] : q3
    ;              : t2 : output[5] : q5
    ;              : t3 : output[7] : q7
    ; get t0 to q1
    vmov.s16        q1,     q3
    ; get t1 to q3
    vswp.s16        q3,     q11
    ; get t2 to q5
    vmov.s16        q5,     q9
    ; get t3 to q7
    vmov.s16        q7,     q10

    ;Return to caller
    bx              lr


do_fadst_8x1
    ; Computes the 8x1 1-D forward sine transform
    ; Input     : Input must be in q0-q7
    ; Output    : Output will be provided in q0-q7
    ; Registers q8-q15 will not be preserved
    ;
    ; input[0]  : q0        output[0]  : q0
    ; input[1]  : q1        output[1]  : q1
    ; input[2]  : q2        output[2]  : q2
    ; input[3]  : q3        output[3]  : q3
    ; input[4]  : q4        output[4]  : q4
    ; input[5]  : q5        output[5]  : q5
    ; input[6]  : q6        output[6]  : q6
    ; input[7]  : q7        output[7]  : q7
    ;
    ; Inputs will be used as
    ; tran_high_t x0 = input[7] : q7
    ; tran_high_t x1 = input[0] : q0
    ; tran_high_t x2 = input[5] : q5
    ; tran_high_t x3 = input[2] : q2
    ; tran_high_t x4 = input[3] : q3
    ; tran_high_t x5 = input[4] : q4
    ; tran_high_t x6 = input[1] : q1
    ; tran_high_t x7 = input[6] : q6

    ; --------------------------------------------------------------------------
    ; STAGE 1 : Processing 0,1,4,5
    ; --------------------------------------------------------------------------
    ;s0 = cospi_2_64  * x0 + cospi_30_64 * x1;
    ;s1 = cospi_30_64 * x0 - cospi_2_64  * x1;
    GET_CONST d30, cospi_2_64,  r2
    GET_CONST d31, cospi_30_64, r2
    DO_BUTTERFLY_STD_DST d0, d1, d14, d15, q0, d31, d30, q8, q9, q10, q11
    ;s4 = cospi_18_64 * x4 + cospi_14_64 * x5;
    ;s5 = cospi_14_64 * x4 - cospi_18_64 * x5;
    GET_CONST d30, cospi_18_64, r2
    GET_CONST d31, cospi_14_64, r2
    DO_BUTTERFLY_STD_DST d8, d9, d6, d7, q4, d31, d30, q12, q13, q14, q0
    ;x0 = fdct_round_shift(s0 + s4);
    ;x4 = fdct_round_shift(s0 - s4);
    DO_BUTTERFLY_NO_COEFFS_DST q8, q9, q12, q13, d6, d7, d8, d9, q4
    ;x1 = fdct_round_shift(s1 + s5);
    ;x5 = fdct_round_shift(s1 - s5);
    DO_BUTTERFLY_NO_COEFFS_DST q10, q11, q14, q0, d16, d17, d18, d19, q9

    ; --------------------------------------------------------------------------
    ; STAGE 1 : Processing 2,3,6,7
    ; --------------------------------------------------------------------------
    ;s2 = cospi_10_64 * x2 + cospi_22_64 * x3;
    ;s3 = cospi_22_64 * x2 - cospi_10_64 * x3;
    GET_CONST d30, cospi_22_64, r2
    GET_CONST d31, cospi_10_64, r2
    DO_BUTTERFLY_STD_DST d4, d5, d10, d11, q2, d30, d31, q10, q11, q12, q13
    ;s6 = cospi_26_64 * x6 + cospi_6_64  * x7;
    ;s7 = cospi_6_64  * x6 - cospi_26_64 * x7;
    GET_CONST d30, cospi_6_64, r2
    GET_CONST d31, cospi_26_64, r2
    DO_BUTTERFLY_STD_DST d12, d13, d2, d3, q6, d30, d31, q14, q2, q5, q0
    ;x2 = fdct_round_shift(s2 + s6);
    ;x6 = fdct_round_shift(s2 - s6);
    DO_BUTTERFLY_NO_COEFFS_DST q10, q11, q14, q2, d12, d13, d2, d3, q1
    ;x3 = fdct_round_shift(s3 + s7);
    ;x7 = fdct_round_shift(s3 - s7);
    DO_BUTTERFLY_NO_COEFFS_DST q12, q13, q5, q0, d20, d21, d22, d23, q11

    ; --------------------------------------------------------------------------
    ; STAGE 2 : Processing 0-7
    ; --------------------------------------------------------------------------
    ;s0 = x0;
    ;s1 = x1;
    ;s2 = x2;
    ;s3 = x3;
    ;s4 = cospi_8_64  * x4 + cospi_24_64 * x5;
    ;s5 = cospi_24_64 * x4 - cospi_8_64  * x5;
    GET_CONST d30, cospi_24_64, r2
    GET_CONST d31, cospi_8_64, r2
    DO_BUTTERFLY_STD_DST d18, d19, d8, d9, q9, d30, d31, q0, q7, q2, q5
    ;s6 = - cospi_24_64 * x6 + cospi_8_64  * x7;
    ;s7 =   cospi_8_64  * x6 + cospi_24_64 * x7;
    GET_CONST d30, cospi_8_64, r2
    GET_CONST d31, cospi_24_64, r2
    ; s6-> q12,q13 s7->q14,q4
    DO_BUTTERFLY_STD_DST d2, d3, d22, d23, q1, d30, d31, q14, q4, q12, q13

    ;x4 = fdct_round_shift(s4 + s6);
    ;x6 = fdct_round_shift(s4 - s6);
    DO_BUTTERFLY_NO_COEFFS_DST q0, q7, q12, q13, d18, d19, d30, d31, q9
    ;x5 = fdct_round_shift(s5 + s7);
    ;x7 = fdct_round_shift(s5 - s7);
    DO_BUTTERFLY_NO_COEFFS_DST q2, q5, q14, q4, d2, d3, d22, d23, q1

    ;x0 = s0 + s2;
    ;x2 = s0 - s2;
    DO_BUTTERFLY_NO_COEFFS q3, q6,  q2
    ;x1 = s1 + s3;
    ;x3 = s1 - s3;
    DO_BUTTERFLY_NO_COEFFS q8, q10, q5

    ; KEEP OUTPUT :  x0 : output[0] : q0
    ;             : -x1 : output[7] : q7
    ;             : -x4 : output[1] : q1
    ;             :  x5 : output[6] : q6
    ; Get x0 to  q0
    vmov.s16        q0,     q6
    ; Get  x5 to q6
    vmov.s16        q6,     q1
    ; Get -x4 to q1
    vneg.s16        q1,     q9
    ; Get -x1 to q7
    vneg.s16        q7,     q10

    ; --------------------------------------------------------------------------
    ; STAGE 3 : Processing 0-7
    ; --------------------------------------------------------------------------
    ;s2 = cospi_16_64 * (x2 + x3);
    ;s3 = cospi_16_64 * (x2 - x3);
    ;x2 = fdct_round_shift(s2);
    ;x3 = fdct_round_shift(s3);
    GET_CONST d20, cospi_16_64, r2
    vmovl.s16       q10,    d20
    DO_BUTTERFLY_SYMMETRIC_COEFFS d4,d5,d10,d11, q10, d8,d9,q12,q13
    ;s6 = cospi_16_64 * (x6 + x7);
    ;s7 = cospi_16_64 * (x6 - x7);
    ;x6 = fdct_round_shift(s6);
    ;x7 = fdct_round_shift(s7);
    GET_CONST d20, cospi_16_64, r2
    vmovl.s16       q10,    d20
    DO_BUTTERFLY_SYMMETRIC_COEFFS d30,d31,d22,d23, q10,d4,d5, q12,q13

    ; KEEP OUTPUT : -x2 : output[3] : q3
    ;             :  x3 : output[4] : q4
    ;             :  x6 : output[2] : q2
    ;             : -x7 : output[5] : q5

    ; get -x2 to q3
    vneg.s16        q3,     q5
    ; get -x7 to q5
    vneg.s16        q5,     q2
    ; get  x3 to q4
    ; already in q4
    ; get x6 to q2
    vmov.s16        q2,     q11

    ; Return back to caller
    bx              lr    ;do_fadst_8x1


    ; end of code
    END

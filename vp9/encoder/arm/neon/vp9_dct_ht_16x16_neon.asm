;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;
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
    EXPORT  |vp9_fdct16x16_neon|
    EXPORT  |vp9_fht16x16_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

    AREA     Block, CODE, READONLY
    ;---------------------------------------------------------------------------
    ; Load values to 16 q registers
    MACRO
    LOAD_INPUT           $ptr, $stride
    vld1.64         q0,  [$ptr],  $stride
    vld1.64         q1,  [$ptr],  $stride
    vld1.64         q2,  [$ptr],  $stride
    vld1.64         q3,  [$ptr],  $stride
    vld1.64         q4,  [$ptr],  $stride
    vld1.64         q5,  [$ptr],  $stride
    vld1.64         q6,  [$ptr],  $stride
    vld1.64         q7,  [$ptr],  $stride
    vld1.64         q8,  [$ptr],  $stride
    vld1.64         q9,  [$ptr],  $stride
    vld1.64         q10, [$ptr],  $stride
    vld1.64         q11, [$ptr],  $stride
    vld1.64         q12, [$ptr],  $stride
    vld1.64         q13, [$ptr],  $stride
    vld1.64         q14, [$ptr],  $stride
    vld1.64         q15, [$ptr],  $stride
    MEND
    ;---------------------------------------------------------------------------
    ;Transpose a 8x8 16bit data matrix. Datas are loaded in q8-q15.
    MACRO
    TRANSPOSE8X8_Q8_TO_Q15
    vswp            d17,  d24
    vswp            d23,  d30
    vswp            d21,  d28
    vswp            d19,  d26
    vtrn.32         q8,   q10
    vtrn.32         q9,   q11
    vtrn.32         q12,  q14
    vtrn.32         q13,  q15
    vtrn.16         q8,   q9
    vtrn.16         q10,  q11
    vtrn.16         q12,  q13
    vtrn.16         q14,  q15
    MEND
    ;---------------------------------------------------------------------------
    ; Transpose a 8x8 16bit data matrix. Datas are loaded in q0-q7.
    MACRO
    TRANSPOSE8X8_Q0_TO_Q7
    vswp            d1,   d8
    vswp            d7,   d14
    vswp            d5,   d12
    vswp            d3,   d10
    vtrn.32         q0,   q2
    vtrn.32         q1,   q3
    vtrn.32         q4,   q6
    vtrn.32         q5,   q7
    vtrn.16         q0,   q1
    vtrn.16         q2,   q3
    vtrn.16         q4,   q5
    vtrn.16         q6,   q7
    MEND
    ; --------------------------------------------------------------------------
    ; Multiply all registers(q0-q15) by 4
    MACRO
    MULTIPLY_BY_4_Q0_TO_Q15
    vshl.i16        q0,   q0,  #2
    vshl.i16        q1,   q1,  #2
    vshl.i16        q2,   q2,  #2
    vshl.i16        q3,   q3,  #2
    vshl.i16        q4,   q4,  #2
    vshl.i16        q5,   q5,  #2
    vshl.i16        q6,   q6,  #2
    vshl.i16        q7,   q7,  #2
    vshl.i16        q8,   q8,  #2
    vshl.i16        q9,   q9,  #2
    vshl.i16        q10,  q10, #2
    vshl.i16        q11,  q11, #2
    vshl.i16        q12,  q12, #2
    vshl.i16        q13,  q13, #2
    vshl.i16        q14,  q14, #2
    vshl.i16        q15,  q15, #2
    MEND

    ; --------------------------------------------------------------------------
    ; $diff cannot be same as $ip1 or $ip2
    MACRO
    DO_BUTTERFLY_NO_COEFFS $ip1, $ip2, $sum, $diff
    vsub.s16        $diff, $ip1,  $ip2
    vadd.s16        $sum,  $ip1,  $ip2
    MEND
    ; --------------------------------------------------------------------------
    ; Touches q12, q15 and the input registers
    ; valid output registers are anything but q12, q15, $ip1, $ip2
    MACRO
    DO_BUTTERFLY_SYMMETRIC_COEFFS  $ip1, $ip2, $constant, $op1, $op2, $op3, $op4
    ; generate scalar constants
    mov             r8,    #$constant  & 0xFF00
    add             r8,    #$constant  & 0x00FF
    vdup.16         $op4,  r8
    vadd.s16        q12,   $ip1, $ip2
    vsub.s16        q15,   $ip1, $ip2
    vmull.s16       $ip1,  d24,  $op4
    vmull.s16       $ip2,  d25,  $op4
    vmull.s16       q12,   d30,  $op4
    vmull.s16       q15,   d31,  $op4
    vqrshrn.s32     $op1,  $ip1, #14
    vqrshrn.s32     $op2,  $ip2, #14
    vqrshrn.s32     $op3,  q12,  #14
    vqrshrn.s32     $op4,  q15,  #14
    MEND
    ; --------------------------------------------------------------------------
    ; Touches q8-q12, q15 (q13-q14 are preserved)
    ; valid output registers are anything but q8-q11
    MACRO
    DO_BUTTERFLY_DCT $ip1, $ip2, $ip3, $ip4, $first_constant, $second_constant, $op1, $op2, $op3, $op4
    ; generate the constants
    mov             r8,  #$first_constant  & 0xFF00
    mov             r12, #$second_constant & 0xFF00
    add             r8,  #$first_constant  & 0x00FF
    add             r12, #$second_constant & 0x00FF
    ; generate vector constants
    vdup.16         d30, r8
    vdup.16         d31, r12
    ; (used) two for inputs (ip3-ip2), one for constants (q15)
    ; do some multiplications (ordered for maximum latency hiding)
    vmull.s16       q8,  $ip1, d30
    vmull.s16       q10, $ip3, d31
    vmull.s16       q9,  $ip2, d30
    vmull.s16       q11, $ip4, d31
    vmull.s16       q12, $ip1, d31
    ; (used) five for intermediate (q8-q12), one for constants (q15)
    ; do some addition/subtractions (to get back two register)
    vsub.s32        q8,  q8,  q10
    vsub.s32        q9,  q9,  q11
    ; do more multiplications (ordered for maximum latency hiding)
    vmull.s16       q10, $ip2, d31
    vmull.s16       q11, $ip3, d30
    vmull.s16       q15, $ip4, d30
    ; (used) six for intermediate (q8-q12, q15)
    ; do more addition/subtractions
    vadd.s32        q11, q12, q11
    vadd.s32        q10, q10, q15
    vqrshrn.s32     $op1,q8,  #14
    vqrshrn.s32     $op2,q9,  #14
    ; (used) four for intermediate (q8-q11)
    ; dct_const_round_shift
    vqrshrn.s32     $op3,q11, #14
    vqrshrn.s32     $op4,q10, #14
    MEND

    ; --------------------------------------------------------------------------
    ; BUTTERFLY for DST
    ; Touches q12, q14
    ; valid output registers are anything but q12 ,q14, $ip1, $ip2, $ip3, $ip4
    MACRO
    DO_BUTTERFLY_DST $ip1, $ip2, $ip3, $ip4, $first_constant, $second_constant, $op1, $op2, $op3, $op4
    ; generate the constants
    mov             r8,    #$first_constant  & 0xFF00
    mov             r12,   #$second_constant & 0xFF00
    add             r8,    #$first_constant  & 0x00FF
    add             r12,   #$second_constant & 0x00FF
    ; generate vector constants
    vdup.16         d28,   r8
    vdup.16         d29,   r12
    ; do some multiplications (ordered for maximum latency hiding)
    vmull.s16       $op1,  $ip1,  d28
    vmull.s16       $op3,  $ip3,  d29
    vmull.s16       $op2,  $ip2,  d28
    vmull.s16       $op4,  $ip4,  d29
    vmull.s16       q12,   $ip1,  d29
    vsub.s32        $op1,  $op1,  $op3
    vsub.s32        $op2,  $op2,  $op4
    ; do more multiplications (ordered for maximum latency hiding)
    vmull.s16       $op3,  $ip3,  d28
    vmull.s16       $op4,  $ip2,  d29
    vmull.s16       q14,   $ip4,  d28
    ; do more addition/subtractions
    vadd.s32        $op3,  $op3,  q12
    vadd.s32        $op4,  $op4,  q14
    MEND
    ; --------------------------------------------------------------------------
    ; BUTTERFLY and ROUND SHIFT for DST
    ; Touches q12, q14
    ; valid output registers are anything but q12 ,q14, $ip1 or $ip2
    ; modifies values of $ip1 and $ip2
    MACRO
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT $ip1, $ip2, $ip3, $ip4, $op1, $op2, $op3, $op4
    vadd.s32        q12,  $ip1, $ip3
    vadd.s32        q14,  $ip2, $ip4
    vsub.s32        $ip1, $ip1, $ip3
    vsub.s32        $ip2, $ip2, $ip4
    vqrshrn.s32     $op1, q12,  #14
    vqrshrn.s32     $op2, q14,  #14
    vqrshrn.s32     $op3, $ip1, #14
    vqrshrn.s32     $op4, $ip2, #14
    MEND

    ; --------------------------------------------------------------------------
    ; Calculate the input for the DCT transform
    ; Inputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    ; Outputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    MACRO
    CALC_INPUT_FOR_DCT   $temp_buffer $pass
    vst1.64         {q15}, [$temp_buffer]
    vadd.s16        q0,  q0,  q15  ; (in_pass0[0*stride] + in_pass0[15*stride])
    vsub.s16        q15, q1,  q14  ; (in_pass0[1*stride] - in_pass0[14*stride])
    vadd.s16        q1,  q1,  q14  ; (in_pass0[1*stride] + in_pass0[14*stride])
    vsub.s16        q14, q2,  q13  ; (in_pass0[2*stride] - in_pass0[13*stride])
    vadd.s16        q2,  q2,  q13  ; (in_pass0[2*stride] + in_pass0[13*stride])
    vsub.s16        q13, q3,  q12  ; (in_pass0[3*stride] - in_pass0[12*stride])
    vadd.s16        q3,  q3,  q12  ; (in_pass0[3*stride] + in_pass0[12*stride])
    vsub.s16        q12, q4,  q11  ; (in_pass0[4*stride] - in_pass0[11*stride])
    vadd.s16        q4,  q4,  q11  ; (in_pass0[4*stride] + in_pass0[11*stride])
    vsub.s16        q11, q5,  q10  ; (in_pass0[5*stride] - in_pass0[10*stride])
    vadd.s16        q5,  q5,  q10  ; (in_pass0[5*stride] + in_pass0[10*stride])
    vsub.s16        q10, q6,  q9   ; (in_pass0[6*stride] - in_pass0[9*stride])
    vadd.s16        q6,  q6,  q9   ; (in_pass0[6*stride] + in_pass0[9*stride])
    vsub.s16        q9,  q7,  q8   ; (in_pass0[7*stride] - in_pass0[8*stride])
    vadd.s16        q7,  q7,  q8   ; (in_pass0[7*stride] + in_pass0[8*stride])
    vld1.64         q8,  [$temp_buffer]
    vshl.i16        q8,  q8,  #1
    vsub.s16        q8,  q0,  q8   ; (in_pass0[0*stride] - in_pass0[15*stride])
    ; multiplying by 4
  IF  $pass = 0
    MULTIPLY_BY_4_Q0_TO_Q15
  ENDIF
    MEND
    ; --------------------------------------------------------------------------
    ; Rounds and shifts value in input registers(for pass=1)
    ; Inputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    ; Outputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    ; touches r8,two vector push  and pops
    MACRO
    ROUND_SHIFT   $transform
    vpush           {q15}
    mov             r8,  #1
    vdup.16         q15, r8
    ; adding 1
    vadd.s16        q0,  q0,  q15
    vadd.s16        q1,  q1,  q15
    vadd.s16        q2,  q2,  q15
    vadd.s16        q3,  q3,  q15
    vadd.s16        q4,  q4,  q15
    vadd.s16        q5,  q5,  q15
    vadd.s16        q6,  q6,  q15
    vadd.s16        q7,  q7,  q15
    vadd.s16        q8,  q8,  q15
    vadd.s16        q9,  q9,  q15
    vadd.s16        q10, q10, q15
    vadd.s16        q11, q11, q15
    vadd.s16        q12, q12, q15
    vadd.s16        q13, q13, q15
    vadd.s16        q14, q14, q15
    ; if hybrid tranform  ,outptr[j * 16 + i] =
    ;            (temp_out[j] + 1 + (temp_out[j] < 0)) >> 2;
 IF $transform = hybrid
    vshr.u16        q15, q0,  #15  ;taking sign bit and adding
    vadd.s16        q0,  q0,  q15
    vshr.u16        q15, q1,  #15
    vadd.s16        q1,  q1,  q15
    vshr.u16        q15, q2,  #15
    vadd.s16        q2,  q2,  q15
    vshr.u16        q15, q3,  #15
    vadd.s16        q3,  q3,  q15
    vshr.u16        q15, q4,  #15
    vadd.s16        q4,  q4,  q15
    vshr.u16        q15, q5,  #15
    vadd.s16        q5,  q5,  q15
    vshr.u16        q15, q6,  #15
    vadd.s16        q6,  q6,  q15
    vshr.u16        q15, q7,  #15
    vadd.s16        q7,  q7,  q15
    vshr.u16        q15, q8,  #15
    vadd.s16        q8,  q8,  q15
    vshr.u16        q15, q9,  #15
    vadd.s16        q9,  q9,  q15
    vshr.u16        q15, q10, #15
    vadd.s16        q10, q10, q15
    vshr.u16        q15, q11, #15
    vadd.s16        q11, q11, q15
    vshr.u16        q15, q12, #15
    vadd.s16        q12, q12, q15
    vshr.u16        q15, q13, #15
    vadd.s16        q13, q13, q15
    vshr.u16        q15, q14, #15
    vadd.s16        q14, q14, q15
 ENDIF
    vpop            {q15}
    vpush           {q14}
    vdup.16         q14, r8
    vadd.s16        q15, q14, q15
 IF $transform = hybrid
   vshr.u16         q14, q15, #15
   vadd.s16         q15, q14, q15
 ENDIF
    vpop            {q14}
    ; divide by 4
    vshr.s16        q0,  q0,  #2
    vshr.s16        q1,  q1,  #2
    vshr.s16        q2,  q2,  #2
    vshr.s16        q3,  q3,  #2
    vshr.s16        q4,  q4,  #2
    vshr.s16        q5,  q5,  #2
    vshr.s16        q6,  q6,  #2
    vshr.s16        q7,  q7,  #2
    vshr.s16        q8,  q8,  #2
    vshr.s16        q9,  q9,  #2
    vshr.s16        q10, q10, #2
    vshr.s16        q11, q11, #2
    vshr.s16        q12, q12, #2
    vshr.s16        q13, q13, #2
    vshr.s16        q14, q14, #2
    vshr.s16        q15, q15, #2
    MEND
    ; --------------------------------------------------------------------------
    ; Does the following tasks
    ;   - calculates DCT transform for a single pass
    ;   - even rows of outputs are stored in intermediate buffer(r5)
    ;   - odd rows of output are  returned in registers q1,q3,q5,q7,q9,q11,q13
    ;     and q15
    ;   - [r7] is used as intermediate buffer
    ; Inputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    ; Outputs
    ;   q1,q3,...  q15 : all odd from Rows1 to Row15 (8 cols)
    ; touches all q registers, r8, r12, r3, r4, r11
    MACRO
    DCT_SINGLE_PASS
    mov             r11,   r7
    ; store step1[i] values for temporary work registers
    ; destination buffer is also used as intermediate buffer
    vst1.64         {q9},  [r7],  r6
    vst1.64         {q10}, [r7],  r6
    vst1.64         {q11}, [r7],  r6
    vst1.64         {q12}, [r7],  r6
    vst1.64         {q13}, [r7],  r6
    vst1.64         {q14}, [r7],  r6
    vst1.64         {q15}, [r7],  r6
    vst1.64         {q8},  [r7],  r6
    ; stage 1
    DO_BUTTERFLY_NO_COEFFS q0, q7, q0, q14
    DO_BUTTERFLY_NO_COEFFS q1, q6, q1, q7
    DO_BUTTERFLY_NO_COEFFS q2, q5, q2, q6
    DO_BUTTERFLY_NO_COEFFS q3, q4, q3, q5

    ; fdct4(step, step);
    DO_BUTTERFLY_NO_COEFFS q0, q3, q4, q13
    DO_BUTTERFLY_NO_COEFFS q1, q2, q0, q3

    ; t0 = (x0 + x1) * cospi_16_64;
    ; t1 = (x0 - x1) * cospi_16_64;
    ; out[0] = fdct_round_shift(t0);
    ; out[8] = fdct_round_shift(t1);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q4, q0, cospi_16_64, d2, d3, d4, d5
    ; t2 = x3 * cospi_8_64  + x2 * cospi_24_64;
    ; t3 = x3 * cospi_24_64 - x2 * cospi_8_64;
    ; out[4] = fdct_round_shift(t2);
    ; out[12] = fdct_round_shift(t3);
    DO_BUTTERFLY_DCT d26, d27, d6, d7, cospi_24_64, cospi_8_64, d0, d1, d8, d9

    ; Stage 2

    ; t0 = (s6 - s5) * cospi_16_64;
    ; t1 = (s6 + s5) * cospi_16_64;
    ; t2 = fdct_round_shift(t0);
    ; t3 = fdct_round_shift(t1);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q7, q6, cospi_16_64, d6, d7, d26, d27

    ; Stage 3
    DO_BUTTERFLY_NO_COEFFS q5,  q13, q7, q6
    DO_BUTTERFLY_NO_COEFFS q14, q3,  q5, q13

    ; Stage 4

    ; t0 = x0 * cospi_28_64 + x3 *   cospi_4_64;
    ; t3 = x3 * cospi_28_64 + x0 *  -cospi_4_64;
    ; out[2] = fdct_round_shift(t0);
    ; out[14] = fdct_round_shift(t3);
    DO_BUTTERFLY_DCT d10, d11, d14, d15, cospi_28_64, cospi_4_64, d6, d7, d28, d29
    ; t1 = x1 * cospi_12_64 + x2 *  cospi_20_64;
    ; t2 = x2 * cospi_12_64 + x1 * -cospi_20_64;
    ; out[6] = fdct_round_shift(t2);
    ; out[10] = fdct_round_shift(t1);
    DO_BUTTERFLY_DCT d26, d27, d12, d13, cospi_12_64, cospi_20_64, d10, d11, d14, d15
    ; storing all even values of the output
    vst1.64         {q1},  [r5],  r6
    vst1.64         {q14}, [r5],  r6
    vst1.64         {q4},  [r5],  r6
    vst1.64         {q5},  [r5],  r6
    vst1.64         {q2},  [r5],  r6
    vst1.64         {q7},  [r5],  r6
    vst1.64         {q0},  [r5],  r6
    vst1.64         {q3},  [r5],  r6
    ; Work on the next eight values; step1 -> odd_results
    mov             r7,    r11
    vld1.64         {q0},  [r7],  r6
    vld1.64         {q1},  [r7],  r6
    vld1.64         {q2},  [r7],  r6
    vld1.64         {q3},  [r7],  r6
    vld1.64         {q4},  [r7],  r6
    vld1.64         {q5},  [r7],  r6
    vld1.64         {q6},  [r7],  r6
    vld1.64         {q7},  [r7],  r6
    ; step 2

    ; temp1 = (step1[5] - step1[2]) * cospi_16_64;
    ; temp2 = (step1[5] + step1[2]) * cospi_16_64;
    ; step2[2] = fdct_round_shift(temp1);
    ; step2[5] = fdct_round_shift(temp2);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q5, q2, cospi_16_64, d26, d27, d28, d29

    ; temp2 = (step1[4] - step1[3]) * cospi_16_64;
    ; temp1 = (step1[4] + step1[3]) * cospi_16_64;
    ; step2[3] = fdct_round_shift(temp2);
    ; step2[4] = fdct_round_shift(temp1);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q4, q3, cospi_16_64, d10, d11, d4, d5

    ; step 3
    DO_BUTTERFLY_NO_COEFFS q0, q2,  q0, q3
    DO_BUTTERFLY_NO_COEFFS q1, q14, q1, q2
    DO_BUTTERFLY_NO_COEFFS q7, q5,  q7, q4
    DO_BUTTERFLY_NO_COEFFS q6, q13, q6, q5
    ; step 4

    ; temp1 = step3[1] *  -cospi_8_64 + step3[6] * cospi_24_64;
    ; temp2 = step3[1] * cospi_24_64 + step3[6] *  cospi_8_64
    ; step2[1] = fdct_round_shift(temp1);
    ; step2[6] = fdct_round_shift(temp2);
    DO_BUTTERFLY_DCT d12, d13, d2, d3, cospi_24_64, cospi_8_64, d26, d27, d28, d29

    ; temp2 = step3[2] * cospi_24_64 + step3[5] *  cospi_8_64;
    ; temp1 = step3[2] * cospi_8_64 - step3[5] * cospi_24_64;
    ; step2[2] = fdct_round_shift(temp2);
    ; step2[5] = fdct_round_shift(temp1);
    DO_BUTTERFLY_DCT d4, d5, d10, d11, cospi_8_64, cospi_24_64, d30, d31, d10, d11

    ; step 5
    DO_BUTTERFLY_NO_COEFFS q3, q5,  q3, q2
    DO_BUTTERFLY_NO_COEFFS q0, q13, q0, q1
    vswp.s16        q2,    q3
    DO_BUTTERFLY_NO_COEFFS q4, q15, q4, q5
    DO_BUTTERFLY_NO_COEFFS q7, q14, q14, q6
    vswp.s16        q4,    q5

    ; step 6

    ; temp1 = step1[1] * -cospi_18_64 + step1[6] * cospi_14_64
    ; out[7] = fdct_round_shift(temp1);
    ; temp2 = step1[1] * cospi_14_64 + step1[6] * cospi_18_64;
    ; out[9] = fdct_round_shift(temp2);
    DO_BUTTERFLY_DCT d12, d13, d2, d3, cospi_14_64, cospi_18_64, d14, d15, d12, d13

    ; temp2 = step1[0] *  -cospi_2_64 + step1[7] * cospi_30_64;
    ; out[15] = fdct_round_shift(temp2);
    ; temp1 = step1[0] * cospi_30_64 + step1[7] *  cospi_2_64;
    ; out[1] = fdct_round_shift(temp1);
    DO_BUTTERFLY_DCT d28, d29, d0, d1, cospi_30_64, cospi_2_64, d0, d1, d2, d3

    ; temp1 = step1[3] * -cospi_26_64 + step1[4] *  cospi_6_64;
    ; out[3] = fdct_round_shift(temp1);
    ; temp2 = step1[3] *  cospi_6_64 + step1[4] * cospi_26_64;
    ; out[13] = fdct_round_shift(temp2)
    DO_BUTTERFLY_DCT d8, d9, d6, d7, cospi_6_64, cospi_26_64, d6, d7, d8, d9

    ; temp2 = step1[2] * -cospi_10_64 + step1[5] * cospi_22_64;
    ; out[11] = fdct_round_shift(temp2);
    ; temp1 = step1[2] * cospi_22_64 + step1[5] * cospi_10_64;
    ; out[5] = fdct_round_shift(temp1);
    DO_BUTTERFLY_DCT d10, d11, d4, d5, cospi_22_64, cospi_10_64, d4, d5, d10, d11
    vmov.i16        q9,    q6
    vmov.i16        q15,   q0
    vmov.i16        q13,   q4
    vmov.i16        q11,   q2
    MEND
    ; --------------------------------------------------------------------------
    ; Does the following tasks
    ;   - calculates DST transform for a single pass
    ;   - even rows of outputs are stored in intermediate buffer[r5]
    ;   - odd rows of output are  returned in registers q1,q3,q5,q7,q9,q11,q13
    ;     and q15
    ;   - [r7] is used as intermediate buffer
    ; Inputs
    ;   q0 - q15 : Rows1 to Row16 (8 cols)
    ; Outputs
    ;   q1,q3,...  q15 : all odd from Rows1 to Row15 (8 cols)
    ; touches all q registers, r8, r12, r3, r4, r11
    MACRO
    DST_SINGLE_PASS
    mov             r11,   r5
    ; store rows 13,4,9,6,3,12,1,14 to intermediate buffer
    ; destination buffer is also used as intermediate buffer
    vst1.64         {q11}, [r5],  r6
    vst1.64         {q4},  [r5],  r6
    vst1.64         {q9},  [r5],  r6
    vst1.64         {q6},  [r5],  r6
    vst1.64         {q3},  [r5],  r6
    vst1.64         {q12}, [r5],  r6
    vst1.64         {q1},  [r5],  r6
    vst1.64         {q14}, [r5],  r6

    ; stage 1 for x0,x1,x2,x3,x8,x9,x10,x11

    ; s0 = x0 * cospi_1_64  + x1 * cospi_31_64;
    ; s1 = x0 * cospi_31_64 - x1 * cospi_1_64;
    DO_BUTTERFLY_DST d30, d31, d0, d1, cospi_31_64, cospi_1_64, q4, q11, q6, q9

    ; s8 = x8 * cospi_17_64 + x9 * cospi_15_64;
    ; s9 = x8 * cospi_15_64 - x9 * cospi_17_64;
    DO_BUTTERFLY_DST d14, d15, d16, d17, cospi_15_64, cospi_17_64, q3, q1, q15, q0

    ; x9 = fdct_round_shift(s1 - s9) ; x1 = fdct_round_shift(s1 + s9);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q4, q11, q3, q1, d14, d15, d8, d9

    ; x8 = fdct_round_shift(s0 - s8), ;x0 = fdct_round_shift(s0 + s8);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q6, q9, q15, q0, d16, d17, d12, d13

    ; s2 = x2 * cospi_5_64  + x3 * cospi_27_64;
    ; s3 = x2 * cospi_27_64 - x3 * cospi_5_64;
    DO_BUTTERFLY_DST d26, d27, d4, d5, cospi_27_64, cospi_5_64, q3, q1, q9, q11

    ; s10 = x10 * cospi_21_64 + x11 * cospi_11_64;
    ; s11 = x10 * cospi_11_64 - x11 * cospi_21_64;
    DO_BUTTERFLY_DST d10, d11, d20, d21, cospi_11_64, cospi_21_64, q0, q15, q13, q2

    ; x3 = fdct_round_shift(s3 + s11);,x11 = fdct_round_shift(s3 - s11);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q3, q1, q0, q15, d0, d1, d6, d7

    ; x2 = fdct_round_shift(s2 + s10);,  x10 = fdct_round_shift(s2 - s10);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q9, q11, q13, q2, d4, d5, d2, d3

    ; move  x0,x1,x2 ,x3,x4,x5,x6,x7 to intermediate buffer
    mov             r10,   r7
    vst1.64         {q6},  [r7],  r6
    vst1.64         {q4},  [r7],  r6
    vst1.64         {q1},  [r7],  r6
    vst1.64         {q3},  [r7],  r6

    vst1.64         {q8},  [r7],  r6
    vst1.64         {q7},  [r7],  r6
    vst1.64         {q2},  [r7],  r6
    vst1.64         {q0},  [r7],  r6

    ; stage 1 for x4,x5,x6,x7,x12,x13,x114,x15
    ; load rows 13,4,9,6,3,12,1,14 from intermediate buffer
    mov             r5,    r11
    vld1.64         {q15}, [r5],  r6
    vld1.64         {q0},  [r5],  r6
    vld1.64         {q13}, [r5],  r6
    vld1.64         {q2},  [r5],  r6
    vld1.64         {q7},  [r5],  r6
    vld1.64         {q8},  [r5],  r6
    vld1.64         {q5},  [r5],  r6
    vld1.64         {q10}, [r5],  r6

    ; s4 = x4 * cospi_9_64  + x5 * cospi_23_64;
    ; s5 = x4 * cospi_23_64 - x5 * cospi_9_64;
    DO_BUTTERFLY_DST d30, d31, d0, d1, cospi_23_64, cospi_9_64, q4, q11, q6, q9

    ; s12 = x12 * cospi_25_64 + x13 * cospi_7_64;
    ; s13 = x12 * cospi_7_64  - x13 * cospi_25_64;
    DO_BUTTERFLY_DST d14, d15, d16, d17, cospi_7_64, cospi_25_64, q3, q1, q15, q0

    ;  x4 = fdct_round_shift(s4 + s12);  x12 = fdct_round_shift(s4 - s12);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q4, q11, q3, q1, d14, d15, d8, d9

    ; x5 = fdct_round_shift(s5 + s13), x13 = fdct_round_shift(s5 - s13);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q6, q9, q15, q0, d16, d17, d12, d13

    ; s6 = x6 * cospi_13_64 + x7 * cospi_19_64;
    ; s7 = x6 * cospi_19_64 - x7 * cospi_13_64;
    DO_BUTTERFLY_DST d26, d27, d4, d5, cospi_19_64, cospi_13_64, q3, q1, q9, q11

    ; s14 = x14 * cospi_29_64 + x15 * cospi_3_64;
    ; s15 = x14 * cospi_3_64  - x15 * cospi_29_64;
    DO_BUTTERFLY_DST d10, d11, d20, d21, cospi_3_64, cospi_29_64, q0, q15, q13, q2

    ; x6 = fdct_round_shift(s6 + s14), x14 = fdct_round_shift(s6 - s14);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q3, q1, q0, q15, d0, d1, d6, d7

    ; x7 = fdct_round_shift(s7 + s15),  x15 = fdct_round_shift(s7 - s15);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q9, q11, q13, q2, d4, d5, d2, d3

    mov             r7,    r10
    ; load x8 - x11 after stage 1
    vld1.64         {q5},  [r7],  r6
    vld1.64         {q9},  [r7],  r6
    vld1.64         {q10}, [r7],  r6
    vld1.64         {q11}, [r7],  r6
    ; store x4 - x7 after stage 1
    mov             r7,    r10
    vst1.64         {q8},  [r7],  r6
    vst1.64         {q7},  [r7],  r6
    vst1.64         {q2},  [r7],  r6
    vst1.64         {q0},  [r7],  r6

    ; stage 2 for x8,x9,x10,x11, x12,x13,x14,x15

    ; s8 =    x8 * cospi_4_64   + x9 * cospi_28_64;
    ; s9 =    x8 * cospi_28_64  - x9 * cospi_4_64;
    DO_BUTTERFLY_DST d10, d11, d18, d19, cospi_28_64, cospi_4_64, q2, q0, q7, q8

    ; s12 = - x12 * cospi_28_64 + x13 * cospi_4_64;
    ; s13 =   x12 * cospi_4_64  + x13 * cospi_28_64;
    DO_BUTTERFLY_DST d8, d9, d12, d13, cospi_4_64, cospi_28_64, q5, q9, q13, q15

    ; x8 = fdct_round_shift(s8 + s12);   x12 = fdct_round_shift(s8 - s12);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q7, q8, q5, q9, d10, d11, d14, d15

    ; x9 = fdct_round_shift(s9 + s13);, x13 = fdct_round_shift(s9 - s13);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q2, q0, q13, q15, d30, d31, d4, d5

    ; s10 =   x10 * cospi_20_64 + x11 * cospi_12_64;
    ; s11 =   x10 * cospi_12_64 - x11 * cospi_20_64;
    DO_BUTTERFLY_DST d20, d21, d22, d23, cospi_12_64, cospi_20_64, q9, q6, q4, q0

    ; s14 = - x14 * cospi_12_64 + x15 * cospi_20_64;
    ; s15 =   x14 * cospi_20_64 + x15 * cospi_12_64;
    DO_BUTTERFLY_DST d6, d7, d2, d3, cospi_20_64, cospi_12_64, q8, q13, q10, q11

    ; x10 = fdct_round_shift(s10 + s14);  x14 = fdct_round_shift(s10 - s14);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q4, q0, q8, q13, d16, d17, d8, d9

    ; x11 = fdct_round_shift(s11 + s15);  x15 = fdct_round_shift(s11 - s15);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q9, q6, q10, q11, d20, d21, d18, d19

    ; stage 3 for x8,x9,x10,x11, x12,x13,x14,x15

    ;s12 = x12 * cospi_8_64  + x13 * cospi_24_64;
    ;s13 = x12 * cospi_24_64 - x13 * cospi_8_64;
    DO_BUTTERFLY_DST d14, d15, d4, d5, cospi_24_64, cospi_8_64, q0, q1, q3, q6

    ; s14 = - x14 * cospi_24_64 + x15 * cospi_8_64;
    ; s15 =   x14 * cospi_8_64  + x15 * cospi_24_64
    DO_BUTTERFLY_DST d18, d19, d8, d9, cospi_8_64, cospi_24_64, q11, q13, q7, q2

    ; x12 = fdct_round_shift(s12 + s14);  x14 = fdct_round_shift(s12 - s14);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q3, q6, q11, q13, d22, d23, d26, d27

    ; x13 = fdct_round_shift(s13 + s15),  x15 = fdct_round_shift(s13 - s15);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q0, q1, q7, q2, d14, d15, d4, d5

    ; x8 = s8 + s10;
    ; x9 = s9 + s11;
    ; x10 = s8 - s10;
    ; x11 = s9 - s11;
    DO_BUTTERFLY_NO_COEFFS q5,  q8,  q5, q0
    DO_BUTTERFLY_NO_COEFFS q15, q10, q1, q8

    ; stage 4 for x8,x9,x10,x11, x12,x13,x14,x15
    mov             r5,    r11
    add             r5,    r5,    r6
    vswp.s16        q5,    q0
    ; output[2] = x12 , store output[2] to intermediate buffer
    vst1.64         {q11}, [r5],  r6
    ; s10 = cospi_16_64 * (x10 + x11);
    ; s11 = cospi_16_64 * (- x10 + x11);
    ; x10 = fdct_round_shift(s10);
    ; x11 = fdct_round_shift(s11);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q8, q5, cospi_16_64, d28, d29, d22, d23
    add             r5,    r5,    r6
    ; output[6] = x10 , store output[2] to intermediate buffer
    vst1.64         {q14}, [r5],  r6
    ; s14 = (- cospi_16_64) * (x14 + x15);
    ; s15 = cospi_16_64 * (x14 - x15);
    ; x14 = fdct_round_shift(s14);
    ; x15 = fdct_round_shift(s15);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q2, q13, -cospi_16_64, d18, d19, d16, d17
    add             r5,    r5,    r6
    ; output[10] = x15 , store output[10] to intermediate buffer
    vst1.64         {q8},  [r5],  r6
    add             r5,    r5,    r6
    ; output[14] = x9 , store output[14] to intermediate buffer
    vst1.64         {q1},  [r5],  r6
    vneg.s16        q0,    q0
    vneg.s16        q7,    q7
    ; storing odd rows of ouputs in stack
    vpush           {q7}
    vpush           {q11}
    vpush           {q9}
    vpush           {q0}
    ; load x0 - x7 after stage 1
    mov             r7,    r10
    vld1.64         {q0},  [r7],  r6
    vld1.64         {q1},  [r7],  r6
    vld1.64         {q3},  [r7],  r6
    vld1.64         {q6},  [r7],  r6
    vld1.64         {q11}, [r7],  r6
    vld1.64         {q12}, [r7],  r6
    vld1.64         {q13}, [r7],  r6
    vld1.64         {q14}, [r7],  r6

    ; stage 2 for x0 -x7
    DO_BUTTERFLY_NO_COEFFS q11, q0, q5, q7
    DO_BUTTERFLY_NO_COEFFS q12, q1, q15, q2
    DO_BUTTERFLY_NO_COEFFS q13, q3, q8, q4
    DO_BUTTERFLY_NO_COEFFS q14, q6, q10, q9

    ; stage 3 for x0 - x7

    ; s4 = x4 * cospi_8_64  + x5 * cospi_24_64;
    ; s5 = x4 * cospi_24_64 - x5 * cospi_8_64;
    DO_BUTTERFLY_DST d14, d15, d4, d5, cospi_24_64, cospi_8_64, q0, q1, q3, q6

    ; s6 = - x6 * cospi_24_64 + x7 * cospi_8_64;
    ; s7 =   x6 * cospi_8_64  + x7 * cospi_24_64;
    DO_BUTTERFLY_DST d18, d19, d8, d9, cospi_8_64, cospi_24_64, q11, q13, q7, q2

    ; x4 = fdct_round_shift(s4 + s6),x6 = fdct_round_shift(s4 - s6);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q3, q6, q11, q13, d22, d23, d26, d27
    ; x5 = fdct_round_shift(s5 + s7) x7 = fdct_round_shift(s5 - s7);
    DO_BUTTERFLY_WITHOUT_COEFFS_AND_ROUNDSHIFT q0, q1, q7, q2, d14, d15, d4, d5
    ; x0 = s0 + s2;
    ; x1 = s1 + s3;
    ; x2 = s0 - s2;
    ; x3 = s1 - s3;
    DO_BUTTERFLY_NO_COEFFS q5, q8, q5, q0
    DO_BUTTERFLY_NO_COEFFS q15, q10, q1, q8
    vswp.s16        q5,    q0
    ; stage 4 for x0 -x7
    mov             r5,    r11
    ; output[0] = x0 , store output[0] to intermediate buffer
    vst1.64         {q0},  [r5],  r6
    vneg.s16        q3,    q11       ; output[3] = - x4;

    ; s6 = cospi_16_64 * (x6 + x7);
    ; s7 = cospi_16_64 * (- x6 + x7);
    ; x6 = fdct_round_shift(s6);
    ; x7 = fdct_round_shift(s7);
    DO_BUTTERFLY_SYMMETRIC_COEFFS q2, q13, cospi_16_64, d28, d29, d22, d23
    add             r5,    r5,    r6
    ; output[4] = x6 , store output[4] to intermediate buffer
    vst1.64         {q14}, [r5],  r6
    ; s2 = (- cospi_16_64) * (x2 + x3);
    ; s3 = cospi_16_64 * (x2 - x3);
    ; x2 = fdct_round_shift(s2);
    ; x3 = fdct_round_shift(s3)
    DO_BUTTERFLY_SYMMETRIC_COEFFS q8, q5, -cospi_16_64, d0, d1, d28, d29
    add             r5,    r5,    r6
    ; output[8] = x3 , store output[8] to intermediate buffer
    vst1.64         {q14}, [r5],  r6
    add             r5,    r5,    r6
    ; output[12] = x5 , store output[12] to intermediate buffer
    vst1.64         {q7},  [r5],  r6
    vmov.s16        q7,    q0
    vneg.s16        q15,   q1
    vpop            {q1}
    vpop            {q5}
    vpop            {q9}
    vpop            {q13}
    MEND

    ;---------------------------------------------------------------------------
    ; BLOCK A = rows 1 to 8 cols 1 to 8
    ; BLOCK B = rows 9 to 16 cols 1 to 8
    ; BLOCK C = rows 1 to 8 cols 9 to 16
    ; BLOCK D = rows 9 to 16 cols 9 to 16
    ; DCT_SINGLE_PASS/DST_SINGLE_PASS process 16 rows and 8 columns in a pass
    ; Output buffer is also used as intermediate buffer
    ; Code flow:
    ;  - Call DCT_SINGLE_PASS/DST_SINGLE_PASS for BLOCK A and B
    ;  - Call DCT_SINGLE_PASS/DST_SINGLE_PASS for BLOCK C and D
    ;  - Transpose the intermediate outputs of BLOCK B and D
    ;  - Call DCT_SINGLE_PASS/DST_SINGLE_PASS for BLOCK B and D
    ;  - Transpose the result and store the final output for BLOCK B and D
    ;  - Transpose the intermediate outputs of BLOCK A and C
    ;  - Call DCT_SINGLE_PASS/DST_SINGLE_PASS for BLOCK A and C
    ;  - Transpose the result and store the final output for BLOCK A and C

|vp9_fht16x16_neon| PROC
    cmp             r3,    #0               ; if type = DCT_DCT branch to
                                            ;           vp9_fdct16x16_neon
    beq vp9_fdct16x16_neon
    push            {r4-r12, lr}            ; push registers to stack
    vpush           {d8-d15}
    mov             r9,    r3
    lsl             r2,    r2,    #1         ; r2 = stride * 2
    push            {r2}
    mov             r5,    r0
    ; load 16 rows and 8 columns(1-8)(BLOCK A and B)
    LOAD_INPUT      r5,    r2
    mov             r2,    #32
    lsl             r6,    r2,    #1         ; r3 = stride * 4
    mov             r5,    r1
    ; if stage 1  = DST
    ands            r3,    r9,    #1
    beq dct1
    ; DST STAGE 1
    ; calculate input for pass = 0
    MULTIPLY_BY_4_Q0_TO_Q15
    mov             r5,    r1
    add             r7,    r5,    r2
    bl              dst_single_pass                      ; BLOCK A and B
    b               end_stage_1_col_1_8
dct1
    ; calculate input for pass = 0
    CALC_INPUT_FOR_DCT     r5,    0
    mov             r5,    r1
    add             r7,    r5,    r2
    bl              dct_single_pass
end_stage_1_col_1_8
    mov             r5,    r1
    add             r7,    r5,    r2
    ; store the odd rows to intermediate buffer (BLOCK A and B)
    vst1.64         {q1},  [r7],  r6
    vst1.64         {q3},  [r7],  r6
    vst1.64         {q5},  [r7],  r6
    vst1.64         {q7},  [r7],  r6
    vst1.64         {q9},  [r7],  r6
    vst1.64         {q11}, [r7],  r6
    vst1.64         {q13}, [r7],  r6
    vst1.64         {q15}, [r7],  r6

    add             r5,    r0,    #16
    ; load 16 rows and 8 columns(9 - 16)(BLOCK C and D)
    pop             {r2}
    LOAD_INPUT      r5,    r2
    add             r5,    r1,    #16
    mov             r2,    #32
    ands            r3,    r9,    #1
    beq dct2

    MULTIPLY_BY_4_Q0_TO_Q15
    add             r5,    r1,    #16
    add             r7,    r5,    r2
    bl              dst_single_pass                      ; BLOCK C and D
    b               end_stage_1_col_9_16
dct2
    CALC_INPUT_FOR_DCT     r5,    0
    add             r5,    r1,    #16
    add             r7,    r5,    r2
    bl              dct_single_pass
end_stage_1_col_9_16
    add             r5,    r1,    #16
    add             r7,    r5,    r2

    ; store the top four odd rows (of BLOCK C)
    vst1.64         {q1},  [r7],  r6
    vst1.64         {q3},  [r7],  r6
    vst1.64         {q5},  [r7],  r6
    vst1.64         {q7},  [r7],  r6

    add             r5,    r1,    #16
    add             r5,    r5,    r6,  lsl #2
    ; load even rows(8 -12) cols (9 -16) of BLOCK D
    vld1.64         q8,    [r5],  r6
    vld1.64         q10,   [r5],  r6
    vld1.64         q12,   [r5],  r6
    vld1.64         q14,   [r5],  r6

    ; now registers q8 - q15 have intermediate values after first pass
    ; of rows(9 - 16) and cols (9 -  16)
    TRANSPOSE8X8_Q8_TO_Q15                  ; BLOCK D

    add             r5,    r1,    r6,  lsl #2
    ; load intermediate values of rows(9 - 6) and cols (1-  8) (BLOCK B)
    vld1.64         q0,    [r5],  r2
    vld1.64         q1,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q3,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q5,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q7,    [r5],  r2
    TRANSPOSE8X8_Q0_TO_Q7                   ; BLOCK B
    ands            r3,    r9,    #2
    ; if stage 2  = DST
    ; round and divide by 4 before second pass
    ROUND_SHIFT     hybrid
    beq             dct3
    ; DST STAGE 2
    add             r5,    r1,    r6, lsl #2
    add             r7,    r5,    #16
    mov             r6,    r2
    ; DCT second pass                       ; BLOCK B and D
    bl              dst_single_pass
    b               end_stage_2_col_9_16

    ; DCT_STAGE2
    ; round and divide by 4 before second pass
dct3;

    add             r5,    r1,    r6, lsl #2
    CALC_INPUT_FOR_DCT     r5,    1
    add             r5,    r1,    r6, lsl #2
    add             r7,    r5,    #16
    mov             r6,    r2
    bl              dct_single_pass

end_stage_2_col_9_16
    lsl             r6,    r2,    #1         ; r3 = stride * 4
    add             r5,    r1,    r6, lsl #2
    ; load the even rows from intermediate buffer
    vld1.64         q0,    [r5],  r2         ; BLOCK B and D
    vld1.64         q2,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q8,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q14,   [r5],  r2

    TRANSPOSE8X8_Q0_TO_Q7                    ; BLOCK B and D
    TRANSPOSE8X8_Q8_TO_Q15

    mov             r5,    r1
    add             r5,    r1,    r6, lsl #2
    add             r10,   r5,    #16
    ; store the result to output
    ; rows(8 - 16) and cols (1 -  16)        ; BLOCK B and D
    vst1.64         {q0},  [r5],  r2
    vst1.64         {q1},  [r5],  r2
    vst1.64         {q2},  [r5],  r2
    vst1.64         {q3},  [r5],  r2
    vst1.64         {q4},  [r5],  r2
    vst1.64         {q5},  [r5],  r2
    vst1.64         {q6},  [r5],  r2
    vst1.64         {q7},  [r5],  r2
    vst1.64         {q8},  [r10], r2
    vst1.64         {q9},  [r10], r2
    vst1.64         {q10}, [r10], r2
    vst1.64         {q11}, [r10], r2
    vst1.64         {q12}, [r10], r2
    vst1.64         {q13}, [r10], r2
    vst1.64         {q14}, [r10], r2
    vst1.64         {q15}, [r10], r2
    add             r5,    r1,    #16
    ; load intermediate values of rows(1 - 8) and cols (9 -  18)
    vld1.64         q8,    [r5],  r2         ; BLOCK C
    vld1.64         q9,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q11,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q13,   [r5],  r2
    vld1.64         q14,   [r5],  r2
    vld1.64         q15,   [r5],  r2
    mov             r5,    r1
    ; load intermediate values of rows(1 - 8) and cols (1 -  8)
    vld1.64         q0,   [r5],   r2          ; BLOCK A
    vld1.64         q1,   [r5],   r2
    vld1.64         q2,   [r5],   r2
    vld1.64         q3,   [r5],   r2
    vld1.64         q4,   [r5],   r2
    vld1.64         q5,   [r5],   r2
    vld1.64         q6,   [r5],   r2
    vld1.64         q7,   [r5],   r2

    TRANSPOSE8X8_Q0_TO_Q7                     ; BLOCK A
    TRANSPOSE8X8_Q8_TO_Q15                    ; BLOCK C

    ands            r3,    r9,    #2
    ROUND_SHIFT     hybrid
    beq             dct4
    ; second pass for rows(1 - 16) and cols (1 -  8)
    mov             r5,    r1
    add             r7,    r1,    #16
    add             r5,    r1,    #0
    mov             r6,    r2
    bl              dst_single_pass
    b               end_stage_2_col_1_8

    ; second pass for rows(1 - 16) and cols (1 -  8)
dct4
    mov             r5,    r1
    CALC_INPUT_FOR_DCT     r5,    1
    add             r7,    r1,    #16
    add             r5,    r1,    #0
    mov             r6,    r2
    bl              dct_single_pass
end_stage_2_col_1_8
    lsl             r6,    r2,    #1           ; r3 = stride * 4
    mov             r5,    r1
    ; load the even rows from intermediate buffer
    vld1.64         q0,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q8,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q14,   [r5],  r2

    TRANSPOSE8X8_Q0_TO_Q7                     ; BLOCK A
    TRANSPOSE8X8_Q8_TO_Q15                    ; BLOCK C
    ; store the result to output
    ; rows(1 - 8) and cols (1 -  16)
    mov             r5,    r1
    add             r10,   r5,    #16
    vst1.64         {q0},  [r5],  r2
    vst1.64         {q1},  [r5],  r2
    vst1.64         {q2},  [r5],  r2
    vst1.64         {q3},  [r5],  r2
    vst1.64         {q4},  [r5],  r2
    vst1.64         {q5},  [r5],  r2
    vst1.64         {q6},  [r5],  r2
    vst1.64         {q7},  [r5],  r2
    vst1.64         {q8},  [r10], r2
    vst1.64         {q9},  [r10], r2
    vst1.64         {q10}, [r10], r2
    vst1.64         {q11}, [r10], r2
    vst1.64         {q12}, [r10], r2
    vst1.64         {q13}, [r10], r2
    vst1.64         {q14}, [r10], r2
    vst1.64         {q15}, [r10], r2

    vpop            {d8-d15}
    pop             {r4-r12, pc}
    ENDP

|vp9_fdct16x16_neon| PROC
    push            {r4-r12, lr}          ; push registers to stack
    vpush           {d8-d15}
    mov             r14,   r2
    lsl             r2,    r2,    #1
    push            {r2}
    mov             r5,    r0
    ; load 16 rows and 8 columns(1-8)
    LOAD_INPUT      r5,    r2
    mov             r2,    #32
    lsl             r6,    r2,    #1      ; r3 = stride * 4
    mov             r5,    r1
    ; calculate input for pass = 0
    CALC_INPUT_FOR_DCT     r5,    0
    mov             r5,    r1
    add             r7,    r5,    r2
    ; DCT SINGLE PASS
    bl dct_single_pass
    mov             r5,    r1
    add             r7,    r5,    r2
    ; Store the odd rows to intermediate buffer
    vst1.64         {q1},  [r7],  r6
    vst1.64         {q3},  [r7],  r6
    vst1.64         {q5},  [r7],  r6
    vst1.64         {q7},  [r7],  r6
    vst1.64         {q9},  [r7],  r6
    vst1.64         {q11}, [r7],  r6
    vst1.64         {q13}, [r7],  r6
    vst1.64         {q15}, [r7],  r6
    add             r5,    r0,    #16
    ; load 16 rows and 8 columns(9 - 16)
    pop             {r2}
    LOAD_INPUT      r5,    r2
    mov             r2,    #32
    add             r5,    r1,    #16
    CALC_INPUT_FOR_DCT     r5,    0
    add             r5,    r1,    #16
    add             r7,    r5,    r2
    ; DCT SINGLE PASS
    bl              dct_single_pass
    add             r5,    r1,    #16
    add             r7,    r5,    r2
    ; store the top four odd rows
    vst1.64         {q1},  [r7],  r6
    vst1.64         {q3},  [r7],  r6
    vst1.64         {q5},  [r7],  r6
    vst1.64         {q7},  [r7],  r6
    add             r5,    r1,    #16
    add             r5,    r5,    r6, lsl #2
    ; load even rows(8 -12) cols (9 -16)
    vld1.64         q8,    [r5],  r6
    vld1.64         q10,   [r5],  r6
    vld1.64         q12,   [r5],  r6
    vld1.64         q14,   [r5],  r6
    ; now registers q8 - q15 have intermediate values after first pass
    ; of rows(9 - 16) and cols (9 -  16)
    TRANSPOSE8X8_Q8_TO_Q15
    add             r5,    r1,    r6, lsl #2
    ; load intermediate values of rows(1 - 8) and cols (9 -  16)
    vld1.64         q0,    [r5],  r2
    vld1.64         q1,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q3,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q5,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q7,    [r5],  r2
    TRANSPOSE8X8_Q0_TO_Q7
    ; roundinf before second pass
    ROUND_SHIFT     dct
    add             r5,    r1,    r6, lsl #2
    CALC_INPUT_FOR_DCT     r5,    1
    add             r5,    r1,    r6, lsl #2
    add             r7,    r5,    #16
    mov             r6,    r2
    ; DCT second pass
    bl              dct_single_pass
    lsl             r6,    r2,    #1            ; r3 = stride * 4
    add             r5,    r1,    r6, lsl #2
    ; load the even rows from intermediate buffer
    vld1.64         q0,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q8,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q14,   [r5],  r2

    TRANSPOSE8X8_Q0_TO_Q7
    TRANSPOSE8X8_Q8_TO_Q15

    mov             r5,    r1
    add             r5,    r1,    r6, lsl #2
    add             r10,   r5,    #16
    ; store the result to output
    ;rows(8 - 16) and cols (1 -  16)
    vst1.64         {q0},  [r5],  r2
    vst1.64         {q1},  [r5],  r2
    vst1.64         {q2},  [r5],  r2
    vst1.64         {q3},  [r5],  r2
    vst1.64         {q4},  [r5],  r2
    vst1.64         {q5},  [r5],  r2
    vst1.64         {q6},  [r5],  r2
    vst1.64         {q7},  [r5],  r2
    vst1.64         {q8},  [r10], r2
    vst1.64         {q9},  [r10], r2
    vst1.64         {q10}, [r10], r2
    vst1.64         {q11}, [r10], r2
    vst1.64         {q12}, [r10], r2
    vst1.64         {q13}, [r10], r2
    vst1.64         {q14}, [r10], r2
    vst1.64         {q15}, [r10], r2

    add             r5,    r1,    #16
    ; load intermediate values of rows(9 - 16) and cols (1 -  8)
    vld1.64         q8,    [r5],  r2
    vld1.64         q9,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q11,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q13,   [r5],  r2
    vld1.64         q14,   [r5],  r2
    vld1.64         q15,   [r5],  r2
    mov             r5,    r1
    ; load intermediate values of rows(1 - 8) and cols (1 -  8)
    vld1.64         q0,    [r5],  r2
    vld1.64         q1,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q3,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q5,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q7,    [r5],  r2

    TRANSPOSE8X8_Q0_TO_Q7
    TRANSPOSE8X8_Q8_TO_Q15

    ; second pass for rows(1 - 16) and cols (1 -  8)
    ROUND_SHIFT     dct
    mov             r5,    r1
    CALC_INPUT_FOR_DCT     r5,    1
    add             r7,    r1,    #16
    add             r5,    r1,    #0
    mov             r6,    r2
    ; DCT second pass
    bl              dct_single_pass
    lsl             r6,    r2,    #1             ; r3 = stride * 4
    mov             r5,    r1
    ; load the even rows from intermediate buffer
    vld1.64         q0,    [r5],  r2
    vld1.64         q2,    [r5],  r2
    vld1.64         q4,    [r5],  r2
    vld1.64         q6,    [r5],  r2
    vld1.64         q8,    [r5],  r2
    vld1.64         q10,   [r5],  r2
    vld1.64         q12,   [r5],  r2
    vld1.64         q14,   [r5],  r2

    TRANSPOSE8X8_Q0_TO_Q7
    TRANSPOSE8X8_Q8_TO_Q15
    ; store the result to output
    ; rows(1 - 8) and cols (1 -  16)
    mov             r5,    r1
    add             r10,   r5,    #16
    vst1.64         {q0},  [r5],  r2
    vst1.64         {q1},  [r5],  r2
    vst1.64         {q2},  [r5],  r2
    vst1.64         {q3},  [r5],  r2
    vst1.64         {q4},  [r5],  r2
    vst1.64         {q5},  [r5],  r2
    vst1.64         {q6},  [r5],  r2
    vst1.64         {q7},  [r5],  r2
    vst1.64         {q8},  [r10], r2
    vst1.64         {q9},  [r10], r2
    vst1.64         {q10}, [r10], r2
    vst1.64         {q11}, [r10], r2
    vst1.64         {q12}, [r10], r2
    vst1.64         {q13}, [r10], r2
    vst1.64         {q14}, [r10], r2
    vst1.64         {q15}, [r10], r2
    vpop            {d8-d15}
    pop             {r4-r12, pc}
    ENDP
    ; --------------------------------------------------------------------------
    ; Labeling DCT_SINGLE_PASS and DST_SINGLE_PASS.
    ;  Eventhough DCT_SINGLE_PASS and DST_SINGLE_PASS are coded as macros
    ;  they are not called as macros from the main function to reduce code size
    ;  for better instruction cache performance. Instead we define them under a
    ;  label here and the main function calls this label using Branch and link.
dct_single_pass
    DCT_SINGLE_PASS
    mov             pc,    lr
dst_single_pass
    DST_SINGLE_PASS
    mov             pc,    lr

    END


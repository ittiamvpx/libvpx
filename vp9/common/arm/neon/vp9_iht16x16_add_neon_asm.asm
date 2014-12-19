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

    EXPORT  |vp9_iadst16x16_256_add_neon_single_pass|

    ARM
    REQUIRE8
    PRESERVE8
    AREA ||.text||, CODE, READONLY, ALIGN=2

    AREA     Block, CODE, READONLY

    ;---------------------------------------------------------------------------
    ; Transpose a 8x8 16bit data matrix. Datas are loaded in q8-q15.
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
    ; $diff cannot be same as $ip1 or $ip2
    MACRO
    DO_BUTTERFLY_NO_COEFFS $ip1, $ip2, $sum, $diff
    vsub.s16        $diff, $ip1,  $ip2
    vadd.s16        $sum,  $ip1,  $ip2
    MEND
    ; --------------------------------------------------------------------------
    ; Touches q12, q15 and the input registers
    ; valid output registers are anything but q12, q15, $ip1, $ip2
    ; temp1 and temp2 are Q registers used as temporary registers
    ; temp1 cannot be same as output registers, q12, q15
    ; temp2 cannot be same as input  registers, q12, q15
    MACRO
    DO_BUTTERFLY_SYM_COEFFS  $ip1, $ip2, $ip3, $ip4, $constant, $op1, $op2, $op3, $op4, $temp1, $temp2
    ; generate scalar constants
    mov             r8,     #$constant  & 0xFF00
    add             r8,     #$constant  & 0x00FF
    vdup.16         $op4,   r8
    vmull.s16       q12,    $ip1,  $op4
    vmull.s16       q15,    $ip3,  $op4
    vadd.s32        $temp2, q12,   q15
    vsub.s32        q12,    q12,   q15
    vqrshrn.s32     $op1,   $temp2,  #14
    vqrshrn.s32     $op3,   q12,   #14
    vdup.16         $op4,   r8
    vmull.s16       q12,    $ip2,  $op4
    vmull.s16       q15,    $ip4,  $op4
    vadd.s32        $temp1, q12,   q15
    vsub.s32        q12,    q12,   q15
    vqrshrn.s32     $op2,   $temp1,  #14
    vqrshrn.s32     $op4,   q12,   #14
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
    DO_DUAL_BUTTERFLY_NO_COEFFS $ip1, $ip2, $ip3, $ip4, $op1, $op2, $op3, $op4
    vadd.s32        q12,  $ip1, $ip3
    vadd.s32        q14,  $ip2, $ip4
    vsub.s32        $ip1, $ip1, $ip3
    vsub.s32        $ip2, $ip2, $ip4
    vqrshrn.s32     $op1, q12,  #14
    vqrshrn.s32     $op2, q14,  #14
    vqrshrn.s32     $op3, $ip1, #14
    vqrshrn.s32     $op4, $ip2, #14
    MEND
    ;---------------------------------------------------------------------------
    ; Performs ADST for a 8x16 buffer
    ; This function should be called twice for full 16x16 transform
    ;
    ; vp9_iadst16x16_256_add_neon_single_pass(const int16_t *src,
    ;                                        int16_t *temp_buffer,
    ;                                        int  do_adding,
    ;                                        void *dest,
    ;                                        int dest_stride);
    ;
    ; r0 = const int16_t *src
    ; r1 = int16_t *temp_buffer
    ; r2 = int do_adding (Boolean flag '0' or '1')
    ; r3 = void *dest
    ; r5 = int dest_stride
    ;
    ; The output pointer *dest points to
    ; int16_t* when processed for row transform
    ; uint8_t* when processed for column transform

|vp9_iadst16x16_256_add_neon_single_pass| PROC
    push            {r4-r12, lr}          ; push registers to stack

    ; load 8 rows 16 cols
    mov             r11,   r0
    vld1.64         q0,    [r11]!
    vld1.64         q8,    [r11]!
    vld1.64         q1,    [r11]!
    vld1.64         q9,    [r11]!
    vld1.64         q2,    [r11]!
    vld1.64         q10,   [r11]!
    vld1.64         q3,    [r11]!
    vld1.64         q11,   [r11]!
    vld1.64         q4,    [r11]!
    vld1.64         q12,   [r11]!
    vld1.64         q5,    [r11]!
    vld1.64         q13,   [r11]!
    vld1.64         q6,    [r11]!
    vld1.64         q14,   [r11]!
    vld1.64         q7,    [r11]!
    vld1.64         q15,   [r11]!

    TRANSPOSE8X8_Q0_TO_Q7
    TRANSPOSE8X8_Q8_TO_Q15

    ; store x4,x5,x6,x7,x12,x13,x14 and x15 to intermediate buffer
    mov             r11,   r1
    vst1.64         {q1},  [r11]!
    vst1.64         {q3},  [r11]!
    vst1.64         {q4},  [r11]!
    vst1.64         {q6},  [r11]!
    vst1.64         {q9},  [r11]!
    vst1.64         {q11}, [r11]!
    vst1.64         {q12}, [r11]!
    vst1.64         {q14}, [r11]!

    ; stage 1 for x0,x1,x2,x3,x8,x9,x10 and x11

    ; s0 = x0 * cospi_1_64  + x1 * cospi_31_64;
    ; s1 = x0 * cospi_31_64 - x1 * cospi_1_64;
    DO_BUTTERFLY_DST d30, d31, d0, d1, cospi_31_64, cospi_1_64, q1, q3, q4, q6

    ; s8 = x8 * cospi_17_64 + x9 * cospi_15_64;
    ; s9 = x8 * cospi_15_64 - x9 * cospi_17_64;
    DO_BUTTERFLY_DST d14, d15, d16, d17, cospi_15_64, cospi_17_64, q9, q11, q15, q0

    ; x0 = dct_const_round_shift(s0 + s8);
    ; x8 = dct_const_round_shift(s0 - s8);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q6, q15, q0, d14, d15, d16, d17

    ; x1 = dct_const_round_shift(s1 + s9);
    ; x9 = dct_const_round_shift(s1 - s9);
    DO_DUAL_BUTTERFLY_NO_COEFFS q1, q3, q9, q11, d8, d9, d12, d13

    ; s2 = x2 * cospi_5_64  + x3 * cospi_27_64;
    ; s3 = x2 * cospi_27_64 - x3 * cospi_5_64;
    DO_BUTTERFLY_DST d26, d27, d4, d5, cospi_27_64, cospi_5_64, q1, q3, q9, q11

    ; s10 = x10 * cospi_21_64 + x11 * cospi_11_64;
    ; s11 = x10 * cospi_11_64 - x11 * cospi_21_64;
    DO_BUTTERFLY_DST d10, d11, d20, d21, cospi_11_64, cospi_21_64, q13, q2, q0, q15

    ; x10 = dct_const_round_shift(s2 - s10);
    ; x2  = dct_const_round_shift(s2 + s10);
    DO_DUAL_BUTTERFLY_NO_COEFFS q9, q11, q0, q15, d10, d11, d20, d21

    ; x11 = dct_const_round_shift(s3 - s11);
    ; x3  = dct_const_round_shift(s3 + s11);
    DO_DUAL_BUTTERFLY_NO_COEFFS q1, q3, q13, q2, d18, d19, d22, d23

    ; store for x0,x1,x2,x3,x8,x9,x10 and x11 to
    ; intermediate buffer after stage 1
    vst1.64         {q7},  [r11]!
    vst1.64         {q4},  [r11]!
    vst1.64         {q5},  [r11]!
    vst1.64         {q9},  [r11]!
    vst1.64         {q8},  [r11]!
    vst1.64         {q6},  [r11]!
    vst1.64         {q10}, [r11]!
    vst1.64         {q11}, [r11]!
    ; load x4,x5,x6,x7,x12,x13,x14 and x15 from intermediate buffer
    mov             r11,   r1
    vld1.64         q5,    [r11]!
    vld1.64         q7,    [r11]!
    vld1.64         q0,    [r11]!
    vld1.64         q2,    [r11]!
    vld1.64         q13,   [r11]!
    vld1.64         q15,   [r11]!
    vld1.64         q8,    [r11]!
    vld1.64         q10,   [r11]!

    ; stage 1 for x4,x5,x6,x7,x12,x13,x14 and x15

    ; s4 = x4 * cospi_9_64  + x5 * cospi_23_64;
    ; s5 = x4 * cospi_23_64 - x5 * cospi_9_64;
    DO_BUTTERFLY_DST d30, d31, d0, d1, cospi_23_64, cospi_9_64, q1, q3, q4, q6

    ; s12 = x12 * cospi_25_64 + x13 * cospi_7_64;
    ; s13 = x12 * cospi_7_64  - x13 * cospi_25_64;
    DO_BUTTERFLY_DST d14, d15, d16, d17, cospi_7_64, cospi_25_64, q9, q11, q15, q0

    ; x4  = dct_const_round_shift(s4 + s12);
    ; x12 = dct_const_round_shift(s4 - s12);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q6, q15, q0, d14, d15, d16, d17

    ; x5  = dct_const_round_shift(s5 + s13);
    ; x13 = dct_const_round_shift(s5 - s13);
    DO_DUAL_BUTTERFLY_NO_COEFFS q1, q3, q9, q11, d8, d9, d12, d13

    ; s6 = x6 * cospi_13_64 + x7 * cospi_19_64;
    ; s7 = x6 * cospi_19_64 - x7 * cospi_13_64;
    DO_BUTTERFLY_DST d26, d27, d4, d5, cospi_19_64, cospi_13_64, q1, q3, q9, q11

    ; s14 = x14 * cospi_29_64 + x15 * cospi_3_64;
    ; s15 = x14 * cospi_3_64  - x15 * cospi_29_64;
    DO_BUTTERFLY_DST d10, d11, d20, d21, cospi_3_64, cospi_29_64, q13, q2, q0, q15

    ; x6  = dct_const_round_shift(s6 + s14);
    ; x14 = dct_const_round_shift(s6 - s14);
    DO_DUAL_BUTTERFLY_NO_COEFFS q9, q11, q0, q15, d10, d11, d20, d21

    ; x7  = dct_const_round_shift(s7 + s15);
    ; x15 = dct_const_round_shift(s7 - s15);
    DO_DUAL_BUTTERFLY_NO_COEFFS q1, q3, q13, q2, d18, d19, d22, d23

    mov             r11,   r1
    ; store x4 - x7 to intermediate buffer
    vst1.64         {q7},  [r11]!
    vst1.64         {q4},  [r11]!
    vst1.64         {q5},  [r11]!
    vst1.64         {q9},  [r11]!
    ; load x8 - x11 from intermediate buffer
    add             r11,   r11,   #128
    vld1.64         q0,    [r11]!
    vld1.64         q1,    [r11]!
    vld1.64         q2,    [r11]!
    vld1.64         q3,    [r11]!

    ; stage 2 for x8,x9,x10,x11,x12,x13,x14 and x15

    ; s8 =  x8 * cospi_4_64   + x9 * cospi_28_64;
    ; s9 =  x8 * cospi_28_64  - x9 * cospi_4_64;
    DO_BUTTERFLY_DST d0, d1, d2, d3, cospi_28_64, cospi_4_64, q4, q5, q7, q9

    ; s12 = -x12 * cospi_28_64 + x13 * cospi_4_64;
    ; s13 =  x12 * cospi_4_64  + x13 * cospi_28_64;
    DO_BUTTERFLY_DST d12, d13, d16, d17, cospi_4_64, cospi_28_64, q13, q15, q0, q1

    ; x8  = dct_const_round_shift(s8 + s12);
    ; x12 = dct_const_round_shift(s8 - s12);
    DO_DUAL_BUTTERFLY_NO_COEFFS q7, q9, q13, q15, d12, d13, d16, d17

    ; x9  = dct_const_round_shift(s9 + s13);
    ; x13 = dct_const_round_shift(s9 - s13);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q5, q0, q1, d14, d15, d18, d19

    ; s10 =   x10 * cospi_20_64 + x11 * cospi_12_64;
    ; s11 =   x10 * cospi_12_64 - x11 * cospi_20_64;
    DO_BUTTERFLY_DST d4, d5, d6, d7, cospi_12_64, cospi_20_64, q4, q5, q0, q1

    ; s14 = -x14 * cospi_12_64 + x15 * cospi_20_64;
    ; s15 =  x14 * cospi_20_64 + x15 * cospi_12_64;
    DO_BUTTERFLY_DST d22, d23, d20, d21, cospi_20_64, cospi_12_64, q2, q3, q13, q15

    ; x10 = dct_const_round_shift(s10 + s14);
    ; x14 = dct_const_round_shift(s10 - s14);
    DO_DUAL_BUTTERFLY_NO_COEFFS q0, q1, q2, q3, d22, d23, d20, d21

    ; x15 = dct_const_round_shift(s11 - s15);
    ; x11 = dct_const_round_shift(s11 + s15);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q5, q13, q15, d0, d1, d2, d3

    ; stage 3 for x8,x9,x10,x11,x12,x13,x14 and x15

    ; s12 = x12 * cospi_8_64  + x13 * cospi_24_64;
    ; s13 = x12 * cospi_24_64 - x13 * cospi_8_64;
    DO_BUTTERFLY_DST d16, d17, d18, d19, cospi_24_64, cospi_8_64, q2, q3, q4, q5

    ; s14 = -x14 * cospi_24_64 + x15 * cospi_8_64;
    ; s15 =  x14 * cospi_8_64  + x15 * cospi_24_64;
    DO_BUTTERFLY_DST d2, d3, d20, d21, cospi_8_64, cospi_24_64, q8, q9, q13, q15

    ; x12 = dct_const_round_shift(s12 + s14);
    ; x14 = dct_const_round_shift(s12 - s14);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q5, q8, q9, d20, d21, d2, d3

    ; x13 = dct_const_round_shift(s13 + s15);
    ; x15 = dct_const_round_shift(s13 - s15);
    DO_DUAL_BUTTERFLY_NO_COEFFS q2, q3, q13, q15, d8, d9, d10, d11

    ; x8 = s8 + s10;
    ; x9 = s9 + s11;
    ; x10 = s8 - s10;
    ; x11 = s9 - s11;
    DO_BUTTERFLY_NO_COEFFS q6, q11, q2, q3
    DO_BUTTERFLY_NO_COEFFS q7, q0, q8, q9

    ; stage 4 for x8,x9,x10,x11,x12,x13,x14 and x15

    ; s10 = cospi_16_64 * ( x10 + x11);
    ; s11 = cospi_16_64 * (-x10 + x11);
    ; x10 = dct_const_round_shift(s10);
    ; x11 = dct_const_round_shift(s11);
    DO_BUTTERFLY_SYM_COEFFS  d18, d19, d6, d7, cospi_16_64, d0, d1, d14, d15, q3, q7

    ; s14 = -cospi_16_64 * (x14 + x15);
    ; s15 =  cospi_16_64 * (x14 - x15);
    ; x14 = dct_const_round_shift(s14);
    ; x15 = dct_const_round_shift(s15);
    DO_BUTTERFLY_SYM_COEFFS  d10, d11, d2, d3, -cospi_16_64, d12, d13, d22, d23, q1, q11


    ldr             r5,    [sp,#40]            ; Loads dest_stride
    cmp             r2,    #0
    beq             skip_adding_dest1          ; first pass

    mov             r10,   r3                  ; for laod
    vneg.s16        q2,    q2
    mov             r11,   r3                  ; for store
    add             r10,   r10,    r5
    add             r11,   r11,    r5
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[1]
    vrshr.s16       q2,    q2,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[1]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[2]
    vrshr.s16       q2,    q10,    #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[2]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vneg.s16        q4,    q4
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[5]
    vrshr.s16       q2,    q6,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[5]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[6]
    vrshr.s16       q2,    q0,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[6]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[5]
    vrshr.s16       q2,    q7,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[5]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[6]
    vrshr.s16       q2,    q11,    #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[6]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[5]
    vrshr.s16       q2,    q4,    #6
    vaddw.u8        q2,    q2,    d2           ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[5]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[6]
    vrshr.s16       q2,    q8,    #6
    vaddw.u8        q2,    q2,    d2           ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[6]
    b               stage2_for_x0_x7

skip_adding_dest1
    vneg.s16        q2,    q2
    mov             r11,   r3
    add             r11,   r11,    r5
    vst1.64         {q2},  [r11],  r5          ; output[1] = -x8;
    vst1.64         {q10}, [r11],  r5          ; output[2] =  x12;
    add             r11,   r11,    r5,  lsl #1
    vneg.s16        q4,    q4
    vst1.64         {q6},  [r11],  r5          ; output[5] =  x14;
    vst1.64         {q0},  [r11],  r5          ; output[6] =  x10;
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q7},  [r11],  r5          ; output[9] =  x11;
    vst1.64         {q11}, [r11],  r5          ; output[10] = x15;
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q4},  [r11],  r5          ; output[13] = -x13;
    vst1.64         {q8},  [r11],  r5          ; output[14] =  x9;



stage2_for_x0_x7
    ; stage 2 for x0,x1,x2,x3,x4,x5,x6 and x7

    ; load x4-x7
    mov             r11,   r1
    vld1.64         {q2},  [r11]!
    vld1.64         {q3},  [r11]!
    vld1.64         {q4},  [r11]!
    vld1.64         {q5},  [r11]!
    ; load x0-x3
    add             r11,   r11,     #64
    vld1.64         q12,   [r11]!
    vld1.64         q13,   [r11]!
    vld1.64         q14,   [r11]!
    vld1.64         q15,   [r11]!

    ; x0 = s0 + s4;
    ; x1 = s1 + s5;
    ; x2 = s2 + s6;
    ; x3 = s3 + s7;
    ; x4 = s0 - s4;
    ; x5 = s1 - s5;
    ; x6 = s2 - s6;
    ; x7 = s3 - s7;
    DO_BUTTERFLY_NO_COEFFS q12, q2, q6, q8
    DO_BUTTERFLY_NO_COEFFS q13, q3, q7, q9
    DO_BUTTERFLY_NO_COEFFS q14, q4, q11, q10
    DO_BUTTERFLY_NO_COEFFS q15, q5, q0, q1

    ; stage 3 for x0,x1,x2,x3,x4,x5,x6 and x7

    ; s4 = x4 * cospi_8_64  + x5 * cospi_24_64;
    ; s5 = x4 * cospi_24_64 - x5 * cospi_8_64;
    DO_BUTTERFLY_DST d16, d17, d18, d19, cospi_24_64, cospi_8_64, q2, q3, q4, q5

    ; s6 = -x6 * cospi_24_64 + x7 * cospi_8_64;
    ; s7 =  x6 * cospi_8_64  + x7 * cospi_24_64;
    DO_BUTTERFLY_DST d2, d3, d20, d21, cospi_8_64, cospi_24_64, q8, q9, q13, q15

    ; x4 = dct_const_round_shift(s4 + s6);
    ; x6 = dct_const_round_shift(s4 - s6);
    DO_DUAL_BUTTERFLY_NO_COEFFS q4, q5, q8, q9, d20, d21, d2, d3

    ; x5 = dct_const_round_shift(s5 + s7);
    ; x7 = dct_const_round_shift(s5 - s7);
    DO_DUAL_BUTTERFLY_NO_COEFFS q2, q3, q13, q15, d8, d9, d10, d11

    ; x0 = s0 + s2;
    ; x1 = s1 + s3;
    ; x2 = s0 - s2;
    ; x3 = s1 - s3;
    DO_BUTTERFLY_NO_COEFFS q6, q11, q2, q3
    DO_BUTTERFLY_NO_COEFFS q7, q0, q8, q9

    ; stage 4 for for x0,x1,x2,x3,x4,x5,x6 and x7

    ; s6 = cospi_16_64 * (x6 + x7);
    ; s7 = cospi_16_64 * (- x6 + x7);
    ; x6 = dct_const_round_shift(s6);
    ; x7 = dct_const_round_shift(s7);
    DO_BUTTERFLY_SYM_COEFFS  d10, d11, d2, d3, cospi_16_64, d0, d1, d14, d15, q5, q7

    ; s2 = -cospi_16_64 * (x2 + x3);
    ; s3 =  cospi_16_64 * (x2 - x3);
    ; x2 = dct_const_round_shift(s2);
    ; x3 = dct_const_round_shift(s3);
    DO_BUTTERFLY_SYM_COEFFS  d18, d19, d6, d7, -cospi_16_64, d12, d13, d22, d23, q3, q11

    cmp             r2,    #0
    beq             skip_adding_dest2

    mov             r10,   r3                  ; for load
    vneg.s16        q10,   q10
    mov             r11,   r3                  ; for store
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[0]
    vrshr.s16       q2,    q2,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[0]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[3]
    vrshr.s16       q2,    q10,    #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[3]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[4]
    vrshr.s16       q2,    q0,    #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[4]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[7]
    vrshr.s16       q2,    q6,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[3]
    vld1.64         {d2},  [r10],  r5          ; load destination data,output[8]
    vrshr.s16       q2,    q11,    #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[8]
    vneg.s16        q8,    q8
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destinatin data,output[11]
    vrshr.s16       q2,    q7,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[1]
    vld1.64         {d2},  [r10],  r5          ; load destinatin data,output[12]
    vrshr.s16       q2,    q4,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[12]
    add             r10,   r10,    r5,  lsl #1
    add             r11,   r11,    r5,  lsl #1
    vld1.64         {d2},  [r10],  r5          ; load destinatin data,output[15]
    vrshr.s16       q2,    q8,     #6
    vaddw.u8        q2,    q2,     d2          ; + dest[j * dest_stride + i]
    vqmovun.s16     d2,    q2                  ; clip pixel
    vst1.64         {d2},  [r11],  r5          ; store the data, output[15]
    b               endfunction

skip_adding_dest2
    vneg.s16        q10,   q10
    mov             r11,   r3
    vst1.64         {q2},  [r11],  r5          ; output[0] =  x0;
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q10}, [r11],  r5          ; output[3] = -x4;
    vst1.64         {q0},  [r11],  r5          ; output[4] =  x6;
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q6},  [r11],  r5          ; output[7] =  x2;
    vst1.64         {q11}, [r11],  r5          ; output[8] =  x3;
    vneg.s16        q8,    q8
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q7},  [r11],  r5          ; output[11] =  x7;
    vst1.64         {q4},  [r11],  r5          ; output[12] =  x5;
    add             r11,   r11,    r5,  lsl #1
    vst1.64         {q8},  [r11],  r5          ; output[15] = -x1;

endfunction
    pop             {r4-r12, pc}
    ENDP

    END

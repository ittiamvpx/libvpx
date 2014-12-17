;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;
    EXPORT  |vp9_fht4x4_neon|
    EXPORT  |vp9_fdct4x4_neon|
    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

    ; Parallel 1D DCT on all the columns of a 4x4 16 bits data matrix which are
    ; loaded in d16-d19. d0 must contain cospi_8_64. d1 must contain
    ; cospi_16_64. d2 must contain cospi_24_64. The output will be stored back
    ; into d16-d19 registers. This macro will touch registers q10-q15 and use
    ; them as buffer during calculation.
    MACRO
    FDCT4x4_1D
    ; stage 1
    vaddl.s16    q12, d16, d19   ; step[0] = input[0] + input[3]
    vaddl.s16    q13, d17, d18   ; step[1] = input[1] + input[2]
    vsubl.s16    q14, d17, d18    ; step[2] = input[1] - input[2]
    vsubl.s16   q15, d16, d19   ; step[3] = input[0] - input[3]

    vadd.s32    q10, q12, q13    ; step[0] + step[1]
    vsub.s32    q11, q12, q13    ; step[0] - step[1]

    vmovl.s16     q8 , d1            ; Get cospi_16_64 into 32 bit
    vmovl.s16    q9 , d2            ; Get cospi_24_64 into 32 bit
    vmovl.s16     q12, d0            ; Get cospi_8_64 into 32 bit

    vmul.s32    q10, q10, q8    ; temp1 = (step[0] + step[1]) * cospi_16_64
    vmul.s32    q11, q11, q8    ; temp2 = (step[0] - step[1]) * cospi_16_64

    vmul.s32    q13, q15, q12    ; step[3] * cospi_8_64
    vmul.s32    q15, q15, q9    ; step[3] * cospi_24_64
    vmla.s32    q13, q14, q9    ; temp1 =  step[2] * cospi_24_64 +
    ;                                      step[3] * cospi_8_64;
    vmls.s32    q15, q14, q12    ; temp2 = -step[2] * cospi_8_64 +
                                ;          step[3] * cospi_24_64;

    ; dct_const_round_shift
    vqrshrn.s32 d16, q10, #14
    vqrshrn.s32 d17, q13, #14
    vqrshrn.s32 d18, q11, #14
    vqrshrn.s32 d19, q15, #14

    MEND

    ; Parallel 1D FADST on all the columns of a 4x4 16bits data matrix which
    ; loaded in d16-d19. d3 must contain sinpi_1_9. d4 must contain sinpi_2_9.
    ; d5 must contain sinpi_4_9. d6 must contain sinpi_3_9. The output will be
    ; stored back into d16-d19 registers. This macro will touch q11,q12,q13,
    ; q14,q15 registers and use them as buffer during calculation.
    MACRO
    FADST4x4_1D
    ; stage 1 variables will not have any prefix
    ; Stage 2 variables will be prfixed by ii
    ; stage 3 variables will be prefixed by iii

    ;stage 1
    vmull.s16   q10, d3, d16    ; s0 = sinpi_1_9 * x0
    vmull.s16   q11, d5, d16    ; s1 = sinpi_4_9 * x0
    vmull.s16   q12, d4, d17    ; s2 = sinpi_2_9 * x1
    vmull.s16   q13, d3, d17    ; s3 = sinpi_1_9 * x1
    vmull.s16   q14, d6, d18    ; s4 = sinpi_3_9 * x2
    vmull.s16   q15, d5, d19    ; s5 = sinpi_4_9 * x3

    ;stage 1 & stage 2
    vadd.s32    q10, q10, q12    ; s0 + s2
    vadd.s32    q10, q10, q15    ; iix0 = s0 + s2 + s5
    vmull.s16     q12, d4 , d19    ; s6   = sinpi_2_9 * x3

    vaddl.s16    q15, d16, d17    ; x0 + x1
    vsubw.s16    q15, q15, d19    ; s7 = x0 + x1 - x3
    vmovl.s16    q8 , d6            ; make sinpi_3_9 into 32 bit
    vmul.s32    q15, q8,  q15    ; iix1 = sinpi_3_9 * s7

    vsub.s32    q11, q11, q13    ; s1 - s3
    vadd.s32    q11, q11, q12    ; iix2 = s1 - s3 + s6
    ;q14                        ; iix3 = s4;

    ;stage 2
    vadd.s32    q12, q10, q14   ; iiis0 = iix0 + iix3
    ;q15                        ; iiis1 = iix1
    vsub.s32    q13, q11, q14    ; iiis2 = iix2 - iix3
    vsub.s32    q11, q11, q10    ; iiix2 - iix0
    vadd.s32    q11, q14        ; iiis3 = iix2 - iix0 + iix3

    ; dct_const_round_shift
    vqrshrn.s32 d16, q12, #14    ;fdct_round_shift(s0);
    vqrshrn.s32 d17, q15, #14    ;fdct_round_shift(s1);
    vqrshrn.s32 d18, q13, #14    ;fdct_round_shift(s2);
    vqrshrn.s32 d19, q11, #14    ;fdct_round_shift(s3);

    MEND

    ; Generate cosine constants in d6 - d8 for the DCT
    MACRO
    GENERATE_COSINE_CONSTANTS
    ; cospi_8_64 = 15137 = 0x3b21
    mov         r0, #0x3b00
    add         r0, #0x21
    ; cospi_16_64 = 11585 = 0x2d41
    mov         r3, #0x2d00
    add         r3, #0x41
    ; cospi_24_64 = 6270 = 0x187e
    mov         r12, #0x1800
    add         r12, #0x7e

    ; generate constant vectors
    vdup.16     d0, r0          ; duplicate cospi_8_64
    vdup.16     d1, r3          ; duplicate cospi_16_64
    vdup.16     d2, r12         ; duplicate cospi_24_64
    MEND

    ; Generate sine constants in d1 - d4 for the FADST.
    MACRO
    GENERATE_SINE_CONSTANTS
    ; sinpi_1_9 = 5283 = 0x14A3
    mov         r0, #0x1400
    add         r0, #0xa3
    ; sinpi_2_9 = 9929 = 0x26C9
    mov         r3, #0x2600
    add         r3, #0xc9
    ; sinpi_4_9 = 15212 = 0x3B6C
    mov         r12, #0x3b00
    add         r12, #0x6c

    ; generate constant vectors
    vdup.16     d3, r0          ; duplicate sinpi_1_9

    ; sinpi_3_9 = 13377 = 0x3441
    mov         r0, #0x3400
    add         r0, #0x41

    vdup.16     d4, r3          ; duplicate sinpi_2_9
    vdup.16     d5, r12         ; duplicate sinpi_4_9
    vdup.16     q3, r0          ; duplicate sinpi_3_9
    MEND

    ; Transpose a 4x4 16bits data matrix. Data is loaded in d16-d19.
    MACRO
    TRANSPOSE4X4
    vtrn.16     d16, d17
    vtrn.16     d18, d19
    vtrn.32     q8,  q9
    MEND

    ; Adds 1 to first element of a vector if its non zero
    ; The values must be in d16, will touch q10
    MACRO
    DC_ADD_ONE
    vmov.s16    d20, #0
    vcgt.u16    d21, d16, d20
    vshl.u64    d21, d21, #63
    vshr.u64    d21, d21, #63
    vadd.s16    d16, d16, d21
    MEND

    ; Checks if all elements of in d16-d19 are zero and updates flag
    ; Touches q15,r2
    MACRO
    CHECK_ALL_ZERO
    vorr.s16    q15, q8,  q9
    vorr.s16    d30, d30, d31
    vpmax.u32   d30, d30, d30
    vmov.s32    r2,  d30[0]
    cmp         r2,  #0
    MEND

    AREA     Block, CODE, READONLY ; name this block of code

;void vp9_fht4x4_neon(const int16_t *input, tran_low_t *output,
;                 int stride, int tx_type)
;
; r0  const int16_t    input
; r1  tran_low_t      *output
; r2  int              stride
; r3  int              tx_type)
; This function will handle tx_type of 0,1,2,3.
|vp9_fht4x4_neon| PROC
    vpush       {d8-d15}

    ;Multiply stride by 2 to account for 2 byte data type
    lsl         r2,    r2,    #1

    ; load the inputs into d16-d19
    vld1.s16    {d16}, [r0],  r2
    vld1.s16    {d17}, [r0],  r2
    vld1.s16    {d18}, [r0],  r2
    vld1.s16    {d19}, [r0]     ;no post increment

    ; Check if matrix is all zero
    ; fast4 has an all zero check. Since transform donot make any coefficient
    ; zero that check is valid at any stage
    ; NOTE r2 is over written at this point
    CHECK_ALL_ZERO
    beq return_output_zero

    ; Do scaling by 4 for input
    vqshl.s16   q8,    q8,    #4
    vqshl.s16   q9,    q9,    #4

    ; Add one to DC value if not zero
    DC_ADD_ONE

    ; decide the type of transform
    cmp            r3,    #0
    ;This macro call may be replace by function call to vp9_fdct4x4_neon
    beq            fdct_fdct
    cmp         r3,    #2
    beq         fdct_fadst
    cmp         r3,    #3
    beq         fadst_fadst

fadst_fdct
    ; generate constants
    GENERATE_COSINE_CONSTANTS
    GENERATE_SINE_CONSTANTS

    ; first transform rows
    FADST4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; then transform columns
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    b end_vp9_fht4x4_neon

fdct_fdct
    ; generate constants
    GENERATE_COSINE_CONSTANTS

    ; first transform rows
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; then transform columns
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    b end_vp9_fht4x4_neon

fdct_fadst
    ; generate constants
    GENERATE_COSINE_CONSTANTS
    GENERATE_SINE_CONSTANTS

    ; first transform rows
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; then transform columns
    FADST4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    b end_vp9_fht4x4_neon

fadst_fadst
    ; generate constants
    GENERATE_SINE_CONSTANTS

    ; first transform rows
    FADST4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; then transform columns
    FADST4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

end_vp9_fht4x4_neon
    ; (temp_out[j] + 1) >> 2;
    vrshr.s16   q8, q8, #1
    vrshr.s16   q9, q9, #1

    vshr.s16    q8, q8, #1
    vshr.s16    q9, q9, #1

    ; do the stores
    vst1.s16    {q8-q9}, [r1]

    vpop        {d8-d15}
    bx          lr

return_output_zero
    vmov.s32    q8,   #0
    vst1.s16    {q8}, [r1]!
    vst1.s16    {q8}, [r1]!

    vpop        {d8-d15}
    bx          lr

    ENDP  ; |vp9_fht4x4_neon|




;void vp9_fdct4x4_neon(const int16_t *input, tran_low_t *output, int stride)
;
; r0  const int16_t *input
; r1  tran_low_t    *output
; r2  int            stride)

|vp9_fdct4x4_neon| PROC

    ; The 2D transform is done with two passes which are actually pretty
    ; similar. We first transform the columns. Then we transpose the matrix
    ; and do another column transform. Finally we transpose the output
    vpush       {d8-d15}

    ; Multiply stride by 2 to account for 2byte datatype
    lsl         r2,     r2,   #1

    ; load the inputs into d16-d19
    vld1.s16    {d16}, [r0],  r2
    vld1.s16    {d17}, [r0],  r2
    vld1.s16    {d18}, [r0],  r2
    vld1.s16    {d19}, [r0]                    ;no post increment

    vshl.s16    q8,    q8,    #4
    vshl.s16    q9,    q9,    #4

    ; transpose the input data
    DC_ADD_ONE

    ; generate constants
    GENERATE_COSINE_CONSTANTS

    ; first transform rows
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; then transform columns
    FDCT4x4_1D

    ; transpose the matrix
    TRANSPOSE4X4

    ; (temp_out[j] + 1) >> 2;
    vrshr.s16   q8,    q8,    #1
    vrshr.s16   q9,    q9,    #1

    vshr.s16    q8,    q8,    #1
    vshr.s16    q9,    q9,    #1

    ; do the stores
    vst1.s16    {q8-q9}, [r1]

    vpop        {d8-d15}
    bx          lr
    ENDP  ; |vp9_fdct4x4_neon|
    END

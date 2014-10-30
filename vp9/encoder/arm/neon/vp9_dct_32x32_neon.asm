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


    EXPORT  |vp9_fdct32x32_rd_neon|
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
    GET_CONST $dst, $const, $tmp
    ; Generate the scalar constant
    mov             $tmp,   #$const  &  0xFF00
    add             $tmp,   #$const  &  0x00FF
    ; Copy the constant to neon register
    vdup.s16        $dst,   $tmp
    MEND

    ;--------------------------------------------------------------------------
    ; Halve values
    ; Halves the given two values, with rounding
    ;  $reg1     : neon register containing 1st value to be halved
    ;  $reg2     : neon register containing 2nd value to be halved
    ;  $tmp1     : Scratch register 1
    ;  $tmp2     : Scratch register 2
    ;  None of the input arguments will be preserved
    MACRO
    HALF_ROUND_SHIFT        $reg1,  $reg2,  $tmp1,  $tmp2
    ;output = (input + 1 + (input < 0)) >> 2;
    
    ; Get the sign bit of both registers
    vshr.u16        $tmp1,  $reg1,  #15
    vshr.u16        $tmp2,  $reg2,  #15
    
    ; Add the sign bit, also move 1
    vadd.s16        $reg1,  $reg1,  $tmp1
    vmov.s16        $tmp1,  #1
    vadd.s16        $reg2,  $reg2,  $tmp2
    
    ; Add 1 to both registers
    vadd.s16        $reg1,  $reg1,  $tmp1
    vadd.s16        $reg2,  $reg2,  $tmp1
    
    ; Right shift by 2
    vshr.s16        $reg1,  $reg1,  #2
    vshr.s16        $reg2,  $reg2,  #2
    MEND

    ;--------------------------------------------------------------------------
    ; Load Values, transpose and write back
    ; Loads two 8x8 blocks, transpose each one and write to memory
    ;   $src1 : address to load first 8x8 matrix from
    ;   $src2 : address to load second 8x8 matrix from
    ;   $dst1 : address to write first 8x8 matrix to
    ;   $dst2 : address to write second 8x8 matrix to
    ; Assumes a stride of 64 for both src and dst
    ; Overwrites neon registers q0-q15
    ; None of the input arguments except $strd will be preserved
    MACRO
    TRANSPOSE_TWO_8x8       $src1,  $src2,  $dst1,  $dst2,  $strd
   
    vld1.s16        {q8},   [$src2],$strd
    vld1.s16        {q0},   [$src1],$strd
   
    vld1.s16        {q9},   [$src2],$strd
    vld1.s16        {q1},   [$src1],$strd
   
    vld1.s16        {q10},  [$src2],$strd
    vld1.s16        {q2},   [$src1],$strd
   
    vld1.s16        {q11},  [$src2],$strd
    vld1.s16        {q3},   [$src1],$strd
   
    vld1.s16        {q12},  [$src2],$strd
    vld1.s16        {q4},   [$src1],$strd
   
    vld1.s16        {q13},  [$src2],$strd
    vld1.s16        {q5},   [$src1],$strd
   
    vld1.s16        {q14},  [$src2],$strd
    vld1.s16        {q6},   [$src1],$strd
   
    vld1.s16        {q15},  [$src2],$strd
    vld1.s16        {q7},   [$src1],$strd
   
    ; transpose the two 8x8 16bit data matrices.
    vswp            d17,    d24
    vswp            d23,    d30
    vswp            d21,    d28
    vswp            d19,    d26
    vswp            d1,     d8
    vswp            d7,     d14
    vswp            d5,     d12
    vswp            d3,     d10
    vtrn.32         q8,     q10
    vtrn.32         q9,     q11
    vtrn.32         q12,    q14
    vtrn.32         q13,    q15
    vtrn.32         q0,     q2
    vtrn.32         q1,     q3
    vtrn.32         q4,     q6
    vtrn.32         q5,     q7
    vtrn.16         q8,     q9
    vtrn.16         q10,    q11
    vtrn.16         q12,    q13
    vtrn.16         q14,    q15
    vtrn.16         q0,     q1
    vtrn.16         q2,     q3
    vtrn.16         q4,     q5
    vtrn.16         q6,     q7
   
    vst1.s16        {q8},   [$dst2],$strd
    vst1.s16        {q0},   [$dst1],$strd
   
    vst1.s16        {q9},   [$dst2],$strd
    vst1.s16        {q1},   [$dst1],$strd
   
    vst1.s16        {q10},  [$dst2],$strd
    vst1.s16        {q2},   [$dst1],$strd
   
    vst1.s16        {q11},  [$dst2],$strd
    vst1.s16        {q3},   [$dst1],$strd
   
    vst1.s16        {q12},  [$dst2],$strd
    vst1.s16        {q4},   [$dst1],$strd
   
    vst1.s16        {q13},  [$dst2],$strd
    vst1.s16        {q5},   [$dst1],$strd
   
    vst1.s16        {q14},  [$dst2],$strd
    vst1.s16        {q6},   [$dst1],$strd
   
    vst1.s16        {q15},  [$dst2],$strd
    vst1.s16        {q7},   [$dst1],$strd
    MEND
   
    ; --------------------------------------------------------------------------
    ; Load from intermediate buffer
    ; Loads a particular row from intermediate buffer
    ;    $row_1  : Row no. from which first data is to be loaded
    ;    $row_2  : Row no. from which second data is to be loaded
    ;    $dst_1  : Register to store data from $row_1
    ;    $dst_2  : Register to store data from $row_2
    ; Assumes r3, contains the address of the buffer
    MACRO
    LOAD_FROM_INTERMEDIATE    $row_1, $row_2, $dst_1, $dst_2
    ;Load first value
    add             r3,     #($row_1 * 16)
    vld1.s16        {$dst_1},[r3]
    ;Load Second value
    add             r3,     #($row_2 - $row_1) * 16
    vld1.s16        {$dst_2},[r3]
    ;Restore pointer
    sub             r3,     #($row_2 * 16)
    MEND
   
    ; --------------------------------------------------------------------------
    ; Store to intermediate buffer
    ; Store two registers to specified rows of intermediate buffer
    ;    $row_1  : Row no. to which first data is to be stored
    ;    $row_2  : Row no. to which second data is to be stored
    ;    $src_1  : Source register for $row_1
    ;    $src_2  : Source register for $row_2
    ; Assumes r3, contains the address of the buffer
    MACRO
    STORE_TO_INTERMEDIATE     $row_1, $row_2, $src_1, $src_2
   
    ; address calculation with proper stride and loading
    add             r3,     #($row_1 * 16)
    vst1.s16        {$src_1},[r3]
    add             r3,     #($row_2 - $row_1) * 16
    vst1.s16        {$src_2},[r3]
    sub             r3,     #($row_2 * 16)
    MEND
   
    ; --------------------------------------------------------------------------
    ; Store to output_buffer
    ; Store a particular register to output buffer
    ; $row1  : Column to which first data is to the stored
    ; $row2  : Column to which second data is to be stored
    ; $src1  : Register to store data for $row1
    ; $src2  : Register to store data for $row2
    ; $sub_band1 : sub_band no. corresponding to $row1
    ; $sub_band2 : sub_band no. corresponding to $row2
    ; For explanation of sub_bands, see Memory layout
    ; Assumes r1, contains the address of the buffer
    MACRO
    STORE_TO_OUTPUT $row1,  $row2,  $src1,  $src2,  $sub_band1, $sub_band2

    ;check for pass 0
    cmp             r5,     #1
    ;Store first value
    addeq           r1,     #($row1 * 64)
    addne           r1,     #($sub_band1 * 16 + ($row1 - $sub_band1 * 8) * 64)
    vst1.s16        {$src1},[r1]
    ;Store second value
    addeq           r1,     #($row2 - $row1) * 64
    addne           r1,     #($sub_band2 * 16 + ($row2 - $sub_band2 * 8) * 64) - ($sub_band1 * 16 + ($row1 - $sub_band1 * 8) * 64)
    vst1.s16        {$src2},[r1]
    ;Store pointer
    subeq           r1,     #($row2 * 64)
    subne           r1,     #($sub_band2 * 16 + ($row2 - $sub_band2 * 8) * 64)
   
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
    ; Computes butterfly. Cosine/Sine coefficients is symmetric/same
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
   
    ;Butterfly for $src1 $src3
    vaddl.s16       $tmp3,  $src1,  $src3
    vsubl.s16       $tmp2,  $src1,  $src3
   
    vmul.s32        $tmp3,  $tmp3,  $scal
    vmul.s32        $tmp2,  $tmp2,  $scal
   
    vrshrn.s32      $src3,  $tmp3,  #14
    vrshrn.s32      $dst1,  $tmp2,  #14
   
    ;Butterfly for $src2 $src4
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
    ; $reg1   : Source and destination for Butterfly (64 bit)
    ; $reg2   : Source and destination for butterfly (64 bit)
    ; $reg3   : Source and destination for Butterfly (64 bit)
    ; $reg4   : Source and destination for butterfly (64 bit)
    ; $scal1  : Scaling value 1                      (64 bit)
    ; $scal2  : Scaling value 2                      (64 bit)
    ; $tmp1, $tmp2, $tmp3 : temporary registers      (128 bit)
    ;
    ; Sreg1 = ( $reg1 * $scal1 + $reg2 * $scal2) >> 14 (rounded)
    ; Sreg2 = ( $reg3 * $scal1 + $reg4 * $scal2) >> 14 (rounded)
    ; Sreg3 = (-$reg1 * $scal2 + $reg2 * $scal1) >> 14 (rounded)
    ; Sreg4 = (-$reg3 * $scal2 + $reg4 * $scal1) >> 14 (rounded)
    MACRO
    DO_BUTTERFLY_STD    $reg1,  $reg2,  $reg3,  $reg4,  $scal1, $scal2, $tmp1,  $tmp2,  $tmp3
   
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
   
   
    ; --------------------------------------------------------------------------
    ; Memory layout
    ; --------------------------------------------------------------------------
    ; We use 3 buffers, input, output and intermediate buffer
    ;
    ; Input buffer
    ;   Input buffer will be a 32x32 buffer with a stride
    ;   Input buffer is used only in first pass, values are just read from it.
    ;
    ; Output buffer
    ;   Output buffer will be a 32x32 buffer
    ;   Output buffer will be used to write outputs in both the passes,
    ;       hence in the second pass it will be used as the input buffer too.
    ;   The structure of reading and writing into output buffer are different
    ;       in different passes.
    ;   In first pass, one 32x8 band is processed at a time. The macros
    ;       for loading and storing will use the row numbers directly to load.
    ;   In second pass, one 8x32 band is processed at a time. The input is
    ;       organized in columns and hence it will have to transposed before
    ;       processing. For the ease of transpose, the 8x32 band is further
    ;       divided into four 8x8 sub-bands. Hence
    ;       - sub-band 0 will correspond to output[0]  to output[7]
    ;       - sub-band 1 will correspond to output[8]  to output[15]
    ;       - sub-band 2 will correspond to output[16] to output[23]
    ;       - sub-band 3 will correspond to output[24] to output[31]
    ;
    ; intermediate buffer
    ;   intermediate buffer is the scratch buffer, with size of 32x8

;void vp9_fdct32x32_rd_c(const int16_t *input, tran_low_t *out, int stride) {
;
;   r0  int16_t    *input,
;   r1  tran_low_t *out,
;   r2  int stride
; loop counters
;   r4  band loop counter
;   r5  pass loop counter
;   r8  sub-band loop counter
;Scratch
;   r6  scratch
;   r7  scratch
;   r9  scratch
;   r10 scratch

|vp9_fdct32x32_rd_neon| PROC
    ; This function does dct32x32 transform.
    ;
    ; This 2-D transform is done by performing a 1-D transform in two passes
    ; The 1-D transform is always performed on columns
    ;  - In the first pass the data is used as is.
    ;  - In the second pass the input is transposed first and then used for
    ;    transforms
    ;  - The final output is transposed again
    ;
    ; Each pass is composed of 4 loops, with each iteration processing 32x8 band
    ; In each iteration the processing is done in blocks A, B, C and D.
    ;   Block A corresponds to row 0-3
    ;   Block B corresponds to row 4-7
    ;   Block C corresponds to row 8-15
    ;   Block D corresponds to row 16-31

    push            {r4-r11}
    vpush           {d8-d15}

    ; Double the stride to accommodate 16 bit input
    lsl             r2,     r2, #1
    ; Initialize a negative input stride
    neg             r9,     r2
    
    ;Decrement stack by 512 -> 32x16 for temporary space
    sub             sp,     #512

    ; r5 = 1 : Column transform pass 1
    ; r5 = 0 : Row transform pass 2
    mov             r5,     #1
    
dct32_pass_loop

    ; Each pass processes a band size 32x8 once, hence 4 such bands
    mov             r4,     #4
dct32_bands_loop

    ; Set up intermediate buffer, sp-8*32
    mov             r3,     sp
    ; if first pass
    cmp             r5,     #1
    ; Get address of 31st row in current 32x8 band
    addeq           r6,     r0, r2, lsl #5
    subeq           r6,     r6, r2
    ; Get pointer to 4th 8x8 sub-band of current 8x32 band
    addne           r6,     r0, #48
    ; Get address of last row of intermediate buffer of size 8x32 ->16*31
    add             r7,     r3, #496
    ; The transpose is done for a sub-band of 2 8x8 coefficients at a time hence
    ; we need 2 passes
    mov             r8,     #2
dct32_subband_pair_loop

    ; Load two 8x8 sub bands. The matrices must be complementary
    ; if coeff i is in a matrix 31-i should be in other. Load first one
    ; into q0-q7 and the second one into q15-q8. There is an input stride too

    vld1.s16        {q15},  [r6],   r9
    vld1.s16        {q0},   [r0],   r2
                                            
    vld1.s16        {q14},  [r6],   r9
    vld1.s16        {q1},   [r0],   r2
                                            
    vld1.s16        {q13},  [r6],   r9
    vld1.s16        {q2},   [r0],   r2
                                            
    vld1.s16        {q12},  [r6],   r9
    vld1.s16        {q3},   [r0],   r2
                                            
    vld1.s16        {q11},  [r6],   r9
    vld1.s16        {q4},   [r0],   r2
                                            
    vld1.s16        {q10},  [r6],   r9
    vld1.s16        {q5},   [r0],   r2
                                            
    vld1.s16        {q9},   [r6],   r9
    vld1.s16        {q6},   [r0],   r2
                                            
    vld1.s16        {q8},   [r6],   r9
    vld1.s16        {q7},   [r0],   r2

    ; In the first pass we need to scale coeffs by 4
    cmp             r5, #1
    bne             no_pass1_preprocess
    ;temp_in[j] = input[j * stride + i]* 4;
    vshl.s16        q0,     q0,     #2
    vshl.s16        q1,     q1,     #2
    vshl.s16        q2,     q2,     #2
    vshl.s16        q3,     q3,     #2
    vshl.s16        q4,     q4,     #2
    vshl.s16        q5,     q5,     #2
    vshl.s16        q6,     q6,     #2
    vshl.s16        q7,     q7,     #2
    vshl.s16        q8,     q8,     #2
    vshl.s16        q9,     q9,     #2
    vshl.s16        q10,    q10,    #2
    vshl.s16        q11,    q11,    #2
    vshl.s16        q12,    q12,    #2
    vshl.s16        q13,    q13,    #2
    vshl.s16        q14,    q14,    #2
    vshl.s16        q15,    q15,    #2
no_pass1_preprocess

    ; In the second pass, we need to transpose the coefficents and
    ;   adjust the magnitude as below
    ;output[j * 32 + i] = (temp_out[j] + 1 + (temp_out[j] > 0)) >> 2;
    cmp             r5,     #0
    beq             pass2_preprocess
back_from_pass2_preprocess

    ; --------------------------------------------------------------------------
    ; STAGE 1, BLOCK A, B, C and D : 0-31
    ; --------------------------------------------------------------------------
    ; 1) Transposition of 32x8 block if its pass 2
    ; 2) First stage computation of 32x8/8x32 block
    ; 3) storing of this 32x8 coeffs into intermediate buffer

    ; Get stride of intermediate buffer
    mov             r10,    #16
    ; Get neg stride of intermediate buffer
    mov             r11,    #-16

    ;step[0]  =  input[0]  + input[(32 - 1)]
    ;step[31] = -input[31] + input[(32 - 32)];
    ; Since we do not have registers to use as tmp storage
    ; hence we will use bit twiddling for this butterfly
    ; a = a + b
    ; b = a - 2b
    vadd.s16        q0,     q0,     q15
    vshl.s16        q15,    q15,    #1
    vsub.s16        q15,    q0,     q15
    vst1.s16        {q15},  [r7],   r11
    vst1.s16        {q0},   [r3],   r10
    ; From here on use the DO_BUTTERFLY_NO_COEFFS and store to intermediate buffer
    ; --------------------------------------------------------------------------
    ;step[1]  =  input[1]  + input[(32 - 2)];
    ;step[30] = -input[30] + input[(32 - 31)];
    DO_BUTTERFLY_NO_COEFFS   q1,    q14,    q15
    vst1.s16        {q15},  [r7],   r11
    vst1.s16        {q14},  [r3],   r10
    ;step[2]  =  input[2]  + input[(32 - 3)];
    ;step[29] = -input[29] + input[(32 - 30)];
    DO_BUTTERFLY_NO_COEFFS   q2,    q13,    q1
    vst1.s16        {q1},   [r7],   r11
    vst1.s16        {q13},  [r3],   r10
    ;step[3]  =  input[3]  + input[(32 - 4)];
    ;step[28] = -input[28] + input[(32 - 29)];
    DO_BUTTERFLY_NO_COEFFS   q3,    q12,    q2
    vst1.s16        {q2},   [r7],   r11
    vst1.s16        {q12},  [r3],   r10
    ;step[4]  =  input[4]  + input[(32 - 5)];
    ;step[27] = -input[27] + input[(32 - 28)];
    DO_BUTTERFLY_NO_COEFFS   q4,    q11,    q3
    vst1.s16        {q3},   [r7],   r11
    vst1.s16        {q11},  [r3],   r10
    ;step[5]  =  input[5]  + input[(32 - 6)];
    ;step[26] = -input[26] + input[(32 - 27)];
    DO_BUTTERFLY_NO_COEFFS   q5,    q10,    q4
    vst1.s16        {q4},   [r7],   r11
    vst1.s16        {q10},  [r3],   r10
    ;step[6]  =  input[6]  + input[(32 - 7)];
    ;step[25] = -input[25] + input[(32 - 26)];
    DO_BUTTERFLY_NO_COEFFS   q6,    q9,     q5
    vst1.s16        {q5},   [r7],   r11
    vst1.s16        {q9},   [r3],   r10
    ;step[7]  =  input[7]  + input[(32 - 8)];
    ;step[24] = -input[24] + input[(32 - 25)];
    DO_BUTTERFLY_NO_COEFFS   q7,    q8,     q6
    vst1.s16        {q6},   [r7],   r11
    vst1.s16        {q8},   [r3],   r10

    ; No need to correct r3, r7 and r1, both point to correct location
    cmp             r5,     #1
    ; For first pass we need to read 17th row next, hence no need to correct
    ; input pointer
    ; For second pass, again we will be at start of 17th row, but we need to be
    ; at the 8th element of 1st row, hence subtract stride*8 - 16
    subne           r0,     r0,     r2,     lsl #3
    addne           r0,     r0,     #16
    ; Go to next 32-i block
    addne           r6,     r0,     #16
    ; Update the subband loop counter
    subs            r8,     #1
    bne             dct32_subband_pair_loop
    ; Restore the pointer for intermediate buffer
    sub             r3,     #256  ;Sub 16(num rows)*16(stride)

    ; --------------------------------------------------------------------------
    ; STAGE 2, BLOCK A, B, C : 0-15
    ; --------------------------------------------------------------------------
    ; 1) Load coeffs 1-16 in i, 16-i format
    ; 2) Store the subtracted coeffs in butterfly (corresponding to BLOCK C)
    ; 3) Don't store the added coeffs, keep them in registers so that we can
    ;    process block A and B together without memory access

    ; Load  step[0]-step[5], step[10]-step[15]
    LOAD_FROM_INTERMEDIATE    0,      15,     q0,     q1
    LOAD_FROM_INTERMEDIATE    1,      14,     q2,     q3
    LOAD_FROM_INTERMEDIATE    2,      13,     q4,     q5
    LOAD_FROM_INTERMEDIATE    3,      12,     q6,     q7
    LOAD_FROM_INTERMEDIATE    4,      11,     q8,     q9
    LOAD_FROM_INTERMEDIATE    5,      10,     q10,    q11

    ;output[0]  =  step[0]  + step[16 - 1];
    ;output[15] = -step[15] + step[16 - 16];
    DO_BUTTERFLY_NO_COEFFS  q0,     q1,     q15
    ;output[1]  =  step[1]  + step[16 - 2];
    ;output[14] = -step[14] + step[16 - 15];
    DO_BUTTERFLY_NO_COEFFS  q2,     q3,     q14
    ;output[2]  =  step[2]  + step[16 - 3];
    ;output[13] = -step[13] + step[16 - 14];
    DO_BUTTERFLY_NO_COEFFS  q4,     q5,     q13
    ;output[3]  =  step[3]  + step[16 - 4];
    ;output[12] = -step[12] + step[16 - 13];
    DO_BUTTERFLY_NO_COEFFS  q6,     q7,     q12
    ; Store subtracted coeffs (corresponding to BLOCK C)
    STORE_TO_INTERMEDIATE     15,     14,     q15,    q14
    STORE_TO_INTERMEDIATE     13,     12,     q13,    q12
    ; Load step[4],step[5],step[10],step[11]
    LOAD_FROM_INTERMEDIATE    6,      9,      q12,    q13
    LOAD_FROM_INTERMEDIATE    7,      8,      q14,    q15
    ;output[4]  =  step[4]  + step[16 - 5];
    ;output[11] = -step[11] + step[16 - 12];
    DO_BUTTERFLY_NO_COEFFS  q8,     q9,     q0
    ;output[5]  =  step[5]  + step[16 - 6];
    ;output[10] = -step[10] + step[16 - 11];
    DO_BUTTERFLY_NO_COEFFS  q10,    q11,    q2
    ;output[6]  =  step[6]  + step[16 - 7];
    ;output[9]  = -step[9]  + step[16 - 10];
    DO_BUTTERFLY_NO_COEFFS q12,     q13,    q4
    ;output[7]  =  step[7]  + step[16 - 8];
    ;output[7]  =  step[7]  + step[16 - 8];
    DO_BUTTERFLY_NO_COEFFS q14,     q15,    q6
    ; Store subtracted coeffs (corresponding to BLOCK C)
    STORE_TO_INTERMEDIATE     11,     10,     q0,     q2
    STORE_TO_INTERMEDIATE     9,      8,      q4,     q6
    ; In the second pass, we need to halve the value between the stages
    cmp             r5,     #0
    beq             round_shift_block_ABC
back_from_round_shift_block_ABC

    ; --------------------------------------------------------------------------
    ; STAGE 3 BLOCK A, B : 0-7
    ; --------------------------------------------------------------------------
    ;step[0]  =  output[0] + output[(8 - 1)];
    ;step[7]  =  -output[7]+ output[(8 - 8)];
    DO_BUTTERFLY_NO_COEFFS  q1,     q15,    q0
    ;step[1]  =  output[1] + output[(8 - 2)];
    ;step[6]  =  output[6] + output[(8 - 7)];
    DO_BUTTERFLY_NO_COEFFS  q3,     q13,    q2
    ;step[2]  =  output[2] + output[(8 - 3)];
    ;step[5]  = -output[5] + output[(8 - 6)];
    DO_BUTTERFLY_NO_COEFFS  q5,     q11,    q4
    ;step[3]  =  output[3] + output[(8 - 4)];
    ;step[4]  = -output[4] + output[(8 - 5)];
    DO_BUTTERFLY_NO_COEFFS  q7,    q9,     q6

    ; --------------------------------------------------------------------------
    ; STAGE 4 BLOCK A : 0-3
    ; --------------------------------------------------------------------------
    ;output[0] =  step[0] + step[3];
    ;output[3] = -step[3] + step[0];
    DO_BUTTERFLY_NO_COEFFS  q15,     q9,     q8
    ;output[1] =  step[1] + step[2];
    ;output[2] = -step[2] + step[1];
    DO_BUTTERFLY_NO_COEFFS  q13,     q11,    q10

    ; --------------------------------------------------------------------------
    ; STAGE 4 BLOCK B : 4-7
    ; --------------------------------------------------------------------------                                                                                        
    ;output[5] = dct_32_round((-step[5] + step[6]) * cospi_16_64);
    ;output[6] = dct_32_round(( step[6] + step[5]) * cospi_16_64);
    GET_CONST       d26,    cospi_16_64, r6
    vmovl.s16       q13,    d26
    DO_BUTTERFLY_SYMMETRIC_COEFFS   d4, d5, d8, d9, q13, d28, d29, q7, q5

    ; --------------------------------------------------------------------------
    ; STAGE 5 BLOCK A : 0-3
    ; --------------------------------------------------------------------------
    ;step[0]   = dct_32_round(( output[0] + output[1]) * cospi_16_64);
    ;step[1]   = dct_32_round((-output[1] + output[0]) * cospi_16_64);
    GET_CONST       d26,    cospi_16_64, r6
    vmovl.s16       q13,    d26
    DO_BUTTERFLY_SYMMETRIC_COEFFS   d18, d19, d22, d23, q13, d6, d7, q7, q2
    ;step[2]   = dct_32_round(output[2] * cospi_24_64 + output[3] * cospi_8_64);
    ;step[2]   = dct_32_round(output[2] * cospi_24_64 + output[3] * cospi_8_64);
    GET_CONST       d4,     cospi_24_64,r6
    GET_CONST       d5,     cospi_8_64,r6
    DO_BUTTERFLY_STD    d20, d21, d16, d17, d4, d5, q5, q7, q9

    ; --------------------------------------------------------------------------
    ; STAGE 5 BLOCK B : 4-7
    ; --------------------------------------------------------------------------
    ;step[4]  =  output[4] + output[5];
    ;step[5]  = -output[5] + output[4];
    DO_BUTTERFLY_NO_COEFFS  q6,     q14,    q2
    ;step[6]  = -output[6] + output[7];
    ;step[7]  =  output[7] + output[6];
    ;step[6] -> q5, step[7] -> q4
    DO_BUTTERFLY_NO_COEFFS  q0,     q4,     q5

    ; --------------------------------------------------------------------------
    ; STORE OUTPUT BLOCK A : 0-3
    ; --------------------------------------------------------------------------
    STORE_TO_OUTPUT 0,  16, q11, q3, 0,2
    STORE_TO_OUTPUT 8,  24, q10, q8, 1,3

    ; --------------------------------------------------------------------------
    ; STAGE 6 BLOCK B : 4-7
    ; --------------------------------------------------------------------------
    ;output[4] = dct_32_round(step[4] * cospi_28_64 +step[7] *  cospi_4_64);
    ;output[7] = dct_32_round(step[7] * cospi_28_64 +step[4] * -cospi_4_64);
    GET_CONST       d2,     cospi_28_64, r6
    GET_CONST       d3,     cospi_4_64,  r6
    DO_BUTTERFLY_STD    d28, d29, d8, d9, d2, d3, q11, q7, q9
    ;output[5] = dct_32_round(step[5] * cospi_12_64 + step[6] *  cospi_20_64);
    ;output[6] = dct_32_round(step[6] * cospi_12_64 + step[5] * -cospi_20_64);
    GET_CONST       d6,     cospi_12_64, r6
    GET_CONST       d7,     cospi_20_64, r6
    DO_BUTTERFLY_STD    d4, d5, d10, d11, d6, d7, q11, q7, q9

    ; --------------------------------------------------------------------------
    ; STORE OUTPUT BLOCK B : 4-7
    ; --------------------------------------------------------------------------
    STORE_TO_OUTPUT     4,  28, q14, q4, 0, 3
    STORE_TO_OUTPUT     20, 12, q2,  q5, 2, 1

    ; --------------------------------------------------------------------------
    ; STAGE 3 BLOCK C : 8-15
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE    10, 13, q1, q0
    LOAD_FROM_INTERMEDIATE    11, 12, q4, q3
    ;step[10] = dct_32_round((-output[10] + output[13]) * cospi_16_64);
    ;step[13] = dct_32_round(( output[13] + output[10]) * cospi_16_64);
    GET_CONST       d28,    cospi_16_64, r6
    vmovl.s16       q14,    d28
    ;step[10] -> q2, step[13] -> q1
    DO_BUTTERFLY_SYMMETRIC_COEFFS   d0, d1, d2, d3, q14, d4, d5, q15, q5
    ;step[11] = dct_32_round((-output[11] + output[12]) * cospi_16_64);
    ;step[12] = dct_32_round(( output[12] + output[11]) * cospi_16_64);
    ;step[11] -> q5, step[12] -> q4
    DO_BUTTERFLY_SYMMETRIC_COEFFS   d6, d7, d8, d9, q14, d10, d11, q15, q0

    ; --------------------------------------------------------------------------
    ; STAGE 4 BLOCK C: 8-15
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE    8,  9,  q6, q7
    LOAD_FROM_INTERMEDIATE    14, 15, q8, q9
    ;output[8]  =  step[8]  + step[11];
    ;output[11] = -step[11] + step[8];
    DO_BUTTERFLY_NO_COEFFS  q6,     q5,     q15
    ;output[9]  =  step[9]  + step[10];
    ;output[10] = -step[10] + step[9];
    DO_BUTTERFLY_NO_COEFFS  q7,     q2,     q14
    ;output[12] = -step[12] + step[15];
    ;output[15] =  step[15] + step[12];
    DO_BUTTERFLY_NO_COEFFS  q9,     q4,     q13
    ;output[13] = -step[13] + step[14];
    ;output[14] =  step[14] + step[13];
    DO_BUTTERFLY_NO_COEFFS  q8 ,    q1,     q12

    ; --------------------------------------------------------------------------
    ; STAGE 5 BLOCK C: 8-15
    ; --------------------------------------------------------------------------
    ;step[9]  = dct_32_round(output[9]  * -cospi_8_64 +
    ;                        output[14] * cospi_24_64 );
    ;step[14] = dct_32_round(output[14] * cospi_8_64  +
    ;                        output[9]  * cospi_24_64 );
    ;step[14] -> q2, step[9] -> q1
    GET_CONST       d0,     cospi_24_64, r6
    GET_CONST       d1,     cospi_8_64,  r6
    DO_BUTTERFLY_STD    d4, d5, d2, d3, d0, d1, q6, q7, q9
    ;step[10] = dct_32_round(output[10] * -cospi_24_64 +
    ;                        output[13] * -cospi_8_64  );
    ;step[13] = dct_32_round(output[13] * cospi_24_64  +
    ;                        output[10] * -cospi_8_64  );
    ;step[10] -> q12, step[13] -> q14
    GET_CONST       d0,     -cospi_24_64, r6
    GET_CONST       d1,     -cospi_8_64 , r6
    DO_BUTTERFLY_STD    d24, d25, d28, d29, d1, d0, q6, q7, q9

    ; --------------------------------------------------------------------------
    ; STAGE 6 BLOCK C: 8-15
    ; --------------------------------------------------------------------------
    ;output[8]  =  step[8]  + step[9];
    ;output[9]  = -step[9]  + step[8];
    DO_BUTTERFLY_NO_COEFFS  q5,     q1,     q0
    ;output[10] = -step[10] + step[11];
    ;output[11] =  step[11] + step[10];
    ;output[10] -> q3, output[11] -> q12
    DO_BUTTERFLY_NO_COEFFS  q15,    q12,    q3
    ;output[12] =  step[12] + step[13];
    ;output[13] = -step[13] + step[12];
    DO_BUTTERFLY_NO_COEFFS  q13,    q14,    q5
    ;output[14] = -step[14] + step[15];
    ;output[15] =  step[15] + step[14];
    DO_BUTTERFLY_NO_COEFFS  q4,     q2,     q6

    ; --------------------------------------------------------------------------
    ; STAGE 7 BLOCK C: 8-15
    ; --------------------------------------------------------------------------
    ;step[8]   = dct_32_round(output[8]  * cospi_30_64 +
    ;                         output[15] * cospi_2_64);
    ;step[15]  = dct_32_round(output[15] * cospi_30_64 +
    ;                         output[8]  * -cospi_2_64);
    GET_CONST       d8,     cospi_30_64, r6
    GET_CONST       d9,     cospi_2_64,  r6
    DO_BUTTERFLY_STD d2, d3, d4, d5, d8, d9, q7, q8, q9
    ;step[9]   = dct_32_round(output[9]  * cospi_14_64 +
    ;                         output[14] * cospi_18_64);
    ;step[14]  = dct_32_round(output[14] * cospi_14_64 +
    ;                         output[9]  * -cospi_18_64);
    GET_CONST       d8,     cospi_14_64, r6
    GET_CONST       d9,     cospi_18_64, r6
    DO_BUTTERFLY_STD d0, d1, d12, d13, d8, d9, q7, q8, q9
    ;step[10]  = dct_32_round(output[10] * cospi_22_64 +
    ;                         output[13] * cospi_10_64 );
    ;step[13]  = dct_32_round(output[13] * cospi_22_64 +
    ;                         output[10] * -cospi_10_64);
    GET_CONST       d8,     cospi_22_64, r6
    GET_CONST       d9,     cospi_10_64, r6
    DO_BUTTERFLY_STD d6, d7, d10, d11, d8, d9, q7, q8, q9
    ;step[11]  = dct_32_round(output[11] * cospi_6_64  +
    ;                         output[12] * cospi_26_64 );
    ;step[12]  = dct_32_round(output[12] * cospi_6_64  +
    ;                         output[11] * -cospi_26_64);
    GET_CONST       d8,     cospi_6_64,  r6
    GET_CONST       d9,     cospi_26_64, r6
    DO_BUTTERFLY_STD d24, d25, d28, d29, d8, d9, q7, q8, q9

    ; --------------------------------------------------------------------------
    ; STORE OUTPUT BLOCK C: 8-15
    ; --------------------------------------------------------------------------
    STORE_TO_OUTPUT 2,  30, q1,  q2,  0, 3        ;Store step[8],  step[15]
    STORE_TO_OUTPUT 18, 14, q0,  q6,  2, 1        ;Store step[9],  step[14]
    STORE_TO_OUTPUT 10, 22, q3,  q5,  1, 2        ;Store step[10], step[13]
    STORE_TO_OUTPUT 26,  6, q12, q14, 3, 0        ;Store step[11], step[12]

    ; --------------------------------------------------------------------------
    ; STAGE 2 BLOCK D: 16-31
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE 20, 27, q0, q7
    LOAD_FROM_INTERMEDIATE 21, 26, q1, q6
    LOAD_FROM_INTERMEDIATE 22, 25, q2, q5
    LOAD_FROM_INTERMEDIATE 23, 24, q3, q4
    GET_CONST       d30,    cospi_16_64, r6
    vmovl.s16       q15,    d30
    ;output[20] = dct_32_round((-step[20] + step[27]) * cospi_16_64);
    ;output[27] = dct_32_round(( step[27] + step[20]) * cospi_16_64);
    DO_BUTTERFLY_SYMMETRIC_COEFFS d14, d15, d0, d1, q15, d16, d17, q14, q13
    ;output[21] = dct_32_round((-step[21] + step[26]) * cospi_16_64);
    ;output[26] = dct_32_round(( step[26] + step[21]) * cospi_16_64);
    DO_BUTTERFLY_SYMMETRIC_COEFFS d12, d13, d2, d3, q15, d18, d19,  q14, q13
    ;output[22] = dct_32_round((-step[22] + step[25]) * cospi_16_64);
    ;output[26] = dct_32_round(( step[26] + step[21]) * cospi_16_64);
    DO_BUTTERFLY_SYMMETRIC_COEFFS d10, d11, d4, d5, q15, d20, d21,  q14, q13
    ;output[23] = dct_32_round((-step[23] + step[24]) * cospi_16_64);
    ;output[24] = dct_32_round(( step[24] + step[23]) * cospi_16_64);
    DO_BUTTERFLY_SYMMETRIC_COEFFS d8,  d9,  d6, d7, q15, d22, d23,  q14, q13
    cmp             r5,     #0
    ; If second pass, we will do the half round shift here for BLOCK D : 16-31
    beq             round_shift_block_D
back_from_round_shift_block_D

    ; --------------------------------------------------------------------------
    ; STAGE 3 BLOCK D: 16-31
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE 16, 17, q4, q5
    LOAD_FROM_INTERMEDIATE 18, 19, q6, q7
    ;step[16]  =  output[16] + output[23];
    ;step[23]  = -output[23] + output[16];
    DO_BUTTERFLY_NO_COEFFS  q4,     q11,    q15
    ;step[17]  =  output[17] + output[22];
    ;step[22]  = -output[22] + output[17];
    DO_BUTTERFLY_NO_COEFFS  q5,     q10,    q14
    ;step[18]  =  output[18]  + output[21];
    ;step[21]  = -output[21] + output[18];
    DO_BUTTERFLY_NO_COEFFS  q6,     q9,     q13
    ;step[19]  =  output[19] + output[20];
    ;step[20]  = -output[20] + output[19];
    DO_BUTTERFLY_NO_COEFFS  q7,     q8,     q12
    ; Store columns not needed for next stage into mem
    STORE_TO_INTERMEDIATE  16, 17, q11, q10
    STORE_TO_INTERMEDIATE  22, 23, q14, q15

    LOAD_FROM_INTERMEDIATE 31, 30, q4,  q5
    LOAD_FROM_INTERMEDIATE 28, 29, q6,  q7
    ;step[24]  = -output[24] + output[31];
    ;step[31]  =  output[31] + output[24];
    DO_BUTTERFLY_NO_COEFFS  q4,     q3,     q10
    ;step[25]  = -output[25] + output[30];
    ;step[30]  =  output[30] + output[25];
    DO_BUTTERFLY_NO_COEFFS  q5,     q2,     q11
    ;step[26]  = -output[26] + output[29];
    ;step[29]  =  output[29] + output[26];
    DO_BUTTERFLY_NO_COEFFS  q7,     q1,     q14
    ;step[27]  = -output[27] + output[28];
    ;step[28]  =  output[28] + output[27];
    DO_BUTTERFLY_NO_COEFFS  q6,     q0,     q15
    ;Store colums not needed for next stage into mem
    STORE_TO_INTERMEDIATE 30, 31, q2,  q3
    STORE_TO_INTERMEDIATE 24, 25, q10, q11

    ; --------------------------------------------------------------------------
    ; STAGE 4 BLOCK D: 16-31
    ; --------------------------------------------------------------------------
    GET_CONST       d4,     cospi_24_64, r6
    GET_CONST       d5,     cospi_8_64,  r6
    ;output[18] = dct_32_round(step[18] * -cospi_8_64 + step[29] * cospi_24_64);
    ;output[29] = dct_32_round(step[29] *  cospi_8_64 + step[18] * cospi_24_64);
    DO_BUTTERFLY_STD d18, d19, d2, d3, d4, d5, q6, q7, q10
    ;18 -> q1, 29->q9
    ;output[19] = dct_32_round(step[19] * -cospi_8_64 + step[28] * cospi_24_64);
    ;output[28] = dct_32_round(step[28] * cospi_8_64  + step[19] * cospi_24_64);
    DO_BUTTERFLY_STD d16, d17, d0, d1, d4, d5, q6, q7, q10
    ; 28 -> q8, 19 -> q0
    vneg.s16        d4,     d4
    vneg.s16        d5,     d5
    ;output[20] = dct_32_round(step[20] * -cospi_24_64 +
    ;                          step[27] * -cospi_8_64  );
    ;output[27] = dct_32_round(step[27] *  cospi_24_64 +
    ;                          step[20] * -cospi_8_64  );
    ;output[20] -> q15, output[27] -> q12
    DO_BUTTERFLY_STD d30, d31, d24, d25, d5, d4, q6, q7, q10
    ;output[21] = dct_32_round(step[21] * -cospi_24_64 +
    ;                          step[26] * -cospi_8_64  );
    ;output[26] = dct_32_round(step[26] * cospi_24_64  +
    ;                          step[21] * -cospi_8_64  );
    ;output[21] -> q14, output[26] -> q13
    DO_BUTTERFLY_STD d28, d29, d26, d27, d5, d4, q6, q7, q10
    STORE_TO_INTERMEDIATE 20, 21, q15, q14
    STORE_TO_INTERMEDIATE 26, 27, q13, q12

    ; --------------------------------------------------------------------------
    ; STAGE 5 BLOCK D PART 1: 16-19 28-31
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE 16,17,q4,q5
    LOAD_FROM_INTERMEDIATE 30,31,q6,q7

    ;step[16] =  output[16] + output[19];
    ;step[19] = -output[19] + output[16];
    DO_BUTTERFLY_NO_COEFFS  q4,     q0,     q11
    ;step[17] =  output[17] + output[18];
    ;step[17] =  output[17] + output[18];
    DO_BUTTERFLY_NO_COEFFS q5,      q1,     q10
    ;step[28] = -output[28] + output[31];
    ;step[31] =  output[31] + output[28];
    ;step[31] -> q8, step[28] -> q4
    DO_BUTTERFLY_NO_COEFFS q7,      q8,     q4
    ;step[29] = -output[29] + output[30];
    ;step[30] =  output[30] + output[29];
    ;step[30] -> q9, step[29] -> q5
    DO_BUTTERFLY_NO_COEFFS q6,      q9,      q5

    ; --------------------------------------------------------------------------
    ; STAGE 6 BLOCK D PART 1 : 16-19,28-31
    ; --------------------------------------------------------------------------
    ;output[17] = dct_32_round(step[17] * -cospi_4_64 +
    ;                          step[30] * cospi_28_64 );
    ;output[30] = dct_32_round(step[30] *  cospi_4_64 +
    ;                          step[17] * cospi_28_64 );
    ;output[17] -> q9, output[30] -> q1
    GET_CONST       d13,    cospi_4_64,  r6
    GET_CONST       d12,    cospi_28_64, r6
    DO_BUTTERFLY_STD d2, d3, d18, d19, d12, d13, q14, q15, q7
    ;output[18] = dct_32_round(step[18] * -cospi_28_64 +
    ;                          step[29] * -cospi_4_64);
    ;output[29] = dct_32_round(step[29] *  cospi_28_64 +
    ;                          step[18] * -cospi_4_64);
    GET_CONST       d12,    -cospi_4_64,  r6
    GET_CONST       d13,    -cospi_28_64, r6
    DO_BUTTERFLY_STD d10, d11, d20, d21, d12, d13, q14, q15, q7

    ; --------------------------------------------------------------------------
    ; STAGE 7 BLOCK D PART 1 : 16-19,28-31
    ; --------------------------------------------------------------------------
    ;step[16] =  output[16] + output[17];
    ;step[17] = -output[17] + output[16];
    DO_BUTTERFLY_NO_COEFFS q0,      q9,     q15
    ;step[18] = -output[18] + output[19];
    ;step[19] =  output[19] + output[18];
    ;step[18] -> q14, step[19] -> q5
    DO_BUTTERFLY_NO_COEFFS q11,     q5,     q14
    ;step[28] =  output[28] + output[29];
    ;step[29] = -output[29] + output[28];
    DO_BUTTERFLY_NO_COEFFS  q4,     q10,    q7
    ;step[30] = -output[30] + output[31];
    ;step[31] =  output[31] + output[30];
    ;step[31] -> q1, step[30] -> q12
    DO_BUTTERFLY_NO_COEFFS  q8,     q1,     q12

    ; --------------------------------------------------------------------------
    ; STAGE 8 BLOCK D PART 1 : 16-19,28-31
    ; --------------------------------------------------------------------------
    ;output[1]  = dct_32_round(step[16] *  cospi_31_64 +
    ;                          step[31] *  cospi_1_64  );
    ;output[31] = dct_32_round(step[31] *  cospi_31_64 +
    ;                          step[16] * -cospi_1_64  );
    GET_CONST       d0,     cospi_31_64, r6
    GET_CONST       d1,     cospi_1_64,  r6
    DO_BUTTERFLY_STD d18, d19, d2, d3, d0, d1, q2, q3, q11
    ;output[17] = dct_32_round(step[17] * cospi_15_64 +
    ;                          step[30] * cospi_17_64 );
    ;output[15] = dct_32_round(step[30] * cospi_15_64 +
    ;                          step[17] * -cospi_17_64)
    GET_CONST       d0,     cospi_15_64, r6
    GET_CONST       d1,     cospi_17_64, r6
    DO_BUTTERFLY_STD d30, d31, d24, d25, d0, d1, q2, q3, q11
    ;output[9]  = dct_32_round(step[18] * cospi_23_64 +
    ;                          step[29] * cospi_9_64  );
    ;output[23] = dct_32_round(step[29] * cospi_23_64 +
    ;                          step[18] * -cospi_9_64 );
    GET_CONST       d0,     cospi_23_64, r6
    GET_CONST       d1,     cospi_9_64,  r6
    DO_BUTTERFLY_STD d28, d29, d14, d15, d0, d1, q2, q3, q11
    ;output[25] = dct_32_round(step[19] * cospi_7_64  +
    ;                          step[28] * cospi_25_64 );
    ;output[7]  = dct_32_round(step[28] * cospi_7_64  +
    ;                          step[19] * -cospi_25_64);
    GET_CONST       d0,     cospi_7_64,  r6
    GET_CONST       d1,     cospi_25_64, r6
    DO_BUTTERFLY_STD d10, d11, d20, d21, d0, d1, q2, q3, q11

    ; --------------------------------------------------------------------------
    ; STORE OUTPUT BLOCK D PART 1 : 16-19,28-31
    ; --------------------------------------------------------------------------
    STORE_TO_OUTPUT 1,  31, q9,  q1,  0, 3     ;Store 16,31
    STORE_TO_OUTPUT 17, 15, q15, q12, 2, 1     ;Store 17,30
    STORE_TO_OUTPUT 9,  23, q14, q7,  1, 2     ;Store 18,29
    STORE_TO_OUTPUT 25, 7,  q5,  q10, 3, 0     ;Store 19,28

    ; --------------------------------------------------------------------------
    ; STAGE 5 BLOCK D PART 2 : 16-19,28-31
    ; --------------------------------------------------------------------------
    LOAD_FROM_INTERMEDIATE 20, 21, q0, q1       ;load 20 21
    LOAD_FROM_INTERMEDIATE 22, 23, q2, q3       ;load 22 23
    LOAD_FROM_INTERMEDIATE 24, 25, q4, q5       ;load 24 25
    LOAD_FROM_INTERMEDIATE 26, 27, q6, q7       ;load 26 27

    ;step[20] = -output[20] + output[23];
    ;step[23] =  output[23] + output[20];
    ;step[20] -> q0, step[23] -> q10
    DO_BUTTERFLY_NO_COEFFS  q3,     q0,     q10
    ;step[21] = -output[21] + output[22];
    ;step[22] =  output[22] + output[21];
    ;step[22] -> q1, step[21] -> q11
    DO_BUTTERFLY_NO_COEFFS  q2,      q1,     q11
    ;step[24] =  output[24] + output[27];
    ;step[27] = -output[27] + output[24];
    DO_BUTTERFLY_NO_COEFFS  q4,      q7,     q3
    ;step[25] =  output[25] + output[26];
    ;step[26] = -output[26] + output[25];
    DO_BUTTERFLY_NO_COEFFS  q5,      q6,     q2

    ; --------------------------------------------------------------------------
    ; STAGE 6 BLOCK D PART 2 : 20-27
    ; --------------------------------------------------------------------------
    ;output[21] = dct_32_round(step[21] * -cospi_20_64 +
    ;                          step[26] *  cospi_12_64);
    ;output[26] = dct_32_round(step[26] *  cospi_20_64 +
    ;                          step[21] *  cospi_12_64);
    ;output[26] -> q11, output[21] -> q2
    GET_CONST       d31,    cospi_20_64, r6
    GET_CONST       d30,    cospi_12_64, r6
    DO_BUTTERFLY_STD d22, d23, d4, d5, d30, d31, q14, q13, q12
    ;output[22] = dct_32_round(step[22] * -cospi_12_64 +
    ;                          step[25] * -cospi_20_64);
    ;output[25] = dct_32_round(step[25] *  cospi_12_64 +
    ;                          step[22] * -cospi_20_64);
    ;output[22] -> q6, output[25] -> q1
    GET_CONST       d31,    -cospi_20_64, r6
    GET_CONST       d30,    -cospi_12_64, r6
    DO_BUTTERFLY_STD d12, d13, d2, d3, d31, d30, q14, q13, q12

    ; --------------------------------------------------------------------------
    ; STAGE 7 BLOCK D PART 2 : 20-27
    ; --------------------------------------------------------------------------
    ;step[20] =  output[20] + output[21];
    ;step[21] = -output[21] + output[20];
    DO_BUTTERFLY_NO_COEFFS  q10,     q2,     q4
    ;step[22] = -output[22] + output[23];
    ;step[23] =  output[23] + output[22];
    ;step[23] -> q6, step[22] -> q5
    DO_BUTTERFLY_NO_COEFFS  q0,      q6,     q5
    ;step[24] =  output[24] + output[25];
    ;step[25] = -output[25] + output[24];
    DO_BUTTERFLY_NO_COEFFS  q7,     q1,     q8
    ;step[26] = -output[26] + output[27];
    ;step[27] =  output[27] + output[26];
    ;step[27] -> q1, step[26] -> q9
    DO_BUTTERFLY_NO_COEFFS  q3,     q11,    q9

    ; --------------------------------------------------------------------------
    ; STAGE 8 BLOCK D PART 2 : 20-27
    ; --------------------------------------------------------------------------
    ;output[5]  = dct_32_round(step[20] *  cospi_27_64 +
    ;                          step[27] *  cospi_5_64  );
    ;output[27] = dct_32_round(step[27] *  cospi_27_64 +
    ;                          step[20] * -cospi_5_64  );
    GET_CONST       d0,     cospi_27_64, r6
    GET_CONST       d1,     cospi_5_64,  r6
    DO_BUTTERFLY_STD d4, d5, d22, d23, d0, d1, q10, q3, q12
    ;output[21] = dct_32_round(step[21] * cospi_11_64 +
    ;                          step[26] * cospi_21_64 );
    ;output[11] = dct_32_round(step[26] * cospi_11_64 +
    ;                          step[21] *-cospi_21_64 );
    GET_CONST       d0,     cospi_11_64, r6
    GET_CONST       d1,     cospi_21_64, r6
    DO_BUTTERFLY_STD d8, d9, d18, d19, d0, d1, q10, q3, q12
    ;output[13] = dct_32_round(step[22] * cospi_19_64 +
    ;                          step[25] * cospi_13_64 );
    ;output[19] = dct_32_round(step[25] * cospi_19_64 +
    ;                          step[22] *-cospi_13_64 );
    GET_CONST       d0,     cospi_19_64, r6
    GET_CONST       d1,     cospi_13_64, r6
    DO_BUTTERFLY_STD d10, d11, d16, d17, d0,d1, q10, q3, q12
    ;output[29] = dct_32_round(step[23] *  cospi_3_64  +
    ;                          step[24] *  cospi_29_64 );
    ;output[3]  = dct_32_round(step[24] *  cospi_3_64  +
    ;                          step[23] * -cospi_29_64 );
    GET_CONST       d0,     cospi_3_64,  r6
    GET_CONST       d1,     cospi_29_64, r6
    DO_BUTTERFLY_STD d12, d13, d2, d3, d0, d1, q10, q3, q12

    ; --------------------------------------------------------------------------
    ; STORE OUTPUT BLOCK D PART 2 : 20-27
    ; --------------------------------------------------------------------------
    STORE_TO_OUTPUT 5,  27, q2, q11, 0, 3     ;Store 20,27
    STORE_TO_OUTPUT 21, 11, q4, q9,  2, 1     ;Store 21,26
    STORE_TO_OUTPUT 13, 19, q5, q8,  1, 2     ;Store 22,25
    STORE_TO_OUTPUT 29, 3,  q6, q1,  3, 0     ;Store 23,24

    ; Done with band loop, now adjust pointers
    cmp             r5,     #1
    ; For pass 1, we will need to go back 16 rows and forward by 8 columns
    subeq           r0,     r0,     r2,     lsl #4 ;-16*input_stride+16
    addeq           r0,     #16
    ; Move to next 8x32 block, add 16 since output pointer is restores every
    ; time in  pass 1
    addeq           r1,     #16
    ; If pass 2 get to next 8x32 block
    addne           r1,     #512 ;Output stride *16
    ; Input and output are same in pass 2
    movne           r0,     r1

    ; Update band loop cntr
    subs            r4,     #1
    bne             dct32_bands_loop

    ; Update test count
    sub             r5,     #1
    ; Update pointers for pass2
    ; At the end of first pass, we will be at the end of first row, so get back
    sub             r1,     #64
    ; Set 2nd stage input as 1st stage output
    mov             r0,     r1
    ; Set output stride as input stride
    mov             r2,     #64
    ; In pass 2 we will read 32 colums in a single row, hence both
    ; strides are positive
    mov             r9,     #64

    cmp             r5,     #-1
    bne             dct32_pass_loop

    ; Now transpose each 8x8 matrices
    ; We have output in r1 with a stride of 64, so Get r1 back to origin
    ; Output pointer will be at the position 32*64(last row), we already
    ; subtracted 64 for pass loop update hence subtract 32*64 - 64
    sub             r1,     #1984

    ; use r2,r3 as src pointers,
    ; r2 for block i r3 fro block i+1
    ; use r4 r5 as corresponding output pointers
    ; Address of block 0 for input
    mov             r2,     r1
    ; Address of block 1 for input
    add             r3,     r1,     #16
    ; Address of block 0 for output
    mov             r4,     r2
    ; Address of block 1 for output
    mov             r5,     r3
    ; Stride of output
    mov             r9,     #64
    mov             r10,    #1
    ; We need to process 16 blocks, hece loop cntr 8
    mov             r6,     #8
transpose_final
    TRANSPOSE_TWO_8x8 r2, r3, r4, r5, r9
    ; update loop  counter
    sub             r6,     #1
    and             r7,     r6, r10
    ; Check odd or even
    cmp             r7,     #0
    ; If even, it means the end of 8x32 block
    ; then move to next b8x32 block
    addeq           r1,     #480
    ; else just move 2 blocks forward
    addne           r1,     #32
    mov             r2,     r1
    add             r3,     r1, #16
    mov             r4,     r2
    mov             r5,     r3
    ; Check loop counter
    cmp             r6,     #0
    bne             transpose_final

    ;Restore SP
    add             sp,     #512
    vpop            {d8-d15}
    pop             {r4-r11}
    bx              lr

    ENDP  ; |vp9_fdct32x32_rd_neon|


pass2_preprocess
    ; transpose the two 8x8 16bit data matrices.
    vswp            d31,    d22
    vswp            d25,    d16
    vswp            d27,    d18
    vswp            d29,    d20
    vswp            d1,     d8
    vswp            d7,     d14
    vswp            d5,     d12
    vswp            d3,     d10
    vtrn.32         q15,    q13
    vtrn.32         q14,    q12
    vtrn.32         q11,    q9
    vtrn.32         q10,    q8
    vtrn.32         q0,     q2
    vtrn.32         q1,     q3
    vtrn.32         q4,     q6
    vtrn.32         q5,     q7
    vtrn.16         q15,    q14
    vtrn.16         q13,    q12
    vtrn.16         q11,    q10
    vtrn.16         q9 ,    q8
    vtrn.16         q0,     q1
    vtrn.16         q2,     q3
    vtrn.16         q4,     q5
    vtrn.16         q6,     q7
    vswp            q15,    q8
    vswp            q14,    q9
    vswp            q13,    q10
    vswp            q12,    q11

    ; Since all 16 registers are full, store to temporary space
    ; so that we have some registers to be used as temporary 
    STORE_TO_INTERMEDIATE 0, 1, q14, q15

    ;output[j * 32 + i] = (temp_out[j] + 1 + (temp_out[j] > 0)) >> 2;
    vshr.u16        q14,    q0,     #15
    vsub.s16        q0,     q0,     q14
    vrshr.s16       q0,     q0,     #2

    vshr.u16        q15,    q1,     #15
    vsub.s16        q1,     q1,     q15
    vrshr.s16       q1,     q1,     #2

    vshr.u16        q14,    q2,     #15
    vsub.s16        q2,     q2,     q14
    vrshr.s16       q2,     q2,     #2

    vshr.u16        q15,    q3,     #15
    vsub.s16        q3,     q3,     q15
    vrshr.s16       q3,     q3,     #2

    vshr.u16        q14,    q4,     #15
    vsub.s16        q4,     q4,     q14
    vrshr.s16       q4,     q4,     #2

    vshr.u16        q15,    q5,     #15
    vsub.s16        q5,     q5,     q15
    vrshr.s16       q5,     q5,     #2

    vshr.u16        q14,    q6,     #15
    vsub.s16        q6,     q6,     q14
    vrshr.s16       q6,     q6,     #2

    vshr.u16        q15,    q7,     #15
    vsub.s16        q7,     q7,     q15
    vrshr.s16       q7,     q7,     #2

    vshr.u16        q14,    q8,     #15
    vsub.s16        q8,     q8,     q14
    vrshr.s16       q8,     q8,     #2

    vshr.u16        q15,    q9,     #15
    vsub.s16        q9,     q9,     q15
    vrshr.s16       q9,     q9,     #2

    ; Store again so that we can process column 0 and column 1 too
    STORE_TO_INTERMEDIATE 2, 3, q8 ,q9

    vshr.u16        q14,    q10,    #15
    vsub.s16        q10,    q10,    q14
    vrshr.s16       q10,    q10,    #2

    vshr.u16        q15,    q11,    #15
    vsub.s16        q11,    q11,    q15
    vrshr.s16       q11,    q11,    #2

    vshr.u16        q14,    q12,    #15
    vsub.s16        q12,    q12,    q14
    vrshr.s16       q12,    q12,    #2

    vshr.u16        q15,    q13,    #15
    vsub.s16        q13,    q13,    q15
    vrshr.s16       q13,    q13,    #2

    ; restore column 0 and column 1
    LOAD_FROM_INTERMEDIATE 0, 1, q14, q15

    vshr.u16        q8,     q14,    #15
    vsub.s16        q14,    q14,    q8
    vrshr.s16       q14,    q14,    #2

    vshr.u16        q9,     q15,    #15
    vsub.s16        q15,    q15,    q9
    vrshr.s16       q15,    q15,    #2

    ; restore column 8 column 9
    LOAD_FROM_INTERMEDIATE 2, 3, q8, q9

    b               back_from_pass2_preprocess

    ; Half round blocks
    ; Does half_round_shift for blocks A, B, and C
    ; Touches q0 - q15
round_shift_block_ABC

    ;half_round_shift(output[11]);
    ;half_round_shift(output[10]);
    HALF_ROUND_SHIFT q0, q2, q12, q14
    ;half_round_shift(output[9]);
    ;half_round_shift(output[8]);
    HALF_ROUND_SHIFT q4, q6, q12, q14
    ; Store column 11 and column 10
    STORE_TO_INTERMEDIATE 11, 10, q0, q2
    ; Store column 9 and column 8
    STORE_TO_INTERMEDIATE 9,  8,  q4, q6

    ; load column 15 and column 14
    LOAD_FROM_INTERMEDIATE 15, 14, q0, q2
    ; load column 13 and column 12
    LOAD_FROM_INTERMEDIATE 13, 12, q4, q6

    ;half_round_shift(output[0]);
    ;half_round_shift(output[4]);
    HALF_ROUND_SHIFT q1, q9,  q12, q14
    ;half_round_shift(output[1]);
    ;half_round_shift(output[5]);
    HALF_ROUND_SHIFT q3, q11, q12, q14
    ;half_round_shift(output[2]);
    ;half_round_shift(output[6]);
    HALF_ROUND_SHIFT q5, q13, q12, q14
    ;half_round_shift(output[3]);
    ;half_round_shift(output[7]);
    HALF_ROUND_SHIFT q7, q15, q12, q14
    ;half_round_shift(output[15]);
    ;half_round_shift(output[14]);
    HALF_ROUND_SHIFT q0, q2,  q12, q14
    ;half_round_shift(output[13]);
    ;half_round_shift(output[12]);
    HALF_ROUND_SHIFT q4, q6,  q12, q14

    ; Store column 15 and column 14
    STORE_TO_INTERMEDIATE 15, 14, q0, q2
    ; Store column 13 and column 12
    STORE_TO_INTERMEDIATE 13, 12, q4, q6
    b               back_from_round_shift_block_ABC

    ; Does half_round_shift for blocks A B and C
    ; Touches q0 - q15
round_shift_block_D

    ;half_round_shift(output[27]);
    ;half_round_shift(output[26]);
    HALF_ROUND_SHIFT q0,  q1,  q14, q15
    ;half_round_shift(output[25]);
    ;half_round_shift(output[24]);
    HALF_ROUND_SHIFT q2,  q3,  q14, q15
    ;half_round_shift(output[23]);
    ;half_round_shift(output[22]);
    HALF_ROUND_SHIFT q8,  q9,  q14, q15
    ;half_round_shift(output[20]);
    ;half_round_shift(output[21]);
    HALF_ROUND_SHIFT q10, q11, q14, q15

    LOAD_FROM_INTERMEDIATE 16, 17, q4, q5
    LOAD_FROM_INTERMEDIATE 18, 19, q6, q7
    ;half_round_shift(output[16]);
    ;half_round_shift(output[17]);
     HALF_ROUND_SHIFT q4, q5 ,q14, q15
    ;half_round_shift(output[18]);
    ;half_round_shift(output[19]);
    HALF_ROUND_SHIFT q6, q7 ,q14, q15
    STORE_TO_INTERMEDIATE 16, 17, q4, q5
    STORE_TO_INTERMEDIATE 18, 19, q6, q7

    LOAD_FROM_INTERMEDIATE 28, 29, q4, q5
    LOAD_FROM_INTERMEDIATE 30, 31, q6, q7
    ;half_round_shift(output[28]);
    ;half_round_shift(output[29]);
    HALF_ROUND_SHIFT q4, q5 ,q14, q15
    ;half_round_shift(output[30]);
    ;half_round_shift(output[31]);
    HALF_ROUND_SHIFT q6, q7, q14, q15
    STORE_TO_INTERMEDIATE 28, 29, q4, q5
    STORE_TO_INTERMEDIATE 30, 31, q6, q7
    b back_from_round_shift_block_D

    
    END

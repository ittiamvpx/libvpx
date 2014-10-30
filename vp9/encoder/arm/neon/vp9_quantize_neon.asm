;
;  Copyright (c) 2013 The WebM project authors. All Rights Reserved.
;
;  Use of this source code is governed by a BSD-style license
;  that can be found in the LICENSE file in the root of the source
;  tree. An additional intellectual property rights grant can be found
;  in the file PATENTS.  All contributing project authors may
;  be found in the AUTHORS file in the root of the source tree.
;

    EXPORT  |vp9_quantize_fp_32x32_neon|
    EXPORT  |vp9_quantize_b_32x32_neon|
    EXPORT  |vp9_quantize_b_neon|

    ARM
    REQUIRE8
    PRESERVE8

    AREA ||.text||, CODE, READONLY, ALIGN=2

    ; --------------------------------------------------------------------------
    ; Does the following tasks on the input 4x4 sub-block
    ;   - Returns the absolute value of input
    ;   - Returns the sign bit(+ve ot -ve) of the input
    ;   - Check whether absolute value of all the elements in a 4x4 sub-block is
    ;     greater than or equal to the given threshold value.
    ;
    ; Inputs
    ;   q5  - Row1 and Row2 of the 4x4 sub-block
    ;   q6  - Row3 and Row4 of the 4x4 sub-block
    ;   q15 - Threshold value
    ; Outputs
    ;   q5  - Absolute value of Row1 and Row2
    ;   q6  - Absolute value of Row3 and Row4
    ;   q10 - Sign bits of Row1 and Row2
    ;   q11 - Sign bits of Row3 and Row4
    ;   q12 - Is Row1 and Row2 values is greater than or equal to threshold
    ;   q13 - Is Row3 and Row4 values is greater than or equal to threshold
    ;   r10 - Set to '0', if all the values in 4x4 is lesser than the threshold
    ; Touches q7 register
    MACRO
    THRESHOLD_4X4_BLOCK $is_first_block
    vshr.s16        q10, q5,  #15
    vshr.s16        q11, q6,  #15
    vabs.s16        q5,  q5
    vabs.s16        q6,  q6
    vcge.u16        q12, q5,  q15
  IF $is_first_block = 1             ; In the first 4x4 block, the threshold
    vmov.s16        d30, d31         ; value is different between Row1 and Row3
  ENDIF                              ; because Row1 would contain the DC
    vcge.u16        q13, q6,  q15
    vorr.16         q7,  q12, q13
    vorr.16         d14, d14, d15
    vpadd.s32       d14, d14, d14
    vmov.32         r10, d14[0]
    MEND
    ; --------------------------------------------------------------------------
    ; Does the following tasks on the input 4x4 sub-block
    ;   - Returns the quantized values
    ;   - Returns the dequantized values
    ;   - Find maximum of iscan values corresponding to processed
    ;     coefficients.
    ; Inputs
    ;   q5  - Absolute value of Row1 and Row2 of the 4x4 sub-block
    ;   q6  - Absolute value of Row3 and Row4 of the 4x4 sub-block
    ;   q10 - Sign bits of Row1 and Row2
    ;   q11 - Sign bits of Row3 and Row4
    ;   q3  - Iscan values corresponding to Row1 and Row2
    ;   q4  - Iscan values corresponding to Row3 and Row4
    ;   q12 - Is Row1 and Row2 values is greater than or equal to threshold
    ;   q13 - Is Row3 and Row4 values is greater than or equal to threshold
    ;   q1  - Rounding factor (ROUND_POWER_OF_TWO(round_ptr[rc != 0], 1);)
    ;   q2  - quant_shift
    ;   d0  - Quantization parameter
    ;   d1  - Dequantization parameter
    ;   q14 - Maximum of iscan values until now
    ; Outputs
    ;   q5  - Quantized value of Row1 and Row2
    ;   q6  - Quantized value of Row3 and Row4
    ;   q8  - Dequantized value of Row1 and Row2
    ;   q9  - Dequantized value of Row3 and Row4
    ;   q14 - Maximum of iscan values.
    ; Touches q7, q12 and q13
    MACRO
    QUANTIZE_4X4_BLOCK $is_first_block $width $quant_type $narrow_factor
    ; saturated addition with the rounding factor
    vqadd.s16       q5,  q5,  q1
  IF $is_first_block = 1           ; In the first 4x4 block, rounding factor
    vmov.s16        d2,  d3        ; is different between Row1 and Row3 because
  ENDIF                            ; Row1 would contain the DC coefficient
    vqadd.s16       q6,  q6,  q1
    ; copies all the values greater than the threshold
    vbif.16         q5,  q12, q12
    vbif.16         q6,  q13, q13
    ; multiplying with  quantization values
    vmull.s16       q7,  d10, d0
  IF $is_first_block = 1           ; In the first 4x4 block, quantization factor
    vdup.16         d0,  d0[1]     ; is different between Row1 and Row3 because
  ENDIF                            ; Row1 would contain the DC coefficient
    vmull.s16       q8,  d11, d0
    vmull.s16       q9,  d12, d0
    vmull.s16       q12, d13, d0
  IF $quant_type = quantize_b
    ; right shift by 16, and narrow down to 16 bits
    vshrn.s32       d14, q7,  #16
    vshrn.s32       d16, q8,  #16
    vshrn.s32       d26, q9,  #16
    vshrn.s32       d27, q12, #16
    ; adding with rounded coeffs
    vaddl.s16       q7,  d10, d14
    vaddl.s16       q8,  d11, d16
    vaddl.s16       q9,  d12, d26
    vaddl.s16       q12, d13, d27
    ; multiplying with quant_shift value
    vmul.s32        q7,  q7,  q2
   IF $is_first_block = 1            ; In the first 4x4 block, the quant_shift
    vdup.32         d4,  d4[1]       ; value is different between Row1 and Row3
   ENDIF                             ; because Row1 would contain the DC
    vmul.s32        q8,  q8,  q2
    vmul.s32        q9,  q9,  q2
    vmul.s32        q12, q12, q2
  ENDIF                              ; IF $quant_type = quantize_b
    ; right shift by 15 for width =32 or by16 for rest, and narrow down to 16bit
    vshrn.s32       d12, q7,  #$narrow_factor
    vshrn.s32       d13, q8,  #$narrow_factor
    vshrn.s32       d14, q9,  #$narrow_factor
    vshrn.s32       d15, q12, #$narrow_factor
    ; if quantized values > 0, corresponding iscan values are copied
    vcgt.s16        q12, q6,  #0
    vcgt.s16        q13, q7,  #0
    vbit.16         q12, q3,  q12
    vbit.16         q13, q4,  q13
    ; finding max of iscan values and keeping them in q14
    vmax.s16        q12, q12, q13
    vmax.s16        q14, q14, q12
  IF $width = 32
    ; multiplying quantized values with dequantization factors
    vmull.s16       q8,  d12, d1
   IF $is_first_block = 1            ; In the first 4x4 block, dequantization
    vdup.16         d1,  d1[1]       ; value is different between Row1 and Row3
   ENDIF                             ; because Row1 would contain the DC.
    vmull.s16       q9,  d13, d1
    vmull.s16       q12, d14, d1
    vmull.s16       q13, d15, d1
    ; restoring sign for quantized values
  ENDIF                            ; IF $width = 32
    veor.16         q5,  q6,  q10
    veor.16         q6,  q7,  q11
    vsub.s16        q5,  q5,  q10
    vsub.s16        q6,  q6,  q11
  IF $width = 0
    ;multiplying with values of dequant_ptr
    vmul.s16        d16, d10, d1
   IF $is_first_block = 1
    vdup.16         d1,  d1[1]
   ENDIF
    vmul.s16        d17, d11, d1
    vmul.s16        d18, d12, d1
    vmul.s16        d19, d13, d1
  ELSE
    ; dividing the dequantized values by 2 and narrowing to 16bit
    vshrn.s32       d16, q8,  #1
    vshrn.s32       d17, q9,  #1
    vshrn.s32       d18, q12, #1
    vshrn.s32       d19, q13, #1
    ; restoring sign for dequantized values
    veor.16         q8,  q8,  q10
    veor.16         q9,  q9,  q11
    vsub.s16        q8,  q8,  q10
    vsub.s16        q9,  q9,  q11
  ENDIF
    MEND

    ; --------------------------------------------------------------------------
    ; Does the following task on the register
    ;   - Find the maximum 16 bit value in the register
    ; Input
    ;   q14    - Maximum of iscan values corresponding to quantized values
    ; Output
    ;   d28[0] - Maximum 16 bit value in q14
    MACRO
    FIND_EOB
    vmax.s16        d28, d28, d29
    vext.16         d29, d28, d28, #2
    vmax.s16        d28, d28, d29
    vext.16         d29, d28, d28, #1
    vmax.s16        d28, d28, d29
    MEND
    ; --------------------------------------------------------------------------
    ; Does the following tasks on the input
    ;   - Peform quantization for the given block
    ;   - Store the quantized and dequantized values
    ;   - Store EOB value
    ; Inputs
    ;   r0-r3       - coming from the function call
    ;   sp          - coming from the function call
    ;   $width      - width of the block being processed($width = 32 for 32x32
    ;                 and $width = 0 for other cases)
    ;   $quant_type - type of qunatization(quantize_b or quantize_fp)
    ; Touches q0-q15,r0-r12 registers
    MACRO
    QUANTIZE $width $quant_type
    push            {r4-r12, lr}         ; push registers to stack
    ldr             r11,   [sp, #40]     ; int16_t *round_ptr
    ldr             r4,    [sp, #44]     ; int16_t *quant_ptr
    ldr             r12,   [sp, #48]     ; int16_t *quant_shift_ptr
    ldr             r5,    [sp, #52]     ; tran_low_t *qcoeff_ptr
    ldr             r6,    [sp, #56]     ; tran_low_t *dqcoeff_ptr
    ldr             r7,    [sp, #60]     ; int16_t *dequant_ptr
    ldr             r8,    [sp, #68]     ; uint16_t *eob_ptr
    ldr             r9,    [sp, #76]     ; int16_t *iscan
    ldr             r14,   [sp, #72]     ; zbin_oq_value
    vpush           {d8-d15}
    ; check if skip_block
    cmp             r2,    #0
    mov             r10,   #0
    beq             not_skip_block_$width$quant_type
    ; when skip_block = 1
skip_block_$width$quant_type
    vdup.16         q10,   r10
    vst1.64         {q10}, [r5]!        ;storing 0  to  qcoeff_ptr
    vst1.64         {q10}, [r6]!        ;storing 0  to  dqcoeff_ptr
    subs            r1,    r1,   #8
    bne  skip_block_$width$quant_type
    ; store eob =0
    str             r10,   [r8]
    ; exiting the function
    b end_func_$width$quant_type
    ; when skip_block = 0
not_skip_block_$width$quant_type
    ; load quantization parmameters
    vld1.16         d0,    [r4]
  IF $quant_type = quantize_b
    ; load zbin values
    vld1.32         d30,   [r3]
    vdup.16         d31,   d30[1]
    ldr             r10,   [r14]
    vdup.16         q7,    r10
    ; q15[0] =(zbin + zbin_oq_value) dc, q15[1-7] =(zbin + zbin_oq_value) ac
    vqadd.s16       q15,   q15,   q7
   IF $width = 32
    vrshr.s16       q15,   q15,   #1
   ENDIF
    ; load dequant parameters
    vld1.32         d1,    [r7]        ; d1[0] = dequant dc,d0[1-3]=dequant ac
    ; load quant_shift_ptr
    vld1.32         d4,    [r12]
    vmovl.s16       q2,    d4
  ELSE
    ; load dequant parameters
    vld1.32         d1,    [r7]        ; d1[0] = dequant dc,d0[1-3] =dequant ac
    vshr.s16        d30,   d1,    #2   ; threshold value = dequant_ptr[i] >> 2
    vdup.16         d31,   d30[1]
  ENDIF
    ; load round parameters
    vld1.32         d2,    [r11]
    vdup.16         d3,    d2[1]
  IF $width = 32
    vrshr.s16       q1,    q1,   #1      ; round factor /2
  ENDIF
    mov             r10,   #0
    vdup.16         q14,   r10           ; q14 holds maximum of iscan value
    ; load  first 16 coeffs
    vld1.64         q5,    [r0]!
    vld1.64         q6,    [r0]!
    ; load  first 16 iscan values
    vld1.64         q3,    [r9]!
    vld1.64         q4,    [r9]!
    ; adding 1 to iscan values
    mov             r10,   #1
    vdup.16         q7,    r10
    vadd.s16        q3,    q3,   q7
    vadd.s16        q4,    q4,   q7
    ; threshold first block
    THRESHOLD_4X4_BLOCK 1
    cmp             r10,   #0             ; if all values are below threshold
    bne             quantize_first_4x4_block_$width$quant_type
    ; storing 0 to quant_ptr and dequant_ptr
    vst1.64         {q7},  [r5]!
    vst1.64         {q7},  [r5]!
    vst1.64         {q7},  [r6]!
    vst1.64         {q7},  [r6]!
    cmp             r1,    #16
    beq             findeob_$width$quant_type
    b               first_4x4_processed_$width$quant_type
    ; first block completed
    ; if block have values above threshold
quantize_first_4x4_block_$width$quant_type
    ; calling corresponding quantization according to width and type
  IF $width = 32
    QUANTIZE_4X4_BLOCK 1 $width $quant_type 15
  ELSE
    QUANTIZE_4X4_BLOCK 1 $width $quant_type 16
  ENDIF

    ; storing quantized values to qcoeff_ptr
    vst1.64         {q5},  [r5]!
    vst1.64         {q6},  [r5]!
    ;storing dequantized values to dqcoeff_ptr
    vst1.64         {q8},  [r6]!
    vst1.64         {q9},  [r6]!
    ; start processing rest of the blocks
    cmp r1, #16
    beq findeob_$width$quant_type
first_4x4_processed_$width$quant_type
    subs            r1,    r1,   #16
    vdup.16         d0,    d0[1]
    vdup.16         d1,    d1[1]
    vdup.16         d2,    d2[1]
    vdup.16         d3,    d2[1]
    vdup.32         d4,    d4[1]
process_4x4_$width$quant_type
    ; load 16 coeffs
    vld1.64         q5,    [r0]!
    vld1.64         q6,    [r0]!
    mov             r10,   #1
    vdup.16         q7,    r10
    ; load first 16 iscan values
    vld1.64         q3,    [r9]!
    vld1.64         q4,    [r9]!
    ; add 1 to iscan values
    vadd.s16        q3,    q3,   q7
    vadd.s16        q4,    q4,   q7
    THRESHOLD_4X4_BLOCK 0
    cmp             r10,   #0      ; if all values are less than threshold
    bne             quantize_$width$quant_type
    ; all values below threshold
    ; storing 0 to qcoeff_ptr
    vst1.64         {q7},  [r5]!
    vst1.64         {q7},  [r5]!
    ; storing 0 to dqcoeff_ptr
    vst1.64         {q7},  [r6]!
    vst1.64         {q7},  [r6]!
    b skip_4x4_block_$width$quant_type
    ; if block have values above threshold
quantize_$width$quant_type
    ; calling corresponding quantization according to width and type
  IF $width = 32
    QUANTIZE_4X4_BLOCK 0 $width $quant_type 15
  ELSE
    QUANTIZE_4X4_BLOCK 0 $width $quant_type 16
  ENDIF
    ; storing quantized values to qcoeff_ptr
    vst1.64         {q5},  [r5]!
    vst1.64         {q6},  [r5]!
    ; storing dequantized values to dqcoeff_ptr
    vst1.64         {q8},  [r6]!
    vst1.64         {q9},  [r6]!
skip_4x4_block_$width$quant_type
    subs            r1,    r1,   #16 ; 16 coeffs processed
    cmp             r1,    #0
    bgt             process_4x4_$width$quant_type
findeob_$width$quant_type
    FIND_EOB
    vst1.16         d28[0],[r8]
end_func_$width$quant_type
    vpop            {d8-d15}
    pop             {r4-r12, pc}
    MEND
|vp9_quantize_fp_32x32_neon| PROC
    QUANTIZE 32     fp_32x32
    ENDP
|vp9_quantize_b_32x32_neon| PROC
    QUANTIZE 32     quantize_b
    ENDP
|vp9_quantize_b_neon| PROC
    QUANTIZE 0      quantize_b
    ENDP

    END

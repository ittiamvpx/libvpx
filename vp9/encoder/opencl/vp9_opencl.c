/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */

#include "vp9/encoder/vp9_encoder.h"

#include "vp9/encoder/opencl/vp9_opencl.h"

#define OPENCL_DEVELOPER_MODE 1
#define BUILD_OPTION_LENGTH 128
#define INTEL_HD_GRAPHICS_ID 32902
#if ARCH_ARM
#define PREFIX_PATH "./"
#else
#define PREFIX_PATH "../../vp9/encoder/opencl/"
#endif

static const int pixel_rows_per_workitem_log2_inter_pred[GPU_BLOCK_SIZES]
                                                         = {3, 2, 0};

static const int pixel_rows_per_workitem_log2_motion_search[GPU_BLOCK_SIZES]
                                                            = {5, 3, 3};

static char *read_src(const char *src_file_name) {
  FILE *fp;
  int32_t err;
  int32_t size;
  char *src;

  fp = fopen(src_file_name, "rb");
  if (fp == NULL)
    return NULL;

  err = fseek(fp, 0, SEEK_END);
  if (err != 0)
    return NULL;


  size = ftell(fp);
  if (size < 0)
    return NULL;

  err = fseek(fp, 0, SEEK_SET);
  if (err != 0)
    return NULL;

  src = (char *)vpx_malloc(size + 1);
  if (src == NULL)
    return NULL;

  err = fread(src, 1, size, fp);
  if (err != size) {
    vpx_free(src);
    return NULL;
  }

  src[size] = '\0';

  return src;
}

static cl_ulong get_event_time_elapsed(cl_event event) {
  cl_ulong startTime, endTime;
  cl_int status = 0;

  status  = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START,
                                   sizeof(cl_ulong), &startTime, NULL);
  status |= clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END,
                                    sizeof(cl_ulong), &endTime, NULL);
  assert(status == CL_SUCCESS);
  return (endTime - startTime);
}


static void vp9_opencl_alloc_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_OPENCL *opencl = (VP9_OPENCL *)cpi->gpu.compute_framework;
  cl_int status;
  GPU_BLOCK_SIZE gpu_bsize;

  opencl->mi_rows = cm->mi_rows;
  opencl->mi_cols = cm->mi_cols;
  opencl->stride  = cm->frame_bufs[0].buf.y_stride;

  opencl->current_frame_size = cpi->lookahead->buf->img.buffer_alloc_sz;
  opencl->current_frame = clCreateBuffer(opencl->context,
                                         CL_MEM_READ_ONLY,
                                         opencl->current_frame_size,
                                         NULL,
                                         &status);
  if (status != CL_SUCCESS || opencl->current_frame == (cl_mem)0)
    goto fail;

  opencl->reference_frame_size = cm->frame_bufs[0].buf.buffer_alloc_sz;
  opencl->reference_frame = clCreateBuffer(opencl->context,
                                           CL_MEM_READ_ONLY,
                                           opencl->reference_frame_size,
                                           NULL,
                                           &status);
  if (status != CL_SUCCESS || opencl->reference_frame == (cl_mem)0)
    goto fail;

  opencl->rd_parameters = clCreateBuffer(opencl->context,
                                         CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                                         sizeof(GPU_RD_PARAMETERS),
                                         NULL,
                                         &status);
  if (status != CL_SUCCESS || opencl->rd_parameters == (cl_mem)0)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int block_cols = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
    const int block_rows = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);
    const int alloc_size = block_cols * block_rows;

    opencl->input_mv_size[gpu_bsize] = alloc_size * sizeof(GPU_INPUT);
    opencl->input_mv[gpu_bsize] = clCreateBuffer(
        opencl->context,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        opencl->input_mv_size[gpu_bsize],
        NULL,
        &status);
    if (status != CL_SUCCESS || opencl->input_mv[gpu_bsize] == (cl_mem)0)
      goto fail;

    opencl->output_rd_size[gpu_bsize] = alloc_size * sizeof(GPU_OUTPUT);
    opencl->output_rd[gpu_bsize] = clCreateBuffer(
        opencl->context,
        CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        opencl->output_rd_size[gpu_bsize],
        NULL,
        &status);
    if (status != CL_SUCCESS || opencl->output_rd[gpu_bsize] == (cl_mem)0)
      goto fail;

    status  = clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 0,
                             sizeof(cl_mem), &opencl->reference_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 1,
                             sizeof(cl_mem), &opencl->current_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 2,
                             sizeof(cl_int), &opencl->stride);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 3,
                             sizeof(cl_mem), &opencl->input_mv[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 4,
                             sizeof(cl_mem), &opencl->output_rd[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 5,
                             sizeof(cl_mem), &opencl->rd_parameters);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 6,
                             sizeof(cl_int), &opencl->mi_rows);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part0[gpu_bsize], 7,
                             sizeof(cl_int), &opencl->mi_cols);
    // For 32x32 block this parameter is ignored. Used only for other block
    // sizes
    status |= clSetKernelArg(
        opencl->vp9_pick_inter_mode_part0[gpu_bsize], 8,
        sizeof(cl_mem),
        &opencl->output_rd[gpu_bsize != GPU_BLOCK_32X32 ? gpu_bsize - 1 : 0]);

    if (status != CL_SUCCESS)
      goto fail;

    status  = clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 0,
                             sizeof(cl_mem), &opencl->reference_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 1,
                             sizeof(cl_mem), &opencl->current_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 2,
                             sizeof(cl_int), &opencl->stride);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 3,
                             sizeof(cl_mem), &opencl->input_mv[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 4,
                             sizeof(cl_mem), &opencl->output_rd[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 5,
                             sizeof(cl_mem), &opencl->rd_parameters);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 6,
                             sizeof(cl_int), &opencl->mi_rows);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part1[gpu_bsize], 7,
                             sizeof(cl_int), &opencl->mi_cols);
    // For 32x32 block this parameter is ignored. Used only for other block
    // sizes
    status |= clSetKernelArg(
        opencl->vp9_pick_inter_mode_part1[gpu_bsize], 8,
        sizeof(cl_mem),
        &opencl->output_rd[gpu_bsize != GPU_BLOCK_32X32 ? gpu_bsize - 1 : 0]);

    if (status != CL_SUCCESS)
      goto fail;

    status  = clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 0,
                             sizeof(cl_mem), &opencl->reference_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 1,
                             sizeof(cl_mem), &opencl->current_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 2,
                             sizeof(cl_int), &opencl->stride);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 3,
                             sizeof(cl_mem), &opencl->input_mv[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 4,
                             sizeof(cl_mem), &opencl->output_rd[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 5,
                             sizeof(cl_mem), &opencl->rd_parameters);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 6,
                             sizeof(cl_int), &opencl->mi_rows);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part2[gpu_bsize], 7,
                             sizeof(cl_int), &opencl->mi_cols);
    // For 32x32 block this parameter is ignored. Used only for other block
    // sizes
    status |= clSetKernelArg(
        opencl->vp9_pick_inter_mode_part2[gpu_bsize], 8,
        sizeof(cl_mem),
        &opencl->output_rd[gpu_bsize != GPU_BLOCK_32X32 ? gpu_bsize - 1 : 0]);

    if (status != CL_SUCCESS)
      goto fail;

    status  = clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 0,
                             sizeof(cl_mem), &opencl->reference_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 1,
                             sizeof(cl_mem), &opencl->current_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 2,
                             sizeof(cl_int), &opencl->stride);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 3,
                             sizeof(cl_mem), &opencl->input_mv[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 4,
                             sizeof(cl_mem), &opencl->output_rd[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 5,
                             sizeof(cl_mem), &opencl->rd_parameters);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 6,
                             sizeof(cl_int), &opencl->mi_rows);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part3[gpu_bsize], 7,
                             sizeof(cl_int), &opencl->mi_cols);
    // For 32x32 block this parameter is ignored. Used only for other block
    // sizes
    status |= clSetKernelArg(
        opencl->vp9_pick_inter_mode_part3[gpu_bsize], 8,
        sizeof(cl_mem),
        &opencl->output_rd[gpu_bsize != GPU_BLOCK_32X32 ? gpu_bsize - 1 : 0]);

    if (status != CL_SUCCESS)
      goto fail;

    status  = clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 0,
                             sizeof(cl_mem), &opencl->reference_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 1,
                             sizeof(cl_mem), &opencl->current_frame);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 2,
                             sizeof(cl_int), &opencl->stride);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 3,
                             sizeof(cl_mem), &opencl->input_mv[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 4,
                             sizeof(cl_mem), &opencl->output_rd[gpu_bsize]);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 5,
                             sizeof(cl_mem), &opencl->rd_parameters);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 6,
                             sizeof(cl_int), &opencl->mi_rows);
    status |= clSetKernelArg(opencl->vp9_pick_inter_mode_part4[gpu_bsize], 7,
                             sizeof(cl_int), &opencl->mi_cols);
    // For 32x32 block this parameter is ignored. Used only for other block
    // sizes
    status |= clSetKernelArg(
        opencl->vp9_pick_inter_mode_part4[gpu_bsize], 8,
        sizeof(cl_mem),
        &opencl->output_rd[gpu_bsize != GPU_BLOCK_32X32 ? gpu_bsize - 1 : 0]);

    if (status != CL_SUCCESS)
      goto fail;

  }

  status  = clSetKernelArg(opencl->vp9_is_8x8_required, 0,
                           sizeof(cl_mem), &opencl->input_mv[GPU_BLOCK_32X32]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 1,
                           sizeof(cl_mem), &opencl->output_rd[GPU_BLOCK_32X32]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 2,
                           sizeof(cl_mem), &opencl->input_mv[GPU_BLOCK_16X16]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 3,
                           sizeof(cl_mem), &opencl->output_rd[GPU_BLOCK_16X16]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 4,
                           sizeof(cl_mem), &opencl->input_mv[GPU_BLOCK_8X8]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 5,
                           sizeof(cl_mem), &opencl->output_rd[GPU_BLOCK_8X8]);
  status |= clSetKernelArg(opencl->vp9_is_8x8_required, 6,
                           sizeof(cl_mem), &opencl->rd_parameters);
  if (status != CL_SUCCESS)
    goto fail;

  return;
fail:
  // TODO(karthick-ittiam): The error set below is ignored by the encoder. This
  // error needs to be handled appropriately. Adding assert as a temporary fix.
  vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                     "Failed to allocate OpenCL buffers");
  assert(0);
  return;
}

static void vp9_opencl_free_buffers(VP9_COMP *cpi) {
  VP9_OPENCL *opencl = (VP9_OPENCL *)cpi->gpu.compute_framework;
  cl_int status;
  GPU_BLOCK_SIZE gpu_bsize;

  status = clReleaseMemObject(opencl->current_frame);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseMemObject(opencl->reference_frame);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseMemObject(opencl->rd_parameters);
  if (status != CL_SUCCESS)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    status = clReleaseMemObject(opencl->input_mv[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseMemObject(opencl->output_rd[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
  }

  return;
fail:
  assert(0);
  return;
}

static void *vp9_opencl_acquire_rd_parameters(VP9_COMP *cpi) {
  VP9_OPENCL * const opencl = (VP9_OPENCL *) cpi->gpu.compute_framework;
  cl_int status;
  if (opencl->rd_parameters_mapped == NULL) {
    opencl->rd_parameters_mapped = clEnqueueMapBuffer(
        opencl->cmd_queue, opencl->rd_parameters,
        CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE, 0, sizeof(GPU_RD_PARAMETERS), 0, NULL, NULL,
        &status);
    assert(status == CL_SUCCESS);
  }
  return opencl->rd_parameters_mapped;
}

static void *vp9_opencl_acquire_input_buffer(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize) {
  VP9_OPENCL *const opencl = (VP9_OPENCL *) cpi->gpu.compute_framework;
  cl_int status;
  if (opencl->input_mv_mapped[gpu_bsize] == NULL) {
    opencl->input_mv_mapped[gpu_bsize] = clEnqueueMapBuffer(
        opencl->cmd_queue, opencl->input_mv[gpu_bsize],
        CL_TRUE,
        CL_MAP_READ | CL_MAP_WRITE, 0, opencl->input_mv_size[gpu_bsize], 0,
        NULL, NULL, &status);
    assert(status == CL_SUCCESS);
  }
  return opencl->input_mv_mapped[gpu_bsize];
}

static GPU_OUTPUT* vp9_opencl_execute(VP9_COMP *cpi,
                        uint8_t* reference_frame, uint8_t* current_frame,
                        GPU_BLOCK_SIZE gpu_bsize) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_OPENCL *const opencl = (VP9_OPENCL *)cpi->gpu.compute_framework;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  cl_int status;
  const size_t workitem_size[2] = { NUM_PIXELS_PER_WORKITEM, 1 };
  size_t MB_size[2];
  size_t local_size[2];
  size_t local_size_motion_search[2], local_size_inter_pred[2];
  size_t global_size[2];
  const int b_width_in_pixels_log2 = b_width_log2(bsize) + 2;
  const int b_height_in_pixels_log2 = b_height_log2(bsize) + 2;
  const int b_width_mask = (1 << b_width_in_pixels_log2) - 1;
  const int b_height_mask = (1 << b_height_in_pixels_log2) - 1;
  int num_block_cols = cm->width >> b_width_in_pixels_log2;
  int num_block_rows = cm->height >> b_height_in_pixels_log2;
#if OPENCL_PROFILING
  cl_event event[NUM_KERNELS];
#endif
  cl_event *event_ptr[NUM_KERNELS];
  int i;

  for(i = 0; i < NUM_KERNELS; i++) {
#if OPENCL_PROFILING
    event_ptr[i] = &event[i];
#else
    event_ptr[i] = NULL;
#endif
  }


  // If width or Height is not a multiple of block size
  if (cm->width & b_width_mask)
    num_block_cols++;

  if (cm->height & b_height_mask)
    num_block_rows++;


  MB_size[0] = 1 << b_width_in_pixels_log2;
  MB_size[1] = 1 << b_height_in_pixels_log2;

  local_size[0] = MB_size[0] / workitem_size[0];
  local_size[1] = MB_size[1] / workitem_size[1];

  status = clEnqueueWriteBuffer(opencl->cmd_queue, opencl->reference_frame,
                                CL_TRUE, 0,
                                opencl->reference_frame_size,
                                (const void *) reference_frame,
                                0, NULL, NULL);
  assert(status == CL_SUCCESS);

  status = clEnqueueWriteBuffer(opencl->cmd_queue, opencl->current_frame,
                                CL_TRUE, 0,
                                opencl->current_frame_size,
                                (const void *) current_frame,
                                0, NULL, NULL);
  assert(status == CL_SUCCESS);

  if(opencl->input_mv_mapped[gpu_bsize] != NULL)
  {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     opencl->input_mv[gpu_bsize],
                                     opencl->input_mv_mapped[gpu_bsize],
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    opencl->input_mv_mapped[gpu_bsize] = NULL;
  }

  if (opencl->output_rd_mapped[gpu_bsize] != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     opencl->output_rd[gpu_bsize],
                                     opencl->output_rd_mapped[gpu_bsize],
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    opencl->output_rd_mapped[gpu_bsize] = NULL;
  }
  if(opencl->rd_parameters_mapped != NULL)
  {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     opencl->rd_parameters,
                                     opencl->rd_parameters_mapped,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    opencl->rd_parameters_mapped = NULL;
  }

  if (gpu_bsize == GPU_BLOCK_8X8) {
    const int log2_of_32 = b_width_log2(BLOCK_32X32) + 2;
    int num_32x32_cols = cm->width >> log2_of_32;
    int num_32x32_rows = cm->height >> log2_of_32;
    global_size[0] = num_32x32_cols;
    global_size[1] = num_32x32_rows;

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    opencl->vp9_is_8x8_required,
                                    2,
                                    NULL,
                                    global_size, NULL,
                                    0, NULL, NULL);
    assert(status == CL_SUCCESS);
  }

  local_size_motion_search[0] = local_size[0];
  local_size_motion_search[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_motion_search[gpu_bsize];

  global_size[0] = num_block_cols * local_size_motion_search[0];
  global_size[1] = num_block_rows * local_size_motion_search[1];

  if(gpu_bsize == GPU_BLOCK_8X8)
    assert(local_size_motion_search[0] * local_size_motion_search[1] == 1);

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue, opencl->vp9_pick_inter_mode_part0[gpu_bsize],
      2,
      NULL,
      global_size,
      (gpu_bsize == GPU_BLOCK_8X8) ? NULL : local_size_motion_search,
      0, NULL, event_ptr[0]);
  assert(status == CL_SUCCESS);

  global_size[0] = num_block_cols;
  global_size[1] = num_block_rows;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  opencl->vp9_pick_inter_mode_part1[gpu_bsize],
                                  2,
                                  NULL,
                                  global_size, NULL,
                                  0, NULL, event_ptr[1]);
  assert(status == CL_SUCCESS);

  global_size[0] = num_block_cols * local_size_motion_search[0];
  global_size[1] = num_block_rows * local_size_motion_search[1];

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue, opencl->vp9_pick_inter_mode_part2[gpu_bsize],
      2,
      NULL,
      global_size,
      (gpu_bsize == GPU_BLOCK_8X8) ? NULL : local_size_motion_search,
      0, NULL, event_ptr[2]);
  assert(status == CL_SUCCESS);

  local_size_inter_pred[0] = local_size[0];
  local_size_inter_pred[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_inter_pred[gpu_bsize];

  global_size[0] = num_block_cols * local_size_inter_pred[0];
  global_size[1] = num_block_rows * local_size_inter_pred[1];

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  opencl->vp9_pick_inter_mode_part3[gpu_bsize],
                                  2,
                                  NULL,
                                  global_size, local_size_inter_pred,
                                  0, NULL, event_ptr[3]);
  assert(status == CL_SUCCESS);

  global_size[0] = num_block_cols;
  global_size[1] = num_block_rows;
  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  opencl->vp9_pick_inter_mode_part4[gpu_bsize],
                                  2,
                                  NULL,
                                  global_size, NULL,
                                  0, NULL, event_ptr[4]);
  assert(status == CL_SUCCESS);

  opencl->output_rd_mapped[gpu_bsize] = clEnqueueMapBuffer(
      opencl->cmd_queue, opencl->output_rd[gpu_bsize],
      CL_TRUE, CL_MAP_READ, 0,
      opencl->output_rd_size[gpu_bsize],
      0, NULL, NULL,
      &status);
  assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
  for(i = 0; i < NUM_KERNELS; i++)
  {
    cl_ulong time_elapsed;
    clWaitForEvents(1, event_ptr[i]);
    time_elapsed = get_event_time_elapsed(*event_ptr[i]);
    opencl->total_time_taken[gpu_bsize][i] += time_elapsed / 1000;
    clReleaseEvent(*event_ptr[i]);
  }
#endif
  return opencl->output_rd_mapped[gpu_bsize];
}

static void vp9_opencl_remove(VP9_COMP *cpi) {
  VP9_OPENCL *const opencl = (VP9_OPENCL *)cpi->gpu.compute_framework;
  GPU_BLOCK_SIZE gpu_bsize;
  cl_int status;
#if OPENCL_PROFILING
  int i;
  cl_ulong total[NUM_KERNELS] = {0};
  cl_ulong grand_total = 0;
  fprintf(stdout, "\nOPENCL PROFILE RESULTS\n");
#endif

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
#if OPENCL_PROFILING
    fprintf(stdout, "\nBlock size idx = %d\n", gpu_bsize);
    for(i = 0; i < NUM_KERNELS; i++)
    {
      total[i] += opencl->total_time_taken[gpu_bsize][i];
      fprintf(stdout, "\tKernel %d - TOTAL = %"PRIu64" microseconds\n", i,
              opencl->total_time_taken[gpu_bsize][i]);
    }
#endif

    status = clReleaseKernel(opencl->vp9_pick_inter_mode_part0[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(opencl->vp9_pick_inter_mode_part1[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(opencl->vp9_pick_inter_mode_part2[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(opencl->vp9_pick_inter_mode_part3[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(opencl->vp9_pick_inter_mode_part4[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
  }
#if OPENCL_PROFILING
  fprintf(stdout, "\nTOTAL FOR ALL BLOCK SIZES\n");
  for(i = 0; i < NUM_KERNELS; i++)
  {
    grand_total += total[i];
    fprintf(stdout,
            "\tKernel %d - TOTAL ALL BLOCK SIZES = %"PRIu64" microseconds\n",
            i, total[i]);
  }
  fprintf(stdout, "\nGRAND TOTAL = %"PRIu64"\n", grand_total);
#endif

  status = clReleaseKernel(opencl->vp9_is_8x8_required);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseCommandQueue(opencl->cmd_queue);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseContext(opencl->context);
  if (status != CL_SUCCESS)
    goto fail;
  return;
fail:
  assert(0);
  return;
}

int vp9_opencl_init(VP9_GPU *gpu) {
  VP9_OPENCL *opencl;
  cl_int status;
  cl_uint num_platforms = 0;
  cl_platform_id platform;
  cl_uint num_devices = 0;
  cl_device_id device;
  cl_uint vendor_id;
  cl_command_queue_properties command_queue_properties = 0;
  cl_program program;
  // TODO(karthick-ittiam) : Pass this prefix path as an input from testbench
  const char *kernel_file_name= PREFIX_PATH"vp9_pick_inter_mode.cl";

  // TODO(karthick-ittiam) : Fix this hardcoding
  char build_options_combined_motion_search[GPU_BLOCK_SIZES][BUILD_OPTION_LENGTH] = {
      "-DBLOCK_SIZE_IN_PIXELS=32 -DPIXEL_ROWS_PER_WORKITEM=32 -DINTEL_HD_GRAPHICS=0",
      "-DBLOCK_SIZE_IN_PIXELS=16 -DPIXEL_ROWS_PER_WORKITEM=8 -DINTEL_HD_GRAPHICS=0",
      "-DBLOCK_SIZE_IN_PIXELS=8 -DPIXEL_ROWS_PER_WORKITEM=8 -DINTEL_HD_GRAPHICS=0"
  };

  const char build_options[GPU_BLOCK_SIZES][BUILD_OPTION_LENGTH] = {
      "-DBLOCK_SIZE_IN_PIXELS=32 -DPIXEL_ROWS_PER_WORKITEM=8",
      "-DBLOCK_SIZE_IN_PIXELS=16 -DPIXEL_ROWS_PER_WORKITEM=4",
      "-DBLOCK_SIZE_IN_PIXELS=8 -DPIXEL_ROWS_PER_WORKITEM=1" };
  char *kernel_src;
  GPU_BLOCK_SIZE gpu_bsize;
  gpu->compute_framework = vpx_calloc(1, sizeof(VP9_OPENCL));
  gpu->alloc_buffers = vp9_opencl_alloc_buffers;
  gpu->free_buffers = vp9_opencl_free_buffers;
  gpu->acquire_input_buffer = vp9_opencl_acquire_input_buffer;
  gpu->acquire_rd_parameters = vp9_opencl_acquire_rd_parameters;
  gpu->execute = vp9_opencl_execute;
  gpu->remove = vp9_opencl_remove;
  opencl = gpu->compute_framework;

#if OPENCL_PROFILING
  command_queue_properties = CL_QUEUE_PROFILING_ENABLE;
#endif

  // Get the number of platforms in the system.
  status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS || num_platforms == 0)
    goto fail;

  // Get the platform ID for one platform
  status = clGetPlatformIDs(1, &platform, NULL);
  if (status != CL_SUCCESS)
    goto fail;

  // Get the number of devices available on the platform
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &num_devices);
  if (status != CL_SUCCESS || num_devices == 0)
    goto fail;

  // Get the device ID for one device
  status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  if (status != CL_SUCCESS)
    goto fail;

  status = clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID,
                           sizeof(cl_uint),
                           &vendor_id,
                           NULL);
  if (status != CL_SUCCESS)
    goto fail;

  // Create OpenCL context for one device
  opencl->context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  if (status != CL_SUCCESS || opencl->context == NULL)
    goto fail;

  // Create a command queue for the device
  opencl->cmd_queue = clCreateCommandQueue(opencl->context, device,
                                           command_queue_properties,
                                           &status);
  if (status != CL_SUCCESS || opencl->cmd_queue == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    // Read kernel source files
    kernel_src = read_src(kernel_file_name);
    if (kernel_src == NULL)
      goto fail;

    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    vpx_free(kernel_src);
    if (status != CL_SUCCESS)
      goto fail;

    if(vendor_id == INTEL_HD_GRAPHICS_ID) {
      int string_length = strlen(build_options_combined_motion_search[gpu_bsize]);
      build_options_combined_motion_search[gpu_bsize][string_length - 1] = '1';
    }
    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options_combined_motion_search[gpu_bsize],
                            NULL, NULL);
    if (status != CL_SUCCESS) {
    // Enable this if you are a OpenCL developer and need to print the build
    // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
          device,
          CL_PROGRAM_BUILD_LOG,
          0,
          NULL,
          &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
          device,
          CL_PROGRAM_BUILD_LOG,
          build_log_size,
          build_log,
          NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }
    opencl->vp9_pick_inter_mode_part0[gpu_bsize] = clCreateKernel(
        program, "vp9_pick_inter_mode_part0", &status);
    if (status != CL_SUCCESS)
      goto fail;

    opencl->vp9_pick_inter_mode_part2[gpu_bsize] = clCreateKernel(
        program, "vp9_pick_inter_mode_part2", &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    // Read kernel source files
    kernel_src = read_src(kernel_file_name);
    if (kernel_src == NULL)
      goto fail;

    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    vpx_free(kernel_src);
    if (status != CL_SUCCESS)
      goto fail;


    // Build the program
    status = clBuildProgram(program, 1, &device, build_options[gpu_bsize],
                            NULL, NULL);
    if (status != CL_SUCCESS) {
    // Enable this if you are a OpenCL developer and need to print the build
    // errors of the OpenCL kernel
#if OPENCL_DEVELOPER_MODE
      uint8_t *build_log;
      size_t build_log_size;

      clGetProgramBuildInfo(program,
          device,
          CL_PROGRAM_BUILD_LOG,
          0,
          NULL,
          &build_log_size);
      build_log = (uint8_t*)vpx_malloc(build_log_size);
      if (build_log == NULL)
        goto fail;

      clGetProgramBuildInfo(program,
          device,
          CL_PROGRAM_BUILD_LOG,
          build_log_size,
          build_log,
          NULL);
      build_log[build_log_size-1] = '\0';
      fprintf(stderr, "Build Log:\n%s\n", build_log);
      vpx_free(build_log);
#endif
      goto fail;
    }
    opencl->vp9_pick_inter_mode_part1[gpu_bsize] = clCreateKernel(
        program, "vp9_pick_inter_mode_part1", &status);
    if (status != CL_SUCCESS)
      goto fail;

    opencl->vp9_pick_inter_mode_part3[gpu_bsize] = clCreateKernel(
        program, "vp9_pick_inter_mode_part3", &status);
    if (status != CL_SUCCESS)
      goto fail;

    opencl->vp9_pick_inter_mode_part4[gpu_bsize] = clCreateKernel(
        program, "vp9_pick_inter_mode_part4", &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

  opencl->vp9_is_8x8_required = clCreateKernel(
      program, "vp9_is_8x8_required", &status);
  if (status != CL_SUCCESS)
    goto fail;

  return 0;
fail:
  return 1;
}


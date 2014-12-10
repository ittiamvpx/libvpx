/*
 *  Copyright (c) 2014 The WebM project authors. All Rights Reserved.
 *
 *  Use of this source code is governed by a BSD-style license
 *  that can be found in the LICENSE file in the root of the source
 *  tree. An additional intellectual property rights grant can be found
 *  in the file PATENTS.  All contributing project authors may
 *  be found in the AUTHORS file in the root of the source tree.
 */


#include "vp9/common/opencl/vp9_opencl.h"

#include "vp9/encoder/opencl/vp9_eopencl.h"

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

#if OPENCL_PROFILING
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
#endif

static void vp9_opencl_alloc_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *gpu = &cpi->egpu;
  VP9_EOPENCL *eopencl = gpu->compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int block_cols = (cm->sb_cols * num_mxn_blocks_wide_lookup[bsize]);
    const int block_rows = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);
    const int alloc_size = block_cols * block_rows;
    opencl_buffer *gpuinput_b_args = &eopencl->gpu_input[gpu_bsize];
    opencl_buffer *gpuoutput_b_args = &eopencl->gpu_output[gpu_bsize];
    opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters[gpu_bsize];
    int subframe_idx;

    // alloc buffer for gpu input
    rdopt_parameters->size = sizeof(GPU_RD_PARAMETERS);
    rdopt_parameters->opencl_mem = clCreateBuffer(
        opencl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        rdopt_parameters->size, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    gpuinput_b_args->size = alloc_size * sizeof(GPU_INPUT);
    gpuinput_b_args->opencl_mem = clCreateBuffer(
        opencl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        gpuinput_b_args->size, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    // alloc buffer for gpu output
    gpuoutput_b_args->size = alloc_size * sizeof(GPU_OUTPUT);
    gpuoutput_b_args->opencl_mem = clCreateBuffer(
        opencl->context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR,
        gpuoutput_b_args->size, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    // create sub buffers
    for (subframe_idx = 0; subframe_idx < MAX_SUB_FRAMES; ++subframe_idx) {
      cl_buffer_region sf_region;
      SubFrameInfo subframe;
      int block_row_offset;
      int block_rows_sf;
      int alloc_size_sf;

      vp9_subframe_init(&subframe, cm, subframe_idx);

      block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);
      block_rows_sf = (mi_cols_aligned_to_sb(subframe.mi_row_end) -
          subframe.mi_row_start) >> mi_height_log2(bsize);

      alloc_size_sf = block_cols * block_rows_sf;

      sf_region.origin = block_row_offset * block_cols * sizeof(GPU_INPUT);
      sf_region.size = alloc_size_sf * sizeof(GPU_INPUT);
      eopencl->gpu_input_sub_buffer[gpu_bsize][subframe_idx] =
          clCreateSubBuffer(gpuinput_b_args->opencl_mem,
                            CL_MEM_READ_ONLY,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &sf_region, &status);
      if (status != CL_SUCCESS)
        goto fail;

      sf_region.origin = block_row_offset * block_cols * sizeof(GPU_OUTPUT);
      sf_region.size = alloc_size_sf * sizeof(GPU_OUTPUT);
      eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_idx] =
          clCreateSubBuffer(gpuoutput_b_args->opencl_mem,
                            CL_MEM_WRITE_ONLY,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &sf_region, &status);
      if (status != CL_SUCCESS)
        goto fail;
    }
  }
  return;

fail:
  // TODO(karthick-ittiam): The error set below is ignored by the encoder. This
  // error needs to be handled appropriately. Adding assert as a temporary fix.
  vpx_internal_error(&cm->error, VPX_CODEC_MEM_ERROR,
                     "Failed to allocate OpenCL buffers");
  assert(0);
}

static void vp9_opencl_free_buffers(VP9_COMP *cpi) {
  VP9_EOPENCL *eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;
  cl_event event;
  GPU_BLOCK_SIZE gpu_bsize;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];
    opencl_buffer *gpu_output = &eopencl->gpu_output[gpu_bsize];
    opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters[gpu_bsize];
    int subframe_id;

    for (subframe_id = 0; subframe_id < MAX_SUB_FRAMES; subframe_id++) {
      status = clReleaseMemObject(
          eopencl->gpu_input_sub_buffer[gpu_bsize][subframe_id]);
      if (status != CL_SUCCESS)
        goto fail;

      status = clReleaseMemObject(
          eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_id]);
      if (status != CL_SUCCESS)
        goto fail;
    }

    if (gpu_input->mapped_pointer != NULL) {
      status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                       gpu_input->opencl_mem,
                                       gpu_input->mapped_pointer,
                                       0, NULL, &event);
      status |= clWaitForEvents(1, &event);
      if (status != CL_SUCCESS)
        goto fail;
      status = clReleaseEvent(event);
      if (status != CL_SUCCESS)
        goto fail;
      gpu_input->mapped_pointer = NULL;
    }
    status = clReleaseMemObject(gpu_input->opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;

    if (rdopt_parameters->mapped_pointer != NULL) {
      status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                       rdopt_parameters->opencl_mem,
                                       rdopt_parameters->mapped_pointer,
                                       0, NULL, &event);
      status |= clWaitForEvents(1, &event);
      if (status != CL_SUCCESS)
        goto fail;
      status = clReleaseEvent(event);
      if (status != CL_SUCCESS)
        goto fail;
      rdopt_parameters->mapped_pointer = NULL;
    }
    status = clReleaseMemObject(rdopt_parameters->opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;

    if (gpu_output->mapped_pointer != NULL) {
      status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                       gpu_output->opencl_mem,
                                       gpu_output->mapped_pointer,
                                       0, NULL, &event);
      status |= clWaitForEvents(1, &event);
      if (status != CL_SUCCESS)
        goto fail;
      status = clReleaseEvent(event);
      if (status != CL_SUCCESS)
        goto fail;
      gpu_output->mapped_pointer = NULL;
    }
    status = clReleaseMemObject(gpu_output->opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
  }
  return;

fail:
  // TODO(karthick-ittiam): The error set below is ignored by the encoder. This
  // error needs to be handled appropriately. Adding assert as a temporary fix.
  vpx_internal_error(&cpi->common.error, VPX_CODEC_MEM_ERROR,
                     "Failed to release OpenCL metadata buffers");
  assert(0);
}

static int vp9_opencl_acquire_buffer(VP9_COMP *cpi, cl_mem *opencl_mem,
                                     size_t size, void **mapped_pointer) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;

  if (*mapped_pointer == NULL) {
    *mapped_pointer =
        clEnqueueMapBuffer(opencl->cmd_queue_memory, *opencl_mem, CL_TRUE,
                           CL_MAP_READ | CL_MAP_WRITE, 0, size, 0, NULL, NULL,
                           &status);
    if (status != CL_SUCCESS)
      goto fail;
  }
  return 0;

fail:
  return 1;
}

static void vp9_opencl_acquire_rd_param_buffer(VP9_COMP *cpi,
                                               GPU_BLOCK_SIZE gpu_bsize,
                                               void **host_ptr) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters[gpu_bsize];

  if (!vp9_opencl_acquire_buffer(cpi, &rdopt_parameters->opencl_mem,
                                 rdopt_parameters->size,
                                 &rdopt_parameters->mapped_pointer)) {
    *host_ptr = rdopt_parameters->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_acquire_input_buffer(VP9_COMP *cpi,
                                            GPU_BLOCK_SIZE gpu_bsize,
                                            void **host_ptr) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];

  if (!vp9_opencl_acquire_buffer(cpi, &gpu_input->opencl_mem,
                                 gpu_input->size,
                                 &gpu_input->mapped_pointer)) {
    *host_ptr = gpu_input->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_acquire_output_buffer(VP9_COMP *cpi,
                                             GPU_BLOCK_SIZE gpu_bsize,
                                             void **host_ptr) {
  VP9_EOPENCL *const opencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_output = &opencl->gpu_output[gpu_bsize];

  if (!vp9_opencl_acquire_buffer(cpi, &gpu_output->opencl_mem,
                                 gpu_output->size,
                                 &gpu_output->mapped_pointer)) {
    *host_ptr = gpu_output->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_set_kernel_args(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                                       int sub_frame_idx) {
  VP9_COMMON *cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  YV12_BUFFER_CONFIG *img_src = cpi->Source;
  YV12_BUFFER_CONFIG *frm_ref = get_ref_frame_buffer(cpi, LAST_FRAME);
  GPU_BLOCK_SIZE gpu_parent_bsize = gpu_bsize - 1;
  cl_mem *gpu_ip = &eopencl->gpu_input[gpu_bsize].opencl_mem;
  cl_mem *gpu_op = &eopencl->gpu_output[gpu_bsize].opencl_mem;
  cl_mem *rdopt_parameters = &eopencl->rdopt_parameters[gpu_bsize].opencl_mem;
  cl_mem *gpu_op_parent = gpu_bsize != GPU_BLOCK_32X32 ?
      &eopencl->gpu_output[gpu_parent_bsize].opencl_mem : NULL;
  cl_mem *gpu_ip_subframe =
      &eopencl->gpu_input_sub_buffer[gpu_bsize][sub_frame_idx];
  cl_mem *gpu_op_subframe =
      &eopencl->gpu_output_sub_buffer[gpu_bsize][sub_frame_idx];
  cl_mem *gpu_op_subframe_parent = gpu_bsize != GPU_BLOCK_32X32 ?
      &eopencl->gpu_output_sub_buffer[gpu_parent_bsize][sub_frame_idx] : NULL;
  cl_int mi_rows = cm->mi_rows;
  cl_int mi_cols = cm->mi_cols;
  cl_int y_stride = cm->frame_bufs[0].buf.y_stride;
  int b_is_8x8 = (gpu_bsize == GPU_BLOCK_8X8);
  cl_int status;

  status = clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 3,
                           sizeof(cl_mem), b_is_8x8 ? gpu_ip : gpu_ip_subframe);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 4,
                           sizeof(cl_mem), b_is_8x8 ? gpu_op : gpu_op_subframe);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 8,
                           sizeof(cl_mem),
                           b_is_8x8 ? gpu_op_parent : gpu_op_subframe_parent);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 8,
                           sizeof(cl_mem), gpu_op_parent);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 3,
                           sizeof(cl_mem), b_is_8x8 ? gpu_ip : gpu_ip_subframe);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 4,
                           sizeof(cl_mem), b_is_8x8 ? gpu_op : gpu_op_subframe);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  status |= clSetKernelArg(eopencl->sub_pixel_search[gpu_bsize], 8,
                           sizeof(cl_mem),
                           b_is_8x8 ? gpu_op_parent : gpu_op_subframe_parent);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip_subframe);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op_subframe);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 8,
                           sizeof(cl_mem), gpu_op_subframe_parent);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 8,
                           sizeof(cl_mem), gpu_op_parent);
  assert(status == CL_SUCCESS);

  if (gpu_bsize == GPU_BLOCK_8X8) {
    status = clSetKernelArg(eopencl->vp9_is_8x8_required, 0, sizeof(cl_mem),
                            &eopencl->gpu_input[GPU_BLOCK_32X32].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 1, sizeof(cl_mem),
                             &eopencl->gpu_output[GPU_BLOCK_32X32].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 2, sizeof(cl_mem),
                             &eopencl->gpu_input[GPU_BLOCK_16X16].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 3, sizeof(cl_mem),
                             &eopencl->gpu_output[GPU_BLOCK_16X16].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 4, sizeof(cl_mem),
                             &eopencl->gpu_input[GPU_BLOCK_8X8].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 5, sizeof(cl_mem),
                             &eopencl->gpu_output[GPU_BLOCK_8X8].opencl_mem);
    status |= clSetKernelArg(eopencl->vp9_is_8x8_required, 6, sizeof(cl_mem),
                             rdopt_parameters);
    assert(status == CL_SUCCESS);
  }
}

static void vp9_opencl_enc_sync_read(VP9_COMP *cpi, cl_int event_id) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  cl_int status;

  assert(event_id < MAX_SUB_FRAMES);
  status = clWaitForEvents(1, &eopencl->event[event_id]);
  if (status != CL_SUCCESS)
    vpx_internal_error(&cpi->common.error, VPX_CODEC_ERROR,
                       "Wait for event failed");
  status = clReleaseEvent(eopencl->event[event_id]);
  if (status != CL_SUCCESS)
    vpx_internal_error(&cpi->common.error, VPX_CODEC_ERROR,
                       "Release event failed");
}

static void vp9_opencl_execute(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                               int subframe_idx) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;

  VP9_COMMON *const cm = &cpi->common;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];
  opencl_buffer *gpu_output = &eopencl->gpu_output[gpu_bsize];
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters[gpu_bsize];
  YV12_BUFFER_CONFIG *img_src = cpi->Source;
  YV12_BUFFER_CONFIG *frm_ref = get_ref_frame_buffer(cpi, LAST_FRAME);

  const int b_width_in_pixels_log2 = b_width_log2(bsize) + 2;
  const int b_width_in_pixels = 1 << b_width_in_pixels_log2;
  const int b_width_mask = b_width_in_pixels - 1;
  const int b_height_in_pixels_log2 = b_height_log2(bsize) + 2;
  const int b_height_in_pixels = 1 << b_height_in_pixels_log2;
  const int b_height_mask = b_height_in_pixels - 1;

  SubFrameInfo subframe;
  int num_block_rows, num_block_cols;
  int block_row_offset;
  int subframe_height;

  const size_t workitem_size[2] = {NUM_PIXELS_PER_WORKITEM, 1};
  size_t local_size[2];
  size_t global_size[2];
  size_t global_offset[2];
  size_t local_size_motion_search[2], local_size_inter_pred[2];

  cl_int status = CL_SUCCESS;

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

  vp9_subframe_init(&subframe, cm, subframe_idx);
  block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);

  subframe_height = (subframe.mi_row_end - subframe.mi_row_start) << MI_SIZE_LOG2;
  num_block_rows = subframe_height >> b_height_in_pixels_log2;
  num_block_cols = cm->width >> b_width_in_pixels_log2;

  // If width or Height is not a multiple of block size
  if (cm->width & b_width_mask)
    num_block_cols++;

  if (subframe_idx == MAX_SUB_FRAMES - 1)
    if (cm->height & b_height_mask)
      num_block_rows++;

  vp9_opencl_set_kernel_args(cpi, gpu_bsize, subframe_idx);

  (void)status;

  if (rdopt_parameters->mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     rdopt_parameters->opencl_mem,
                                     rdopt_parameters->mapped_pointer,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    rdopt_parameters->mapped_pointer = NULL;
  }

  if (gpu_input->mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     gpu_input->opencl_mem,
                                     gpu_input->mapped_pointer,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    gpu_input->mapped_pointer = NULL;
  }

  if (gpu_output->mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     gpu_output->opencl_mem,
                                     gpu_output->mapped_pointer,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    gpu_output->mapped_pointer = NULL;
  }

  if (img_src->buffer_alloc != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     img_src->gpu_mem,
                                     img_src->buffer_alloc,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    img_src->buffer_alloc = img_src->y_buffer = img_src->u_buffer =
        img_src->v_buffer = NULL;
  }
  if (frm_ref->buffer_alloc != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     frm_ref->gpu_mem,
                                     frm_ref->buffer_alloc,
                                     0, NULL, NULL);
    assert(status == CL_SUCCESS);
    frm_ref->buffer_alloc = frm_ref->y_buffer = frm_ref->u_buffer =
        frm_ref->v_buffer = NULL;
  }

  // launch 8x8 decision kernel
  if (gpu_bsize == GPU_BLOCK_8X8) {
    const int log2_of_32 = b_width_log2(BLOCK_32X32) + 2;
    int subframe_offset = subframe.mi_row_start >> mi_height_log2(BLOCK_32X32);

    global_size[0] = cm->width >> log2_of_32;
    global_size[1] = subframe_height >> log2_of_32;

    global_offset[0] = 0;
    global_offset[1] = subframe_offset;

    status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                    eopencl->vp9_is_8x8_required,
                                    2, global_offset, global_size, NULL,
                                    0, NULL, NULL);
    assert(status == CL_SUCCESS);
  }

  // launch full pixel search new mv analysis kernel
  // number of workitems per block
  local_size[0] = b_width_in_pixels / workitem_size[0];
  local_size[1] = b_height_in_pixels / workitem_size[1];

  local_size_motion_search[0] = local_size[0];
  local_size_motion_search[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_motion_search[gpu_bsize];

  // total number of workitems
  global_size[0] = num_block_cols * local_size_motion_search[0];
  global_size[1] = num_block_rows * local_size_motion_search[1];

  // if the frame is partitioned in to sub-frames, the global work item
  // size is scaled accordingly. the global offset determines the subframe
  // that is being analysed by the gpu.
  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_motion_search[1];

  if (gpu_bsize == GPU_BLOCK_8X8)
    assert(local_size_motion_search[0] * local_size_motion_search[1] == 1);

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->full_pixel_search[gpu_bsize], 2,
      global_offset, global_size,
      ((gpu_bsize == GPU_BLOCK_8X8) ? NULL : local_size_motion_search),
      0, NULL, event_ptr[0]);
  assert(status == CL_SUCCESS);

  // launch full pixel search kernel zero mv analysis
  // total number of workitems
  global_size[0] = num_block_cols;
  global_size[1] = num_block_rows;

  // if the frame is partitioned in to sub-frames, the global work item
  // size is scaled accordingly. the global offset determines the subframe
  // that is being analysed by the gpu.
  global_offset[0] = 0;
  global_offset[1] = block_row_offset;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->rd_calculation_zeromv[gpu_bsize],
                                  2, global_offset, global_size, NULL,
                                  0, NULL, event_ptr[1]);
  assert(status == CL_SUCCESS);

  global_size[0] = num_block_cols * local_size_motion_search[0];
  global_size[1] = num_block_rows * local_size_motion_search[1];

  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_motion_search[1];

  // launch sub pixel search kernel
  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->sub_pixel_search[gpu_bsize], 2,
      global_offset, global_size,
      ((gpu_bsize == GPU_BLOCK_8X8) ? NULL : local_size_motion_search),
      0, NULL, event_ptr[2]);
  assert(status == CL_SUCCESS);

  // launch inter prediction and sse compute kernel
  local_size_inter_pred[0] = local_size[0];
  local_size_inter_pred[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_inter_pred[gpu_bsize];

  global_size[0] = num_block_cols * local_size_inter_pred[0];
  global_size[1] = num_block_rows * local_size_inter_pred[1];

  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_inter_pred[1];

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->inter_prediction_and_sse[gpu_bsize],
                                  2,
                                  global_offset, global_size,
                                  local_size_inter_pred,
                                  0, NULL, event_ptr[3]);
  assert(status == CL_SUCCESS);

  // launch rd compute kernel
  global_size[0] = num_block_cols;
  global_size[1] = num_block_rows;
  global_offset[0] = 0;
  global_offset[1] = block_row_offset;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->rd_calculation[gpu_bsize], 2,
                                  global_offset, global_size, NULL,
                                  0, NULL, event_ptr[4]);
  assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
  for(i = 0; i < NUM_KERNELS; i++) {
    cl_ulong time_elapsed;
    status = clWaitForEvents(1, event_ptr[i]);
    assert(status == CL_SUCCESS);
    time_elapsed = get_event_time_elapsed(*event_ptr[i]);
    eopencl->total_time_taken[gpu_bsize][i] += time_elapsed / 1000;
    status = clReleaseEvent(*event_ptr[i]);
    assert(status == CL_SUCCESS);
  }
#endif

  status = clFlush(opencl->cmd_queue);
  assert(status == CL_SUCCESS);

  if (gpu_bsize == GPU_BLOCK_8X8) {
    status = clEnqueueMarker(opencl->cmd_queue,
                             &eopencl->event[subframe_idx]);
    assert(status == CL_SUCCESS);
  }
  return;
}

static void vp9_opencl_remove(VP9_COMP *cpi) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
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
    for(i = 0; i < NUM_KERNELS; i++) {
      total[i] += eopencl->total_time_taken[gpu_bsize][i];
      fprintf(stdout, "\tKernel %d - TOTAL = %"PRIu64" microseconds\n", i,
              eopencl->total_time_taken[gpu_bsize][i]);
    }
#endif
    status = clReleaseKernel(eopencl->full_pixel_search[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->rd_calculation_zeromv[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->sub_pixel_search[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->inter_prediction_and_sse[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->rd_calculation[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
  }

#if OPENCL_PROFILING
  fprintf(stdout, "\nTOTAL FOR ALL BLOCK SIZES\n");
  for(i = 0; i < NUM_KERNELS; i++) {
    grand_total += total[i];
    fprintf(stdout,
            "\tKernel %d - TOTAL ALL BLOCK SIZES = %"PRIu64" microseconds\n",
            i, total[i]);
  }
  fprintf(stdout, "\nGRAND TOTAL = %"PRIu64"\n", grand_total);
#endif

  status = clReleaseKernel(eopencl->vp9_is_8x8_required);
  if (status != CL_SUCCESS)
    goto fail;

  return;

fail:
  assert(0);
  return;
}

int vp9_eopencl_init(VP9_COMP *cpi) {
  VP9_COMMON *cm = &cpi->common;
  VP9_GPU *gpu = &cm->gpu;
  VP9_OPENCL *opencl = gpu->compute_framework;
  VP9_EGPU *egpu = &cpi->egpu;
  VP9_EOPENCL *eopencl;
  cl_int status;
  cl_device_id device;
  cl_uint vendor_id;
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
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;

  egpu->compute_framework = vpx_calloc(1, sizeof(VP9_EOPENCL));
  egpu->alloc_buffers = vp9_opencl_alloc_buffers;
  egpu->free_buffers = vp9_opencl_free_buffers;
  egpu->acquire_input_buffer = vp9_opencl_acquire_input_buffer;
  egpu->acquire_output_buffer = vp9_opencl_acquire_output_buffer;
  egpu->acquire_rd_param_buffer = vp9_opencl_acquire_rd_param_buffer;
  egpu->execute = vp9_opencl_execute;
  egpu->enc_sync_read = vp9_opencl_enc_sync_read;
  egpu->remove = vp9_opencl_remove;

  eopencl = egpu->compute_framework;

  eopencl->opencl = opencl;

  // device id
  device = opencl->device;

  // vendor id
  status = clGetDeviceInfo(device, CL_DEVICE_VENDOR_ID,
                           sizeof(cl_uint),
                           &vendor_id,
                           NULL);
  if (status != CL_SUCCESS)
    goto fail;

  // Read kernel source files
  kernel_src = read_src(kernel_file_name);
  if (kernel_src == NULL)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {

    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    if (vendor_id == INTEL_HD_GRAPHICS_ID) {
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
    eopencl->full_pixel_search[gpu_bsize] = clCreateKernel(
        program, "vp9_full_pixel_search", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->sub_pixel_search[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {

    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
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
    eopencl->rd_calculation_zeromv[gpu_bsize] = clCreateKernel(
        program, "vp9_full_pixel_search_zeromv", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->inter_prediction_and_sse[gpu_bsize] = clCreateKernel(
        program, "vp9_inter_prediction_and_sse", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->rd_calculation[gpu_bsize] = clCreateKernel(
        program, "vp9_rd_calculation", &status);
    if (status != CL_SUCCESS)
      goto fail;

    if(gpu_bsize == GPU_BLOCK_8X8) {
      eopencl->vp9_is_8x8_required = clCreateKernel(
          program, "vp9_is_8x8_required", &status);
      if (status != CL_SUCCESS)
        goto fail;
    }

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

fail:
  if(kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}


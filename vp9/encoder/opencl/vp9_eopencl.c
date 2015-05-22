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
                                                         = {2, 2};

static const int pixel_rows_per_workitem_log2_full_pixel[GPU_BLOCK_SIZES]
                                                                = {3, 2};

static const int pixel_rows_per_workitem_log2_sub_pixel[GPU_BLOCK_SIZES]
                                                                = {4, 3};

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

static void vp9_opencl_set_static_kernel_args(VP9_COMP *cpi,
                                              GPU_BLOCK_SIZE gpu_bsize) {
  VP9_COMMON *cm = &cpi->common;
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  cl_mem *gpu_ip = &eopencl->gpu_input[gpu_bsize].opencl_mem;
  cl_mem *gpu_op = &eopencl->gpu_output[gpu_bsize];
  cl_mem *sum_sse = &eopencl->rd_calc_tmp_buffers[gpu_bsize];
  cl_mem *rdopt_parameters = &eopencl->rdopt_parameters.opencl_mem;
  cl_int mi_rows = cm->mi_rows;
  cl_int mi_cols = cm->mi_cols;
  cl_int y_stride = cm->frame_bufs[0].buf.y_stride;
  cl_int status;

  status  = clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 6,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 7,
                           sizeof(cl_int), &mi_cols);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 5,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 6,
                           sizeof(cl_mem), sum_sse);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 7,
                           sizeof(cl_int), &mi_rows);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 8,
                           sizeof(cl_int), &mi_cols);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 5,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize], 0,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize], 1,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize], 2,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 2,
                           sizeof(cl_int), &y_stride);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 4,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 5,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize], 0,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize], 1,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize], 2,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

  status  = clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 2,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 3,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 4,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 0,
                           sizeof(cl_mem), gpu_ip);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 1,
                           sizeof(cl_mem), gpu_op);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 2,
                           sizeof(cl_mem), rdopt_parameters);
  status |= clSetKernelArg(eopencl->rd_calculation[gpu_bsize], 3,
                           sizeof(cl_mem), sum_sse);
  assert(status == CL_SUCCESS);

}

static void vp9_opencl_set_dynamic_kernel_args(VP9_COMP *cpi,
                                               GPU_BLOCK_SIZE gpu_bsize) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  YV12_BUFFER_CONFIG *img_src = cpi->Source;
  YV12_BUFFER_CONFIG *frm_ref = get_ref_frame_buffer(cpi, LAST_FRAME);
  cl_int status;
  SPEED_FEATURES *const sf = &cpi->sf;
  SEARCH_METHODS search_method = sf->mv.search_method;

  status = clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  status |= clSetKernelArg(eopencl->full_pixel_search[gpu_bsize], 8,
                           sizeof(SEARCH_METHODS), &search_method);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->rd_calculation_zeromv[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  assert(status == CL_SUCCESS);

  status = clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 0,
                          sizeof(cl_mem), &frm_ref->gpu_mem);
  status |= clSetKernelArg(eopencl->inter_prediction_and_sse[gpu_bsize], 1,
                           sizeof(cl_mem), &img_src->gpu_mem);
  assert(status == CL_SUCCESS);

}

static void vp9_opencl_map_rd_param_buffer(VP9_COMP *cpi,
                                           GPU_RD_PARAMETERS **host_ptr) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters;

  if (!vp9_opencl_map_buffer(eopencl->opencl, rdopt_parameters, CL_MAP_WRITE,
                             CL_FALSE)) {
    *host_ptr = rdopt_parameters->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_map_input_buffer(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                                        GPU_INPUT **host_ptr) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];

  if (!vp9_opencl_map_buffer(eopencl->opencl, gpu_input, CL_MAP_WRITE,
                             CL_FALSE)) {
    *host_ptr = gpu_input->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_acquire_output_buffer(VP9_COMP *cpi,
                                             GPU_BLOCK_SIZE gpu_bsize,
                                             void **host_ptr,
                                             int sub_frame_idx) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  opencl_buffer *gpu_output_sub_buffer =
      &eopencl->gpu_output_sub_buffer[gpu_bsize][sub_frame_idx];

  if (!vp9_opencl_map_buffer(eopencl->opencl, gpu_output_sub_buffer,
                             CL_MAP_READ, CL_TRUE)) {
    *host_ptr = gpu_output_sub_buffer->mapped_pointer;
    return;
  }

  *host_ptr = NULL;
  assert(0);
}

static void vp9_opencl_alloc_buffers(VP9_COMP *cpi) {
  VP9_COMMON *const cm = &cpi->common;
  VP9_EGPU *gpu = &cpi->egpu;
  VP9_EOPENCL *eopencl = gpu->compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status;
  GPU_BLOCK_SIZE gpu_bsize;
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters;

  // alloc buffer for gpu rd params
  rdopt_parameters->size = sizeof(GPU_RD_PARAMETERS);
  rdopt_parameters->opencl_mem = clCreateBuffer(
      opencl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
      rdopt_parameters->size, NULL, &status);
  if (status != CL_SUCCESS)
    goto fail;

  vp9_opencl_map_rd_param_buffer(cpi, &gpu->gpu_rd_parameters);

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
    const int blocks_in_col = (cm->sb_rows * num_mxn_blocks_high_lookup[bsize]);
    const int blocks_in_row = cpi->blocks_in_row[gpu_bsize];
    const int alloc_size = blocks_in_row * blocks_in_col;
    opencl_buffer *gpuinput_b_args = &eopencl->gpu_input[gpu_bsize];
    int subframe_idx;

    // alloc buffer for gpu input
    gpuinput_b_args->size = alloc_size * sizeof(GPU_INPUT);
    gpuinput_b_args->opencl_mem = clCreateBuffer(
        opencl->context, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
        gpuinput_b_args->size, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    vp9_opencl_map_input_buffer(cpi, gpu_bsize, &gpu->gpu_input[gpu_bsize]);

    // alloc buffer for gpu output
    eopencl->gpu_output[gpu_bsize] = clCreateBuffer(
        opencl->context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR,
        alloc_size * sizeof(GPU_OUTPUT), NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->rd_calc_tmp_buffers[gpu_bsize] = clCreateBuffer(
        opencl->context,  CL_MEM_READ_WRITE,
        alloc_size * sizeof(SUM_SSE_BUFFER), NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;

    // create output sub buffers
    for (subframe_idx = CPU_SUB_FRAMES; subframe_idx < MAX_SUB_FRAMES;
         ++subframe_idx) {
      cl_buffer_region sf_region;
      SubFrameInfo subframe;
      int block_row_offset;
      int block_rows_sf;
      int alloc_size_sf;

      vp9_subframe_init(&subframe, cm, subframe_idx);

      block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);
      block_rows_sf = (mi_cols_aligned_to_sb(subframe.mi_row_end) -
          subframe.mi_row_start) >> mi_height_log2(bsize);

      alloc_size_sf = blocks_in_row * block_rows_sf;

      sf_region.origin = block_row_offset * blocks_in_row * sizeof(GPU_OUTPUT);
      sf_region.size = alloc_size_sf * sizeof(GPU_OUTPUT);
      eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_idx].size =
          sf_region.size;
      eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_idx].opencl_mem =
          clCreateSubBuffer(eopencl->gpu_output[gpu_bsize],
                            CL_MEM_READ_WRITE,
                            CL_BUFFER_CREATE_TYPE_REGION,
                            &sf_region, &status);
      if (status != CL_SUCCESS)
        goto fail;
    }

    vp9_opencl_set_static_kernel_args(cpi, gpu_bsize);
  }

  status = clFinish(opencl->cmd_queue);
  if (status != CL_SUCCESS)
    goto fail;

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
  GPU_BLOCK_SIZE gpu_bsize;
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters;

  if (vp9_opencl_unmap_buffer(opencl, rdopt_parameters, CL_TRUE)) {
    goto fail;
  }
  status = clReleaseMemObject(rdopt_parameters->opencl_mem);
  if (status != CL_SUCCESS)
    goto fail;

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];
    int subframe_id;

    for (subframe_id = CPU_SUB_FRAMES; subframe_id < MAX_SUB_FRAMES;
         ++subframe_id) {
      opencl_buffer *gpu_output_sub_buffer =
          &eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_id];

      if (vp9_opencl_unmap_buffer(opencl, gpu_output_sub_buffer, CL_TRUE)) {
        goto fail;
      }
      status = clReleaseMemObject(
          eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_id].opencl_mem);
      if (status != CL_SUCCESS)
        goto fail;
    }


    if (vp9_opencl_unmap_buffer(opencl, gpu_input, CL_TRUE)) {
      goto fail;
    }
    status = clReleaseMemObject(gpu_input->opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseMemObject(eopencl->gpu_output[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseMemObject(eopencl->rd_calc_tmp_buffers[gpu_bsize]);
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

static void vp9_opencl_enc_sync_read(VP9_COMP *cpi, cl_int event_id) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  cl_int status;

  assert(event_id < MAX_SUB_FRAMES);
  status = clWaitForEvents(1, &eopencl->event[event_id]);
  if (status != CL_SUCCESS)
    vpx_internal_error(&cpi->common.error, VPX_CODEC_ERROR,
                       "Wait for event failed");
}

static void vp9_map_unmap_buffers(VP9_COMP *cpi) {
  VP9_EGPU *gpu = &cpi->egpu;
  VP9_EOPENCL *const eopencl = gpu->compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  GPU_BLOCK_SIZE gpu_bsize;
  int subframe_idx;
  cl_int status = CL_SUCCESS;
  (void)status;

  vp9_opencl_map_rd_param_buffer(cpi, &gpu->gpu_rd_parameters);

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {

    vp9_opencl_map_input_buffer(cpi, gpu_bsize, &gpu->gpu_input[gpu_bsize]);

    for (subframe_idx = CPU_SUB_FRAMES; subframe_idx < MAX_SUB_FRAMES;
         subframe_idx++) {

      opencl_buffer *gpu_output_sub_buffer =
          &eopencl->gpu_output_sub_buffer[gpu_bsize][subframe_idx];
      if (vp9_opencl_unmap_buffer(opencl, gpu_output_sub_buffer, CL_FALSE)) {
        assert(0);
      }
    }
  }
  status = clFlush(opencl->cmd_queue);
  assert(status == CL_SUCCESS);

}

static void vp9_frame_cache_sync(VP9_COMP *cpi, YV12_BUFFER_CONFIG *yuv) {
  VP9_EGPU *gpu = &cpi->egpu;
  VP9_EOPENCL *const eopencl = gpu->compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;
  cl_int status = CL_SUCCESS;

  (void)status;

  status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                   yuv->gpu_mem,
                                   yuv->buffer_alloc,
                                   0, NULL, NULL);
  assert(status == CL_SUCCESS);

  yuv->buffer_alloc = yuv->y_buffer = yuv->u_buffer =
      yuv->v_buffer = NULL;
  vp9_acquire_frame_buffer(&cpi->common, yuv);

  status = clFlush(opencl->cmd_queue);
  assert(status == CL_SUCCESS);
}

static void vp9_opencl_execute(VP9_COMP *cpi, GPU_BLOCK_SIZE gpu_bsize,
                               int subframe_idx) {
  VP9_EOPENCL *const eopencl = cpi->egpu.compute_framework;
  VP9_OPENCL *const opencl = eopencl->opencl;

  VP9_COMMON *const cm = &cpi->common;
  const BLOCK_SIZE bsize = get_actual_block_size(gpu_bsize);
  opencl_buffer *gpu_input = &eopencl->gpu_input[gpu_bsize];
  opencl_buffer *rdopt_parameters = &eopencl->rdopt_parameters;

  const int b_width_in_pixels_log2 = b_width_log2(bsize) + 2;
  const int b_width_in_pixels = 1 << b_width_in_pixels_log2;
  const int b_height_in_pixels_log2 = b_height_log2(bsize) + 2;
  const int b_height_in_pixels = 1 << b_height_in_pixels_log2;
  const int b_height_mask = b_height_in_pixels - 1;

  SubFrameInfo subframe;
  int blocks_in_col, blocks_in_row;
  int block_row_offset;
  int subframe_height;

  const size_t workitem_size[2] = {NUM_PIXELS_PER_WORKITEM, 1};
  size_t local_size[2];
  size_t global_size[2];
  size_t global_offset[2];
  size_t local_size_full_pixel[2], local_size_sub_pixel[2];
  size_t local_size_inter_pred[2];
  const int ms_pixels = (num_8x8_blocks_wide_lookup[bsize] / 2) * 8;

  cl_int status = CL_SUCCESS;

#if OPENCL_PROFILING
  cl_event event[NUM_KERNELS];
#endif
  cl_event *event_ptr[NUM_KERNELS];
  int i;

  for (i = 0; i < NUM_KERNELS; i++) {
#if OPENCL_PROFILING
    event_ptr[i] = &event[i];
#else
    event_ptr[i] = NULL;
#endif
  }

  vp9_subframe_init(&subframe, cm, subframe_idx);
  block_row_offset = subframe.mi_row_start >> mi_height_log2(bsize);

  subframe_height = (subframe.mi_row_end - subframe.mi_row_start) << MI_SIZE_LOG2;
  blocks_in_col = subframe_height >> b_height_in_pixels_log2;
  blocks_in_row = cpi->blocks_in_row[gpu_bsize];

  if (subframe_idx == MAX_SUB_FRAMES - 1)
    if ((cm->height & b_height_mask) > ms_pixels)
      blocks_in_col++;

  if (subframe_idx == 0)
    vp9_opencl_set_dynamic_kernel_args(cpi, gpu_bsize);

  (void)status;

  if (vp9_opencl_unmap_buffer(opencl, rdopt_parameters, CL_FALSE)) {
    assert(0);
  }

  if (vp9_opencl_unmap_buffer(opencl, gpu_input, CL_FALSE)) {
    assert(0);
  }

  // For very small resolutions, this could happen for the last few sub-frames
  if (blocks_in_col == 0)
    goto skip_execution;

  // launch full pixel search new mv analysis kernel
  // number of workitems per block
  local_size[0] = b_width_in_pixels / workitem_size[0];
  local_size[1] = b_height_in_pixels / workitem_size[1];

  local_size_full_pixel[0] = local_size[0];
  local_size_full_pixel[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_full_pixel[gpu_bsize];

  // total number of workitems
  global_size[0] = blocks_in_row * local_size_full_pixel[0];
  global_size[1] = blocks_in_col * local_size_full_pixel[1];

  // if the frame is partitioned in to sub-frames, the global work item
  // size is scaled accordingly. the global offset determines the subframe
  // that is being analysed by the gpu.
  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_full_pixel[1];

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->full_pixel_search[gpu_bsize], 2,
      global_offset, global_size,
      local_size_full_pixel,
      0, NULL, event_ptr[0]);
  assert(status == CL_SUCCESS);

  // launch full pixel search kernel zero mv analysis
  // total number of workitems
  global_size[0] = blocks_in_row;
  global_size[1] = blocks_in_col;

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

  local_size_sub_pixel[0] = local_size[0];
  local_size_sub_pixel[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_sub_pixel[gpu_bsize];

  global_size[0] = blocks_in_row * local_size_sub_pixel[0] * 4;
  global_size[1] = blocks_in_col * local_size_sub_pixel[1];

  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_sub_pixel[1];

  // launch sub pixel search kernel
  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize], 2,
      global_offset, global_size,
      local_size_sub_pixel,
      0, NULL, event_ptr[2]);
  assert(status == CL_SUCCESS);

  global_size[0] = blocks_in_row;
  global_size[1] = blocks_in_col;

  global_offset[0] = 0;
  global_offset[1] = block_row_offset;

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize], 2,
      global_offset, global_size, NULL,
      0, NULL, event_ptr[3]);
  assert(status == CL_SUCCESS);

  global_size[0] = blocks_in_row * local_size_sub_pixel[0] * 4;
  global_size[1] = blocks_in_col * local_size_sub_pixel[1];

  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_sub_pixel[1];

  // launch sub pixel search kernel
  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize], 2,
      global_offset, global_size,
      local_size_sub_pixel,
      0, NULL, event_ptr[4]);
  assert(status == CL_SUCCESS);

  global_size[0] = blocks_in_row;
  global_size[1] = blocks_in_col;

  global_offset[0] = 0;
  global_offset[1] = block_row_offset;

  status = clEnqueueNDRangeKernel(
      opencl->cmd_queue,
      eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize], 2,
      global_offset, global_size,
      NULL,
      0, NULL, event_ptr[5]);
  assert(status == CL_SUCCESS);

  // launch inter prediction and sse compute kernel
  local_size_inter_pred[0] = local_size[0];
  local_size_inter_pred[1] =
      local_size[1] >> pixel_rows_per_workitem_log2_inter_pred[gpu_bsize];

  global_size[0] = blocks_in_row * local_size_inter_pred[0] * 2;
  global_size[1] = blocks_in_col * local_size_inter_pred[1];

  global_offset[0] = 0;
  global_offset[1] = block_row_offset * local_size_inter_pred[1];

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->inter_prediction_and_sse[gpu_bsize],
                                  2,
                                  global_offset, global_size,
                                  local_size_inter_pred,
                                  0, NULL, event_ptr[6]);
  assert(status == CL_SUCCESS);

  // launch rd compute kernel
  global_size[0] = blocks_in_row;
  global_size[1] = blocks_in_col;
  global_offset[0] = 0;
  global_offset[1] = block_row_offset;

  status = clEnqueueNDRangeKernel(opencl->cmd_queue,
                                  eopencl->rd_calculation[gpu_bsize], 2,
                                  global_offset, global_size, NULL,
                                  0, NULL, event_ptr[7]);
  assert(status == CL_SUCCESS);

#if OPENCL_PROFILING
  for (i = 0; i < NUM_KERNELS; i++) {
    cl_ulong time_elapsed;
    status = clWaitForEvents(1, event_ptr[i]);
    assert(status == CL_SUCCESS);
    time_elapsed = get_event_time_elapsed(*event_ptr[i]);
    eopencl->total_time_taken[gpu_bsize][i] += time_elapsed / 1000;
    status = clReleaseEvent(*event_ptr[i]);
    assert(status == CL_SUCCESS);
  }
#endif

skip_execution:
  status = clFlush(opencl->cmd_queue);
  assert(status == CL_SUCCESS);

  if (gpu_bsize == GPU_BLOCK_SIZES - 1) {
    if (eopencl->event[subframe_idx] != NULL) {
      status = clReleaseEvent(eopencl->event[subframe_idx]);
      eopencl->event[subframe_idx] = NULL;
      assert(status == CL_SUCCESS);
    }

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
  int i;
#if OPENCL_PROFILING
  cl_ulong total[NUM_KERNELS] = {0};
  cl_ulong grand_total = 0;
  fprintf(stdout, "\nOPENCL PROFILE RESULTS\n");
#endif

  for (i = 0; i < MAX_SUB_FRAMES; i++) {
    if (eopencl->event[i] != NULL) {
      status = clReleaseEvent(eopencl->event[i]);
      eopencl->event[i] = NULL;
      assert(status == CL_SUCCESS);
    }
  }

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
#if OPENCL_PROFILING
    fprintf(stdout, "\nBlock size idx = %d\n", gpu_bsize);
    for (i = 0; i < NUM_KERNELS; i++) {
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
    status = clReleaseKernel(eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize]);
    if (status != CL_SUCCESS)
      goto fail;
    status = clReleaseKernel(eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize]);
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
  for (i = 0; i < NUM_KERNELS; i++) {
    grand_total += total[i];
    fprintf(stdout,
            "\tKernel %d - TOTAL ALL BLOCK SIZES = %"PRIu64" microseconds\n",
            i, total[i]);
  }
  fprintf(stdout, "\nGRAND TOTAL = %"PRIu64"\n", grand_total);
#endif

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
  char build_options_full_pixel_search[GPU_BLOCK_SIZES][BUILD_OPTION_LENGTH] = {
      "-DBLOCK_SIZE_IN_PIXELS=32 -DPIXEL_ROWS_PER_WORKITEM=8 -DSTRIDE=0 -DINTEL_HD_GRAPHICS=0",
      "-DBLOCK_SIZE_IN_PIXELS=16 -DPIXEL_ROWS_PER_WORKITEM=4 -DSTRIDE=0 -DINTEL_HD_GRAPHICS=0"
  };

  char build_options_sub_pixel_search[GPU_BLOCK_SIZES][BUILD_OPTION_LENGTH] = {
      "-DBLOCK_SIZE_IN_PIXELS=32 -DPIXEL_ROWS_PER_WORKITEM=16 -DSTRIDE=0 -DINTEL_HD_GRAPHICS=0",
      "-DBLOCK_SIZE_IN_PIXELS=16 -DPIXEL_ROWS_PER_WORKITEM=8 -DSTRIDE=0 -DINTEL_HD_GRAPHICS=0"
  };

  char build_options[GPU_BLOCK_SIZES][BUILD_OPTION_LENGTH] = {
      "-DBLOCK_SIZE_IN_PIXELS=32 -DPIXEL_ROWS_PER_WORKITEM=4 -DINTEL_HD_GRAPHICS=",
      "-DBLOCK_SIZE_IN_PIXELS=16 -DPIXEL_ROWS_PER_WORKITEM=4 -DINTEL_HD_GRAPHICS="
  };
  char *kernel_src = NULL;
  GPU_BLOCK_SIZE gpu_bsize;

  const int aligned_width = (cm->width + 7) & ~7;
  const int stride = ((aligned_width + 2 * VP9_ENC_BORDER_IN_PIXELS) + 31) & ~31;


  egpu->compute_framework = vpx_calloc(1, sizeof(VP9_EOPENCL));
  egpu->alloc_buffers = vp9_opencl_alloc_buffers;
  egpu->free_buffers = vp9_opencl_free_buffers;
  egpu->acquire_output_buffer = vp9_opencl_acquire_output_buffer;
  egpu->execute = vp9_opencl_execute;
  egpu->enc_sync_read = vp9_opencl_enc_sync_read;
  egpu->remove = vp9_opencl_remove;
  egpu->prepare_control_buffers = vp9_map_unmap_buffers;
  egpu->frame_cache_sync = vp9_frame_cache_sync;

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
      int string_length = strlen(build_options_full_pixel_search[gpu_bsize]);
      build_options_full_pixel_search[gpu_bsize][string_length - 1] = '1';
    }

    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options_full_pixel_search[gpu_bsize],
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

    if (vendor_id == INTEL_HD_GRAPHICS_ID) {
      int string_length = strlen(build_options_sub_pixel_search[gpu_bsize]);
      build_options_sub_pixel_search[gpu_bsize][string_length - 1] = '1';
    }

    // Build the program
    status = clBuildProgram(program, 1, &device,
                            build_options_sub_pixel_search[gpu_bsize],
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

    eopencl->sub_pixel_search_halfpel_filtering[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_halfpel_filtering", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->sub_pixel_search_halfpel_bestmv[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_halfpel_bestmv", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->sub_pixel_search_quarterpel_filtering[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_quarterpel_filtering", &status);
    if (status != CL_SUCCESS)
      goto fail;

    eopencl->sub_pixel_search_quarterpel_bestmv[gpu_bsize] = clCreateKernel(
        program, "vp9_sub_pixel_search_quarterpel_bestmv", &status);
    if (status != CL_SUCCESS)
      goto fail;

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  for (gpu_bsize = 0; gpu_bsize < GPU_BLOCK_SIZES; gpu_bsize++) {
    int string_length = strlen(build_options[gpu_bsize]);

    program = clCreateProgramWithSource(opencl->context, 1,
                                        (const char**)(void *)&kernel_src,
                                        NULL,
                                        &status);
    if (status != CL_SUCCESS)
      goto fail;

    // Passing stride as a compile time option. In this way compiler generates
    // better code for "vp9_inter_prediction_and_sse" kernel.
    if (vendor_id == INTEL_HD_GRAPHICS_ID) {
      sprintf(build_options[gpu_bsize] + string_length, "1 -DSTRIDE=%d", stride);
    } else {
      sprintf(build_options[gpu_bsize] + string_length, "0 -DSTRIDE=%d", stride);
    }

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

    status = clReleaseProgram(program);
    if (status != CL_SUCCESS)
      goto fail;
  }

  vpx_free(kernel_src);
  return 0;

fail:
  if (kernel_src != NULL)
    vpx_free(kernel_src);
  return 1;
}


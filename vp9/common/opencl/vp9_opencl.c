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

static void *vp9_opencl_alloc_frame_buffers(VP9_COMMON *cm, int frame_size,
                                            void **opencl_mem) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  void *mapped_pointer;
  cl_int status;

  *opencl_mem = clCreateBuffer(opencl->context,
                               CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR,
                               frame_size, NULL, &status);
  if (status != CL_SUCCESS)
    return NULL;
  mapped_pointer = clEnqueueMapBuffer(opencl->cmd_queue, *opencl_mem,
                                      CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                      0, frame_size, 0, NULL, NULL,
                                      &status);
  if (status != CL_SUCCESS) {
    clReleaseMemObject(*opencl_mem);
    *opencl_mem = NULL;
    return NULL;
  }

  return mapped_pointer;
}

static void vp9_opencl_release_frame_buffers(VP9_COMMON *cm, void **opencl_mem,
                                             void **mapped_pointer) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  cl_event event;
  cl_int status;

  if (*mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     *opencl_mem,
                                     *mapped_pointer,
                                     0, NULL, &event);
    status |= clWaitForEvents(1, &event);
    if (status != CL_SUCCESS)
      goto fail;
    *mapped_pointer = NULL;

    status = clReleaseEvent(event);
    if (status != CL_SUCCESS)
      goto fail;
  }

  if (*opencl_mem != NULL) {
    status = clReleaseMemObject(*opencl_mem);
    if (status != CL_SUCCESS)
      goto fail;
    *opencl_mem = NULL;
  }
  return;

fail:
  assert(0);
}

static void vp9_opencl_acquire_frame_buffers(VP9_COMMON *cm, void **opencl_mem,
                                             void **mapped_pointer, int size) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  cl_int status;

  if (*mapped_pointer == NULL) {
    *mapped_pointer =
        clEnqueueMapBuffer(opencl->cmd_queue_memory, *opencl_mem, CL_TRUE,
                           CL_MAP_READ | CL_MAP_WRITE, 0, size, 0,
                           NULL, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;
  }

  return;

fail:
  assert(0);
}

// NOTE : Mapping happens in a different command queue than NDRangeKernel
int vp9_opencl_map_buffer(VP9_OPENCL *const opencl,
                          opencl_buffer *opencl_buf,
                          cl_map_flags map_flags) {
  cl_int status;

  if (opencl_buf->mapped_pointer == NULL) {
    opencl_buf->mapped_pointer = clEnqueueMapBuffer(opencl->cmd_queue_memory,
                                                    opencl_buf->opencl_mem,
                                                    CL_TRUE,
                                                    map_flags,
                                                    0,
                                                    opencl_buf->size,
                                                    0, NULL, NULL, &status);
    if (status != CL_SUCCESS)
      goto fail;
  }
  return 0;

fail:
  return 1;
}

// NOTE : Unmapping happens in the same command queue as NDRangeKernel
int vp9_opencl_unmap_buffer(VP9_OPENCL *const opencl,
                            opencl_buffer *opencl_buf,
                            cl_bool is_blocking) {
  cl_int status;


  if (opencl_buf->mapped_pointer != NULL) {
    status = clEnqueueUnmapMemObject(opencl->cmd_queue,
                                     opencl_buf->opencl_mem,
                                     opencl_buf->mapped_pointer,
                                     0, NULL, NULL);
    opencl_buf->mapped_pointer = NULL;
    if (status != CL_SUCCESS)
      goto fail;

    if (is_blocking == CL_TRUE) {
      status = clFinish(opencl->cmd_queue);
      if (status != CL_SUCCESS)
        goto fail;
    }
  }
  return 0;

fail:
  return 1;
}

static void vp9_opencl_remove(VP9_COMMON *cm) {
  VP9_OPENCL *opencl = cm->gpu.compute_framework;
  cl_int status;

  status = clReleaseCommandQueue(opencl->cmd_queue);
  if (status != CL_SUCCESS)
    goto fail;
  status = clReleaseCommandQueue(opencl->cmd_queue_memory);
  if (status != CL_SUCCESS)
    goto fail;

  status = clReleaseContext(opencl->context);
  if (status != CL_SUCCESS)
    goto fail;

  return;

fail:
  assert(0);
}

int vp9_opencl_init(VP9_COMMON *cm) {
  VP9_GPU *gpu = &cm->gpu;
  VP9_OPENCL *opencl;
  cl_int status;
  cl_uint num_platforms = 0;
  cl_platform_id platform;
  cl_uint num_devices = 0;
  cl_device_id device;
  cl_command_queue_properties command_queue_properties = 0;

#if OPENCL_PROFILING
  command_queue_properties = CL_QUEUE_PROFILING_ENABLE;
#endif

  gpu->compute_framework = vpx_calloc(1, sizeof(VP9_OPENCL));

  gpu->alloc_frame_buffers = vp9_opencl_alloc_frame_buffers;
  gpu->release_frame_buffers = vp9_opencl_release_frame_buffers;
  gpu->acquire_frame_buffers = vp9_opencl_acquire_frame_buffers;
  gpu->remove = vp9_opencl_remove;

  opencl = gpu->compute_framework;

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
  opencl->device = device;

 // Create OpenCL context for one device
  opencl->context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  if (status != CL_SUCCESS || opencl->context == NULL)
    goto fail;

  // Create command queues for the device
  opencl->cmd_queue_memory = clCreateCommandQueue(opencl->context, device,
                                                  command_queue_properties,
                                                  &status);
  if (status != CL_SUCCESS || opencl->cmd_queue_memory == NULL)
    goto fail;

  opencl->cmd_queue = clCreateCommandQueue(opencl->context, device,
                                           command_queue_properties,
                                           &status);
  if (status != CL_SUCCESS || opencl->cmd_queue == NULL)
    goto fail;
  return 0;

fail:
  return 1;
}

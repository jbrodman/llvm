//==---------------- usm.hpp - SYCL USM ------------------------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#include <cstddef>

#include <CL/sycl/detail/clusm.hpp>

#pragma once

namespace cl {
namespace sycl {

#ifdef INTEL_USM

///
// Explicit USM
///
void *sycl_malloc_device(size_t size) {
  auto selector = default_selector();
  auto dev = selector.select_device();
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clDeviceMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size, 0,
                               nullptr);
  // Need code to check errors and throw exception
}

void *sycl_malloc_device(size_t size, const device &dev) {
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clDeviceMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size, 0,
                               nullptr);
}

void *sycl_aligned_alloc_device(size_t alignment, size_t size) {
  auto selector = default_selector();
  auto dev = selector.select_device();
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clDeviceMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size,
                               alignment, nullptr);
}

void *sycl_aligned_alloc_device(size_t alignment, size_t size,
                                const device &dev) {
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clDeviceMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size,
                               alignment, nullptr);
}

void sycl_free(void *ptr) {
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  clMemFreeINTEL(c, ptr);
}

event handler::sycl_memcpy(void *dest, const void *src, size_t count) {
  cl_event e;
  cl_command_queue q = MQueue->get();

  clEnqueueMemcpyINTEL(q,
                       /* blocking */ false, dest, src, count, 0, nullptr, &e);

  clReleaseCommandQueue(q);
  return event(e, MQueue->get_context());
}

event queue::sycl_memcpy(void *dest, const void *src, size_t count) {
  cl_event e;
  cl_command_queue q = get();

  clEnqueueMemcpyINTEL(q,
                       /* blocking */ false, dest, src, count, 0, nullptr, &e);

  // get() retains
  // This is kind of gross.
  // Inc/Dec ref count for 1 call is overkill.
  clReleaseCommandQueue(q);

  return event(e, get_context());
}

event handler::sycl_memset(void *ptr, int value, size_t count) {
  cl_event e;
  cl_command_queue q = MQueue->get();

  clEnqueueMemsetINTEL(q, ptr, value, count, 0, nullptr, &e);

  clReleaseCommandQueue(q);
  return event(e, MQueue->get_context());
}

event queue::sycl_memset(void *ptr, int value, size_t count) {
  cl_event e;
  cl_command_queue q = get();

  clEnqueueMemsetINTEL(q, ptr, value, count, 0, nullptr, &e);

  clReleaseCommandQueue(q);
  return event(e, get_context());
}

///
// Restricted USM
///
void *sycl_malloc_host(size_t size) {
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clHostMemAllocINTEL(c, CL_MEM_ALLOC_DEFAULT_INTEL, size, 0, nullptr);
  // check errors?
}

void *sycl_malloc(size_t size) {
  auto selector = default_selector();
  auto dev = selector.select_device();
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clSharedMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size, 0,
                               nullptr);
}

void *sycl_malloc(size_t size, const device &dev) {
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clSharedMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size, 0,
                               nullptr);
}

void *sycl_aligned_alloc_host(size_t alignment, size_t size) {
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clHostMemAllocINTEL(c, CL_MEM_ALLOC_DEFAULT_INTEL, size, alignment,
                             nullptr);
}

void *sycl_aligned_alloc(size_t alignment, size_t size) {
  auto selector = default_selector();
  auto dev = selector.select_device();
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clSharedMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size,
                               alignment, nullptr);
}

void *sycl_aligned_alloc(size_t alignment, size_t size, const device &dev) {
  cl_device_id id = dev.get();
  cl_int error;
  cl_context c = clGetDefaultContextINTEL(&error);

  return clSharedMemAllocINTEL(c, id, CL_MEM_ALLOC_DEFAULT_INTEL, size,
                               alignment, nullptr);
}

#endif // INTEL_USM
} // namespace sycl
} // namespace cl

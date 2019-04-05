//==---------------- oclstubs.cpp - OpenCL extension for USM ---*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //


#include <CL/sycl/detail/clusm.hpp>
#include <string>

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_context CL_API_CALL
clGetDefaultContextINTEL(cl_int *errcode_ret) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_context retVal = clusm->getDefaultContext();

    if (errcode_ret) {
      errcode_ret[0] = CL_SUCCESS;
    }

    return retVal;
  }

  return NULL;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clSetDefaultContextINTEL(cl_context context) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = CL_SUCCESS;
    clusm->setDefaultContext(context);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY void *CL_API_CALL
clHostMemAllocINTEL(cl_context context, cl_mem_alloc_flags_intel flags,
                    size_t size, cl_uint alignment, cl_int *errcode_ret) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    void *retVal =
        clusm->hostMemAlloc(context, flags, size, alignment, errcode_ret);

    return retVal;
  }

  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY void *CL_API_CALL
clDeviceMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_alloc_flags_intel flags, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    void *retVal = clusm->deviceMemAlloc(context, device, flags, size,
                                         alignment, errcode_ret);

    return retVal;
  }

  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY void *CL_API_CALL
clSharedMemAllocINTEL(cl_context context, cl_device_id device,
                      cl_mem_alloc_flags_intel flags, // TBD: needed?
                      size_t size, cl_uint alignment, cl_int *errcode_ret) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    void *retVal = clusm->sharedMemAlloc(context, device, flags, size,
                                         alignment, errcode_ret);

    return retVal;
  }

  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clMemFreeINTEL(cl_context context,
                                               const void *ptr) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = clusm->memFree(context, ptr);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clGetMemAllocInfoINTEL(
    cl_context context, const void *ptr, cl_mem_info_intel param_name,
    size_t param_value_size, void *param_value, size_t *param_value_size_ret) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal =
        clusm->getMemAllocInfoINTEL(context, ptr, param_name, param_value_size,
                                    param_value, param_value_size_ret);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clSetKernelArgMemPointerINTEL(
    cl_kernel kernel, cl_uint arg_index, const void *arg_value) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = clSetKernelArgSVMPointer(kernel, arg_index, arg_value);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL
clEnqueueMemsetINTEL(cl_command_queue queue, void *dst_ptr, cl_int value,
                     size_t count, cl_uint num_events_in_wait_list,
                     const cl_event *event_wait_list, cl_event *event) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = CL_SUCCESS;
    const cl_uchar pattern = (cl_uchar)value;

    retVal =
        clEnqueueSVMMemFill(queue, dst_ptr, &pattern, sizeof(pattern), count,
                            num_events_in_wait_list, event_wait_list, event);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemcpyINTEL(
    cl_command_queue queue, cl_bool blocking, void *dst_ptr,
    const void *src_ptr, size_t size, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = CL_SUCCESS;

    retVal =
        clEnqueueSVMMemcpy(queue, blocking, dst_ptr, src_ptr, size,
                           num_events_in_wait_list, event_wait_list, event);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMigrateMemINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_migration_flags flags, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = CL_SUCCESS;

    // We could check for OpenCL 2.1 and call the SVM migrate
    // functions, but for now we'll just enqueue a marker.
#if 0
    retVal = clEnqueueSVMMigrateMem(
      queue,
      1,
      &ptr,
      &size,
      flags,
      num_events_in_wait_list,
      event_wait_list,
      event );
#else
    retVal = clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                         event_wait_list, event);
#endif

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

///////////////////////////////////////////////////////////////////////////////
//
CL_API_ENTRY cl_int CL_API_CALL clEnqueueMemAdviseINTEL(
    cl_command_queue queue, const void *ptr, size_t size,
    cl_mem_advice_intel advice, cl_uint num_events_in_wait_list,
    const cl_event *event_wait_list, cl_event *event) {
  CLUSM *clusm = GetCLUSM();

  if (clusm) {
    cl_int retVal = CL_SUCCESS;

    // TODO: What should we do here?
    retVal = clEnqueueMarkerWithWaitList(queue, num_events_in_wait_list,
                                         event_wait_list, event);

    return retVal;
  }

  return CL_INVALID_OPERATION;
}

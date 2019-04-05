//==---------------- clusm.hp - SYCL USM for CL Utils ----------*- C++ -*---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //

#pragma once

#include <map>
#include <vector>

#include <CL/cl.h>
#include <CL/cl_usm_ext.h>

class CLUSM {
public:
  static bool Create(CLUSM *&pCLUSM);
  static void Delete(CLUSM *&pCLUSM);

  void setInitialDefaultContext(cl_context context);
  void setDefaultContext(cl_context context);
  inline cl_context getDefaultContext(void) const { return mDefaultContext; }

  void *hostMemAlloc(cl_context context, cl_mem_alloc_flags_intel flags,
                     size_t size, cl_uint alignment, cl_int *errcode_ret);
  void *deviceMemAlloc(cl_context context, cl_device_id device,
                       cl_mem_alloc_flags_intel flags, size_t size,
                       cl_uint alignment, cl_int *errcode_ret);
  void *sharedMemAlloc(cl_context context, cl_device_id device,
                       cl_mem_alloc_flags_intel flags, size_t size,
                       cl_uint alignment, cl_int *errcode_ret);

  cl_int memFree(cl_context context, const void *ptr);

  cl_int getMemAllocInfoINTEL(cl_context context, const void *ptr,
                              cl_mem_info_intel param_name,
                              size_t param_value_size, void *param_value,
                              size_t *param_value_size_ret);

  cl_int setKernelExecInfo(cl_kernel kernel, cl_kernel_exec_info param_name,
                           size_t param_value_size, const void *param_value);

  cl_int setKernelIndirectUSMExecInfo(cl_command_queue queue, cl_kernel kernel);

  void EnterCriticalSection();
  void LeaveCriticalSection();

  template <class T>
  cl_int writeParamToMemory(size_t param_value_size, T param,
                            size_t *param_value_size_ret, T *pointer) const;

private:
  bool mDefaultContextInitialized;
  cl_context mDefaultContext;

#if defined(__linux__)
  pthread_mutex_t mCriticalSection;
#elif defined(_WIN32)
  CRITIAL_SECTION mCriticalSection;
#endif

  CLUSM();
  ~CLUSM();
  bool init();

  struct SUSMAllocInfo {
    SUSMAllocInfo()
        : Type(CL_MEM_TYPE_UNKNOWN_INTEL), BaseAddress(NULL), Size(0),
          Alignment(0) {}

    cl_unified_shared_memory_type_intel Type;

    const void *BaseAddress;
    size_t Size;
    size_t Alignment;
  };

  typedef std::map<const void *, SUSMAllocInfo> CUSMAllocMap;
  typedef std::vector<const void *> CUSMAllocVector;

  struct SUSMContextInfo {
    CUSMAllocMap AllocMap;

    CUSMAllocVector HostAllocVector;
    // TODO: Support multiple devices by mapping device-> vector?
    CUSMAllocVector DeviceAllocVector;
    CUSMAllocVector SharedAllocVector;
  };

  // TODO: Support multiple contexts by mapping context -> USMContextInfo?
  SUSMContextInfo mUSMContextInfo;

  struct SUSMKernelInfo {
    SUSMKernelInfo()
        : IndirectHostAccess(false), IndirectDeviceAccess(false),
          IndirectSharedAccess(false) {}

    bool IndirectHostAccess;
    bool IndirectDeviceAccess;
    bool IndirectSharedAccess;

    std::vector<void *> SVMPtrs;
  };

  typedef std::map<cl_kernel, SUSMKernelInfo> CUSMKernelInfoMap;

  CUSMKernelInfoMap mUSMKernelInfoMap;
};

extern CLUSM *gCLUSM;
inline CLUSM *GetCLUSM() {
  if (gCLUSM == nullptr) {
    CLUSM::Create(gCLUSM);
  }
  return gCLUSM;
}

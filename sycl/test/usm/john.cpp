// XFAIL: cuda
// piextUSM*Alloc functions for CUDA are not behaving as described in
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/USM.adoc
// https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/USM/cl_intel_unified_shared_memory.asciidoc
//
// RUN: %clangxx -fsycl -fsycl-unnamed-lambda -fsycl-targets=%sycl_triple %s -o %t1.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t1.out
// RUN: %CPU_RUN_PLACEHOLDER %t1.out
// RUN: %GPU_RUN_PLACEHOLDER %t1.out

//==------------------- mixed.cpp - Mixed Memory test ---------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  const int MAGIC_NUM = 42;
  const int N = 17;
  
  queue q;
  auto dev = q.get_device();
  auto ctxt = q.get_context();

  auto A = malloc_shared<int>(N, q);
  auto B = malloc_shared<int>(N, q);
  auto C = malloc_shared<int>(std::max(N,64), q);

  for (int i = 0; i < N; i++) {
    A[i] = 1;
    B[i] = 2;
  }

  q.parallel_for(range<1>(N), [=](id<1> i) { C[i] = A[i] + B[i]; });
  q.wait();

  for (int i = 0; i < N; i++) {
    assert(C[i] == 3);
  }

  for (int i = 0; i < 64; i++) {
    std::cout << C[i] << " ";
  }
  std::cout << std::endl;
  
  free(A, q);
  free(B, q);
  free(C, q);

  return 0;
}

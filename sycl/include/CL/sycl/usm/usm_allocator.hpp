//==------ usm_allocator.hpp - SYCL USM Allocator ------*- C++ -*-----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===--------------------------------------------------------------------=== //
#pragma once

#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/usm/usm_enums.hpp>

#include <cstdlib>
#include <memory>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {

// Forward declarations.
void *aligned_alloc(size_t alignment, size_t size, const device &dev,
                    const context &ctxt, usm::alloc kind);
void free(void *ptr, const context &ctxt);

template <class T, usm::alloc AllocKind, size_t Alignment = 0>
class usm_allocator {
public:
  using value_type = T;

  template <typename U> struct rebind {
    typedef usm_allocator<U, AllocKind, Alignment> other;
  };

  usm_allocator() noexcept = delete;
  usm_allocator(const context &Ctxt, const device &Dev) noexcept
      : MContext(Ctxt), MDevice(Dev) {}
  usm_allocator(const queue &Q) noexcept
      : MContext(Q.get_context()), MDevice(Q.get_device()) {}
  usm_allocator(const usm_allocator &Other) noexcept
      : MContext(Other.MContext), MDevice(Other.MDevice) {}

  template <class U> usm_allocator(usm_allocator<U, AllocKind, Alignment> const &) noexcept {}

  /// Allocates memory.
  ///
  /// \param NumberOfElements is a count of elements to allocate memory for.
  T *allocate(size_t NumberOfElements) {

    auto Result = reinterpret_cast<T *>(
        aligned_alloc(getAlignment(), NumberOfElements * sizeof(value_type),
                      MDevice, MContext, AllocKind));
    if (!Result) {
      throw memory_allocation_error();
    }
    return Result;
  }

  /// Deallocates memory.
  ///
  /// \param Ptr is a pointer to memory being deallocated.
  /// \param Size is a number of elements previously passed to allocate.
  void deallocate(T *Ptr, size_t Size) {
    if (Ptr) {
      free(Ptr, MContext);
    }
  }

  /// Constructs an object on memory pointed by Ptr.
  ///
  /// Note: AllocKind == alloc::device is not allowed.
  ///
  /// \param Ptr is a pointer to memory that will be used to construct the
  /// object.
  /// \param Val is a value to initialize the newly constructed object.
  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0,
      class U, class... ArgTs>
  void construct(U *Ptr, ArgTs &&... Args) {
    ::new (Ptr) U(std::forward<ArgTs>(Args)...);
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0,
      class U, class... ArgTs>
  void construct(U *Ptr, ArgTs &&... Args) {
    throw feature_not_supported(
      "Device pointers do not support construct on host",
      PI_INVALID_OPERATION);
  }

  /// Destroys an object.
  ///
  /// Note:: AllocKind == alloc::device is not allowed
  ///
  /// \param Ptr is a pointer to memory where the object resides.
  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT != usm::alloc::device, int>::type = 0>
  void destroy(T *Ptr) {
    Ptr->~value_type();
  }

  template <
      usm::alloc AllocT = AllocKind,
      typename std::enable_if<AllocT == usm::alloc::device, int>::type = 0>
  void destroy(T *Ptr) {
    throw feature_not_supported(
      "Device pointers do not support destroy on host", PI_INVALID_OPERATION);
  }

private:
  constexpr size_t getAlignment() const {
    /*
      // This form might be preferable if the underlying implementation
      // doesn't do the right thing when given 0 for alignment
    return ((Alignment == 0)
            ? alignof(value_type)
            : Alignment);
    */
    return Alignment;
  }

  const context MContext;
  const device MDevice;
};

/// Equality Comparison
///
/// Allocators only compare equal if they are of the same USM kind and alignment
template <class T, usm::alloc AllocKindT, size_t AlignmentT, class U,
          usm::alloc AllocKindU, size_t AlignmentU>
bool operator==(const usm_allocator<T, AllocKindT, AlignmentT> &,
                const usm_allocator<U, AllocKindU, AlignmentU> &) noexcept {
  return (AllocKindT == AllocKindU) && (AlignmentT == AlignmentU);
}

/// Inequality Comparison
///
/// Allocators only compare unequal if they are not of the same USM kind and alignment
template <class T, class U, usm::alloc AllocKind, size_t Alignment = 0>
bool operator!=(const usm_allocator<T, AllocKind, Alignment> &allocT,
                const usm_allocator<U, AllocKind, Alignment> &allocU) noexcept {
  return !(allocT == allocU);
}

} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)

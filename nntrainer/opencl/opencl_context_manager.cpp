// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file    opencl_context_manager.cpp
 * @date    06 Feb 2024
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Debadri Samaddar <s.debadri@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   OpenCL wrapper for context management
 *
 */

#include "opencl_context_manager.h"

#include <vector>

#include "opencl_loader.h"
#include "third_party/cl.h"

#include <nntrainer_log.h>

namespace nntrainer::opencl {

/**
 * @brief Get the global instance object
 *
 * @return ContextManager global instance
 */
ContextManager &ContextManager::GetInstance() {
  static ContextManager instance;
  return instance;
}

/**
 * @brief Get the OpenCL context object
 *
 * @return const cl_context
 */
const cl_context &ContextManager::GetContext() {
  // loading the OpenCL library and required functions
  LoadOpenCL();

  if (context_) {
    // increments the context reference count
    clRetainContext(context_);
    return context_;
  }

  bool result = true;

  do {
    result = CreateDefaultOpenCLHandles();
    if (!result) {
      break;
    }

    // increments the context reference count
    clRetainContext(context_);

  } while (false);

  if (!result) {
    ml_loge("Failed to create OpenCL Context");
    context_ = nullptr;
  }

  return context_;
}

/**
 * @brief Release OpenCL context
 *
 */
void ContextManager::ReleaseContext() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
  }
}

/**
 * @brief Get the Device Id object
 *
 * @return const cl_device_id
 */
const cl_device_id ContextManager::GetDeviceId() { return device_id_; }

/**
 * @brief Destroy the Context Manager object
 *
 */
ContextManager::~ContextManager() {
  if (context_) {
    // decrements the context reference count
    clReleaseContext(context_);
    context_ = nullptr;
  }
}

/**
 * @brief Checks whether selected OpenCL device supports requested extension.
 *
 * @param extension requested extension
 *
 * @return true if device supports extension
 */
bool ContextManager::CheckDeviceExtensionSupport(const char *extension) {
  cl_int status = CL_SUCCESS;
  size_t extension_size = 0;

  status =
    clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, 0, NULL, &extension_size);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceInfo returned %d", status);
    return false;
  }

  std::vector<char> extensions(extension_size);
  status = clGetDeviceInfo(device_id_, CL_DEVICE_EXTENSIONS, extension_size,
                           extensions.data(), NULL);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceInfo returned %d", status);
    return false;
  }

  if (std::string(extensions.data()).find(extension) == std::string::npos) {
    ml_loge("Extension %s is not supported by given device", extension);
    return false;
  }

  return true;
}

/**
 * @brief Create OpenCL platform
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateOpenCLPlatform() {
  const size_t default_platform_index = 0;

  cl_int status = CL_SUCCESS;
  cl_uint num_platforms = 0;

  // returns number of OpenCL supported platforms
  status = clGetPlatformIDs(0, nullptr, &num_platforms);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }
  if (num_platforms == 0) {
    ml_loge("No supported OpenCL platform.");
    return false;
  }

  // getting the platform IDs
  std::vector<cl_platform_id> platforms(num_platforms);
  status = clGetPlatformIDs(num_platforms, platforms.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetPlatformIDs returned %d", status);
    return false;
  }

  // platform is a specific OpenCL implementation, for instance ARM
  platform_id_ = platforms[default_platform_index];

  return true;
}

/**
 * @brief Create OpenCL device
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateOpenCLDevice() {
  const size_t default_device_index = 0;

  cl_int status = CL_SUCCESS;
  cl_uint num_devices = 0;

  // getting available GPU devices
  status =
    clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, 0, nullptr, &num_devices);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceIDs returned %d", status);
    return false;
  }
  if (num_devices == 0) {
    ml_loge("No GPU on current platform.");
    return false;
  }

  // getting the GPU device IDs
  std::vector<cl_device_id> devices(num_devices);
  status = clGetDeviceIDs(platform_id_, CL_DEVICE_TYPE_GPU, num_devices,
                          devices.data(), nullptr);
  if (status != CL_SUCCESS) {
    ml_loge("clGetDeviceIDs returned %d", status);
    return false;
  }

  // setting the first GPU ID and platform (ARM)
  device_id_ = devices[default_device_index];

  return true;
}

/**
 * @brief Create OpenCL context
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateOpenCLContext() {
  cl_int error_code = CL_SUCCESS;
  cl_context_properties properties[] = {CL_CONTEXT_PLATFORM,
                                        (cl_context_properties)platform_id_, 0};

  // creating valid ARM GPU OpenCL context, will return NULL with error code if
  // fails
  context_ =
    clCreateContext(properties, 1, &device_id_, nullptr, nullptr, &error_code);
  if (!context_) {
    ml_loge("Failed to create a compute context. OpenCL error code: %d",
            error_code);
    return false;
  }

  return true;
}

/**
 * @brief Create default OpenCL handles (platform, device and context)
 *
 * @return true if successful or false otherwise
 */
bool ContextManager::CreateDefaultOpenCLHandles() {
  if (!CreateOpenCLPlatform()) {
    return false;
  }

  if (!CreateOpenCLDevice()) {
    return false;
  }

#ifdef ENABLE_FP16
  // check for fp16 (half) support available on device
  // getting extensions
  if (!CheckDeviceExtensionSupport("cl_khr_fp16")) {
    return false;
  }
#endif

  if (!CreateOpenCLContext()) {
    return false;
  }

  return true;
}
} // namespace nntrainer::opencl

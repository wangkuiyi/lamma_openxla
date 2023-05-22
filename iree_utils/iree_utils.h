// Copyright 2023 Apple Inc.
#pragma once

#include <stdint.h>

#include "iree/runtime/api.h"

iree_status_t iree_utils_runtime_instance_create(
    iree_runtime_instance_t** instance);

int iree_utils_status_print_and_free(iree_status_t statue);

iree_status_t iree_utils_runtime_session_create(
    iree_runtime_instance_t* instance, const char* device_name,
    iree_runtime_session_t** session);

iree_status_t iree_utils_load_module(iree_runtime_session_t* session,
                                     const char* module_path);

iree_host_size_t num_elements(const iree_hal_dim_t* shape,
                              iree_host_size_t shape_rank);

iree_status_t iree_utils_f32_tensor_create(iree_runtime_session_t* session,
                                           const iree_hal_dim_t* shape,
                                           iree_host_size_t shape_rank,
                                           const float* data,
                                           iree_hal_buffer_view_t** tensor);

iree_status_t iree_utils_i32_tensor_create(iree_runtime_session_t* session,
                                           const iree_hal_dim_t* shape,
                                           iree_host_size_t shape_rank,
                                           const int32_t* data,
                                           iree_hal_buffer_view_t** tensor);

void iree_utils_tensor_print(iree_runtime_session_t* session,
                             iree_hal_buffer_view_t* tensor);

iree_status_t iree_utils_make_call(iree_runtime_session_t* session,
                                   const char* function_name,
                                   iree_runtime_call_t* call, int num_args,
                                   ...);

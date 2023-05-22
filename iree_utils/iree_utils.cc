// Copyright 2023 Apple Inc.

#include "iree_utils/iree_utils.h"

#include <stdarg.h>

/*
   An IREE runtime instance is like a VM.
   iree_utils_runtime_instance_create calls
   iree_runtime_instance_create with the default options. The caller
   needs to free up the instance by calling
   iree_runtimee_instance_release after using it.
 */
iree_status_t iree_utils_runtime_instance_create(
    iree_runtime_instance_t** instance) {
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  return iree_runtime_instance_create(&instance_options,
                                      iree_allocator_system(), instance);
}

/*
   Note that a status is a handle and must be freeed.  Before freeing
   it, this function prints it.
 */
int iree_utils_status_print_and_free(iree_status_t status) {
  int ret = iree_status_code(status);
  if (!iree_status_is_ok(status)) {
    iree_status_fprint(stderr, status);
    iree_status_ignore(status);
  }
  return ret;
}

/*
   A session is like a process that executes a module.
   iree_utils_runtime_session_create creates a session on the default
   device.  The caller needs to free up the session by calling
   iree_runtime_session_release.
 */
iree_status_t iree_utils_runtime_session_create(
    iree_runtime_instance_t* instance, const char* device_name,
    iree_runtime_session_t** session) {
  if (device_name == NULL) device_name = "metal";

  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view(device_name), &device));

  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  *session = NULL;
  iree_status_t status = iree_runtime_session_create_with_device(
      instance, &session_options, device,
      iree_runtime_instance_host_allocator(instance), session);
  iree_hal_device_release(device);
  return status;
}

/*
   After the creation of a session and before invoking any function,
   we need to load a module.  This is like the POSIX system call fork
   mapping a program into the process space.
 */
iree_status_t iree_utils_load_module(iree_runtime_session_t* session,
                                     const char* module_path) {
  return iree_runtime_session_append_bytecode_module_from_file(session,
                                                               module_path);
}

iree_host_size_t num_elements(const iree_hal_dim_t* shape,
                              iree_host_size_t shape_rank) {
  iree_host_size_t n = 1;
  for (iree_host_size_t i = 0; i < shape_rank; ++i) n *= shape[i];
  return n;
}

/*
   Create an on-device tensor from data on the host (CPU).
 */
iree_status_t iree_utils_tensor_create(iree_runtime_session_t* session,
                                       const iree_hal_dim_t* shape,
                                       iree_host_size_t shape_rank,
                                       iree_hal_element_type_t element_type,
                                       const void* data,
                                       iree_hal_buffer_view_t** tensor) {
  return iree_hal_buffer_view_allocate_buffer(
      iree_runtime_session_device_allocator(session), shape_rank, shape,
      element_type, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
          // Access to allow to this memory (this is .rodata so READ only):
          .access = IREE_HAL_MEMORY_ACCESS_READ,
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      },
      iree_make_const_byte_span(
          data, iree_hal_element_dense_byte_count(element_type) *
                    num_elements(shape, shape_rank)),
      tensor);
}

iree_status_t iree_utils_f32_tensor_create(iree_runtime_session_t* session,
                                           const iree_hal_dim_t* shape,
                                           iree_host_size_t shape_rank,
                                           const float* data,
                                           iree_hal_buffer_view_t** tensor) {
  return iree_utils_tensor_create(session, shape, shape_rank,
                                  IREE_HAL_ELEMENT_TYPE_FLOAT_32, data, tensor);
}

iree_status_t iree_utils_i32_tensor_create(iree_runtime_session_t* session,
                                           const iree_hal_dim_t* shape,
                                           iree_host_size_t shape_rank,
                                           const int32_t* data,
                                           iree_hal_buffer_view_t** tensor) {
  return iree_utils_tensor_create(session, shape, shape_rank,
                                  IREE_HAL_ELEMENT_TYPE_INT_32, data, tensor);
}

void iree_utils_tensor_print(iree_runtime_session_t* session,
                             iree_hal_buffer_view_t* tensor) {
  IREE_IGNORE_ERROR(iree_hal_buffer_view_fprint(
      stdout, tensor, /*max_element_count=*/4096,
      iree_runtime_session_host_allocator(session)));
}

/*
   Please make sure to load the module before make calls.  The caller
   must free up the call by calling iree_runtime_call_deinitialize.
 */
iree_status_t iree_utils_make_call(iree_runtime_session_t* session,
                                   const char* function_name,
                                   iree_runtime_call_t* call, int num_args,
                                   ...) {
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view(function_name), call));

  va_list args;
  va_start(args, num_args);
  for (int i = 0; i < num_args; i++) {
    iree_hal_buffer_view_t* arg = va_arg(args, iree_hal_buffer_view_t*);
    IREE_RETURN_IF_ERROR(
        iree_runtime_call_inputs_push_back_buffer_view(call, arg));
  }
  va_end(args);
  return iree_runtime_call_invoke(call, /*flags=*/0);
}

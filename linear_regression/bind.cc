// Copyright 2023 Apple Inc.
#include "linear_regression/bind.h"

#include <stdlib.h>

#include "iree_utils/iree_utils.h"

iree_status_t logistic_regression_train_step(iree_runtime_session_t* session,
                                             iree_hal_buffer_view_t* x,
                                             iree_hal_buffer_view_t* y) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_utils_make_call(
      session, "linear_regression.train_step", &call, 2, x, y));
  iree_runtime_call_deinitialize(&call);
  return iree_ok_status();
}

iree_status_t logistic_regression_get_params(iree_runtime_session_t* session) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(
      iree_utils_make_call(session, "linear_regression.get_params", &call, 0));

  // TODO(wyi): get values out of the buffer and return in a C/C++ array.
  iree_hal_buffer_view_t* ret = NULL;
  IREE_RETURN_IF_ERROR(
      iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret));
  printf("params = \n");
  iree_utils_tensor_print(session, ret);
  iree_hal_buffer_view_release(ret);

  iree_runtime_call_deinitialize(&call);
  return iree_ok_status();
}

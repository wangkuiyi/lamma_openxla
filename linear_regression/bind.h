// Copyright 2023 Apple Inc.
#ifndef LINEAR_REGRESSION_BIND_H_
#define LINEAR_REGRESSION_BIND_H_

// This file defines the C bindings of MLIR methods in the module
// linear_regression, which was derived by iree-jax from
// linear_regression.py.

#include <iree/runtime/api.h>

// We offer C other than C++ bindings of MLIR so Swift and Objective-C
// programs can call them.
#ifdef __cplusplus
extern "C" {
#endif

// INPUT_SHAPE and OUTPUT_SHAPE in linear_regression.py
const iree_hal_dim_t input_shape[2] = {1, 4};
const iree_hal_dim_t output_shape[2] = {1, 1};

// LinearRegressionProgram.train_step in export_mlir.py
extern iree_status_t logistic_regression_train_step(
    iree_runtime_session_t* session, iree_hal_buffer_view_t* x,
    iree_hal_buffer_view_t* y);

// LinearRegressionProgram.get_params in export_mlir.py
extern iree_status_t logistic_regression_get_params(
    iree_runtime_session_t* session);

// TODO(wyi): Add C/C++ binding of the MLIR method linear_regression.predict

#ifdef __cplusplus
}
#endif

#endif  // LINEAR_REGRESSION_BIND_H_

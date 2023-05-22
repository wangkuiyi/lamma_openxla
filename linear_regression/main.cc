// Copyright 2023 Apple Inc.
#include <iostream>
#include <random>

#include "iree_utils/iree_utils.h"
#include "linear_regression/bind.h"

static void synthesize_data(int epoch_size, float* x_data, float* y_data) {
  std::mt19937 rng(0);
  for (int i = 0; i < epoch_size; i++) {
    y_data[i] = 0.0f;
    for (int j = 0; j < 4; j++) {
      float r = static_cast<float>(rng()) / static_cast<float>(rng.max());
      x_data[i * 4 + j] = r;
      y_data[i] += r;
    }
  }
}

static iree_status_t train_and_predict(const char* vmfb_path, int epoch_size,
                                       float* x_data, float* y_data) {
  // Create the IREE runtime instance.
  iree_runtime_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(iree_utils_runtime_instance_create(&instance));

  // Create a session in the instance to hold the context of a program
  // run.
  iree_runtime_session_t* session = NULL;
  IREE_RETURN_IF_ERROR(
      iree_utils_runtime_session_create(instance, NULL, &session));

  // Load the program into the session.
  IREE_RETURN_IF_ERROR(iree_utils_load_module(session, vmfb_path));

  // Call the function linear_regression.train_step repeatedly to
  // train the model.
  for (int i = 0; i < epoch_size; ++i) {
    iree_hal_buffer_view_t* x = NULL;
    iree_hal_buffer_view_t* y = NULL;
    IREE_RETURN_IF_ERROR(iree_utils_f32_tensor_create(session, input_shape, 2,
                                                      x_data + i * 4, &x));
    IREE_RETURN_IF_ERROR(
        iree_utils_f32_tensor_create(session, output_shape, 2, y_data + i, &y));

    IREE_RETURN_IF_ERROR(logistic_regression_train_step(session, x, y));

    iree_hal_buffer_view_release(x);
    iree_hal_buffer_view_release(y);
  }

  // Call linear_regression.get_params to retrieve and print the estimated model
  // parameters.
  IREE_RETURN_IF_ERROR(logistic_regression_get_params(session));

  // Release the session and the runtime instance.
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "Usage: " << argv[0] << " vmfb_file \n"
              << "Note: the vmfb file must be built for Metal GPU";
    return -1;
  }
  const char* vmfb_path = argv[1];

  const int kEpochSize = 5000;
  float x[kEpochSize * 4];
  float y[kEpochSize];
  synthesize_data(kEpochSize, x, y);

  return iree_utils_status_print_and_free(
      train_and_predict(vmfb_path, kEpochSize, x, y));
}

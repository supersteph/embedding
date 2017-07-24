// Copyright by Naturali. 2016
// Author Xibai, Pluto, Sean
// All rights reserved.

#include <memory>
#include <string>
#include <utility>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
using namespace tensorflow;

template<typename T>
class LookaheadCpuOp : public OpKernel {
 public:
  explicit LookaheadCpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    auto input = input_tensor.tensor<T, 3>();


    auto TS = input_tensor.dim_size(0);
    auto B = input_tensor.dim_size(1);
    auto F = input_tensor.dim_size(2);
    // Create output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor));
    auto output = output_tensor->template tensor<T, 3>();

    for (int t = 0; t < TS; t++) {
      for (int b = 0; b < B; b++) {
        int i = 0;
        int count = 0;
        for (int f = 0; f < F; f++) {
          if(i==f){
            output(t,b,i) = input(t,b,i);
            int num1 = 2*count+1;
            int num2 = 2*count+2;
            if(num1>F){
              continue;
            }
            if(input(t,b,num1)>input(t,b,num2)){
              i = num1;
            }
            else{
              i = num2;
            }
          }
          else{
            output(t,b,f)=0;

          }
        }
      }
    }
  }
};
REGISTER_KERNEL_BUILDER(Name("Lookahead").Device(DEVICE_CPU), LookaheadCpuOp<float>);

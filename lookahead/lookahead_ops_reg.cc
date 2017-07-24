#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"


using namespace tensorflow;
REGISTER_OP("Lookahead")
	.Attr("T: realnumbertype")
	.Input("input: T")
	.Output("output: T");
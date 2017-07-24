from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
from tensorflow.python.framework import ops
from plugins.op_compile import OperaterCompiler

compiler = OperaterCompiler('Lookahead', osp.dirname(osp.abspath(__file__)))
compiler.record_cpu_basis(
    ['lookahead_ops.cc', 'lookahead_ops_reg.cc'],
    '_lookahead_ops.so'
)
_lookahead_ops_so = compiler.compile()


def lookahead(x1):
    return _lookahead_ops_so.lookahead(x1)

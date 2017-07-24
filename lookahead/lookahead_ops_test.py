from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import plugins.lookahead.lookahead_ops as lookahead_ops


class LookaheadTest(tf.test.TestCase):
    _use_gpu = True

    def testLookahead(self):
        with self.test_session(use_gpu=self._use_gpu, graph=tf.Graph()):
            x1 = [[[1.0,2.0,3.0]]]
            result = lookahead_ops.lookahead(x1)
            self.assertAllEqual(
                result.eval(),
                [[[1.0,0.0,3.0]]])

            self.assertAllEqual(tf.test.is_built_with_cuda(), 1)

if __name__ == "__main__":
    tf.test.main()

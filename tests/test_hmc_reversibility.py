import numpy as np
import unittest
import scipy.stats

import numdifftools as nd


from sigvisa.infer.mcmc_basic import hmc_step_reversing, hmc_step


class TestReversibleHMC(unittest.TestCase):


    def setUp(self):

        C = np.array([[4, 5.8], [5.8, 9]])
        rv = scipy.stats.multivariate_normal(cov=C)
        self.logpdf = rv.logpdf

        J = nd.Jacobian(rv.logpdf)
        self.logpdf_grad = lambda z : J(z.flatten()).flatten()


    def test_reversible_noblocks(self):
        seed = 0

        np.random.seed(seed)
        x0 = np.random.randn(2)
        p0 = np.random.randn(2)

        x1, p0, p1, accept_lp = hmc_step(x0, self.logpdf, self.logpdf_grad, 10, 0.1, force_p=p0)

        x00, p1, p00, accept_lp = hmc_step(x1, self.logpdf, self.logpdf_grad, 10, 0.1, force_p=-p1)

        self.assertTrue( (np.abs(x00-x0) < 1e-8).all()  )
        self.assertTrue( (np.abs(p00+p0) < 1e-8).all()  )


    def _test_reversibility(self, num_blocks, block_size, min_block_std=1, max_block_std=1000, seed=0):

        np.random.seed(seed)
        x0 = np.random.randn(2)
        p0 = np.random.randn(2)

        x1, p0, p1, accept_lp = hmc_step_reversing(x0, self.logpdf, self.logpdf_grad, num_blocks, 0.1, force_p=p0, block_size=block_size, min_block_std=min_block_std, max_block_std=max_block_std)

        x00, p1, p00, accept_lp = hmc_step_reversing(x1, self.logpdf, self.logpdf_grad, num_blocks, 0.1, force_p=-p1, block_size=block_size, min_block_std=min_block_std, max_block_std=max_block_std)

        self.assertTrue( (np.abs(x00-x0) < 1e-8).all()  )
        self.assertTrue( (np.abs(p00+p0) < 1e-8).all()  )


    def test_quickreverse(self):
        self._test_reversibility(22, block_size=3, min_block_std=99999)

    def test_noreverse(self):
        self._test_reversibility(22, block_size=2, min_block_std=0, max_block_std=9999999)
        self._test_reversibility(5, block_size=5, min_block_std=0, max_block_std=9999999)

    def test_eventual_reverse(self):

    # cases to test:
    # go for multiple blocks, reverse but don't get back to the origin
        self._test_reversibility(7, block_size=13, min_block_std=0.021, max_block_std=0.022, seed=0)

    # go for multiple blocks, reverse past the origin but don't double-reverse
        self._test_reversibility(10, block_size=13, min_block_std=0.021, max_block_std=0.022, seed=0)

    # go for multiple blocks, reverse past the origin, then double-reverse after a different number of multiple blocks
        self._test_reversibility(20, block_size=13, min_block_std=0.021, max_block_std=0.022, seed=0)

    #
    #    self._test_reversibility(5, block_size=5, min_block_std=0, max_block_std=9999999)
    #    self._test_reversibility(10, block_size=13, min_block_std=0, max_block_std=9999999)



if __name__ == '__main__':
    unittest.main()

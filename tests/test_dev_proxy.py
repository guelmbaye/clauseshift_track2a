import unittest
from tools.compare_e0_e1c_dev import spearman
class ProxyTests(unittest.TestCase):
    def test_perfect_and_reverse(self):
        self.assertAlmostEqual(spearman([0,1,2,3],[0,1,2,3]),1)
        self.assertAlmostEqual(spearman([0,1,2,3],[3,2,1,0]),-1)
    def test_constant_returns_none(self):
        self.assertIsNone(spearman([0,0,0],[1,2,3]))
    def test_ties(self):
        self.assertAlmostEqual(spearman([0,0,1,1],[1,1,3,3]),1)
if __name__=='__main__':unittest.main()

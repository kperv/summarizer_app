import unittest

from model.py import Model

class TestModel(unittest.TestCase):

    def setUp(self):
        self.model = Model()

    def test_dataloader(self):
        pass

    def test_input_shape(self):
        pass


if __name__ == '__main__':
    unittest.main()
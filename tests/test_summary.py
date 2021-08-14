import unittest



class TestSummary(unittest.TestCase):

    def test_isEmpty(self):
        self.assertIsNotNone(summary)

    def test_isString(self):
        self.assertIsInstance(summary, 'str')

    def test_summary_length(self):
        self.assertEqual(len(summary), NUMBER)

if __name__ == '__main__':
    unittest.main()
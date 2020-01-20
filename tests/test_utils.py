import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
from utils import (
    remove_delimiter,
    remove_separator,
    remove_empty,
    remove_two_spaces,
    remove_three_spaces,
)
import unittest

class TestTextCleaning(unittest.TestCase):
    def test_remove_delimiter(self):
        """
        Test that it can sum a list of integers
        """
        text = "grab food is my \n fav lah"
        self.assertEqual(remove_delimiter(text), "grab food is my  fav lah")
        text = "grab food is my fav lah \n "
        self.assertEqual(remove_delimiter(text), "grab food is my fav lah  ")
        text = "grab food is my \n fav lah \n "
        self.assertEqual(remove_delimiter(text), "grab food is my  fav lah  ")

    def test_remove_separator(self):
        """
        Test that it can sum a list of integers
        """
        text = "grab food is my \r fav lah"
        self.assertEqual(remove_separator(text), "grab food is my  fav lah")
        text = "grab food is my fav lah \r "
        self.assertEqual(remove_separator(text), "grab food is my fav lah  ")

    def test_remove_two_spaces(self):
        """
        Test that it can sum a list of integers
        """
        text = "grab food is my fav lah  "
        self.assertEqual(remove_two_spaces(text), "grab food is my fav lah ")
        text = "grab food is my fav   lah"
        self.assertEqual(remove_two_spaces(text), "grab food is my fav  lah")

if __name__ == '__main__':
    unittest.main()
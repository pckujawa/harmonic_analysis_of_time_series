import json
import os
import sys
import unittest

import numpy as np

from hants import get_starter_matrix

this_file_dir = os.path.dirname(os.path.abspath(__file__))


def get_valid_file_path(subdir, test_name):
    filename = '{}.json'.format(test_name)
    filepath = os.path.join(this_file_dir, subdir, filename)
    try:
        os.mkdir(os.path.dirname(filepath))
    except:
        pass  # folder already exists, most likely
    return filepath


def get_approved(test_name):
    filepath = get_valid_file_path('approved', test_name)
    with open(filepath) as f:
        return json.load(f)


def set_received(test_name, value):
    filepath = get_valid_file_path('received', test_name)
    with open(filepath, 'wb') as f:
        json.dump(value, f, sort_keys=True, indent=2)


class HantsTests(unittest.TestCase):

    def test_get_starter_matrix(self):
        """
        This test uses a hand-rolled copy of
        https://github.com/approvals/ApprovalTests.Python
        """
        this_test_name = sys._getframe().f_code.co_name
        base_period_len = 26
        actual = get_starter_matrix(base_period_len, base_period_len, 3).tolist()  # ndarray can't serialize

        ang = 2 * np.pi * np.arange(base_period_len) / base_period_len
        cs = np.cos(ang)
        sn = np.sin(ang)
        self.assertTrue(np.allclose(actual[0], np.ones_like(actual[0])))
        self.assertTrue(np.allclose(actual[1], cs))
        self.assertTrue(np.allclose(actual[3], np.tile(cs[::2], 2)))
        self.assertTrue(np.allclose(actual[5], np.concatenate((cs[::3], cs[1::3], cs[2::3]))))
        self.assertTrue(np.allclose(actual[2], sn))
        self.assertTrue(np.allclose(actual[4], np.tile(sn[::2], 2)))
        self.assertTrue(np.allclose(actual[6], np.concatenate((sn[::3], sn[1::3], sn[2::3]))))

        try:
            expected = get_approved(this_test_name)
            self.assertSequenceEqual(expected, actual)
        except (AssertionError, IOError) as exc:
            set_received(this_test_name, actual)
            self.fail(exc)


if __name__ == '__main__':
    unittest.main(verbosity=2)  # show test names

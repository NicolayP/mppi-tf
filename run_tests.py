import unittest
import argparse

from tests.test_integrate import integrate

def parse_args():
    parser = argparse.ArgumentParser(prog="Test",
                                     description="Trains AUV model with pypose")

    parser.add_argument("-i", "--integrate", action=argparse.BooleanOptionalAction,
                        help="If set, this will run integration function on the dataset.\
                        It helps to see if everything is working as expected")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()

    if args.integrate:
        integrate("data/csv/tests2")
        exit()

    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    unittest.TextTestRunner().run(test_suite)
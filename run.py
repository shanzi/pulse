#!/usr/bin/env python
# encoding: utf-8

from pulse import run_test

def main():
    run_test('./data/Train/', './data/Test', './output/')


if __name__ == '__main__':
    main()

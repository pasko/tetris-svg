#!/usr/bin/python2.7

import sys


def main():
  message = """
<svg version="1.1"
     baseProfile="full"
     width="200" height="200"
     xmlns="http://www.w3.org/2000/svg">

  <polyline fill="none" stroke="black"
      points="10,10 50,10 50,20 10,20 10,10"/>

</svg>
"""
  print message
  return 0


if __name__ == '__main__':
  sys.exit(main())

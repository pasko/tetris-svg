#!/usr/bin/python2.7

from __future__ import print_function
import sys


TILES = [(1,1), (2,1), (3,1), (4,1)]
TILE_SIZE = 10


SVG_HEADER = (
"""
<svg version="1.1"
     baseProfile="full"
     width="200" height="200"
     xmlns="http://www.w3.org/2000/svg">
""")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


def add_vector(xy_tuple, xy_vector):
  return (xy_tuple[0] + xy_vector[0], xy_tuple[1] + xy_vector[1])


def is_positively_oriented_basis(ab, cd):
  (a, b) = ab
  (c, d) = cd
  return (a * d - b * c) > 0


class Tile(object):
  """Represents a 2D square tile with a position and size."""

  def __init__(self, xy, size):
    assert isinstance(xy, tuple) and len(xy) == 2
    self.x = xy[0]
    self.y = xy[1]
    self.size = size

  @staticmethod
  def __check_line_type(line):
    assert isinstance(line, list)
    assert len(line) == 2
    [xy_start, xy_end] = line
    assert isinstance(xy_start, tuple) and len(xy_start) == 2
    assert isinstance(xy_end, tuple) and len(xy_end) == 2

  def get_corners(self):
    x = self.x
    y = self.y
    size = self.size
    return [(x, y), (x + size, y), (x, y + size), (x + size, y + size)]

  def is_edge_line(self, line):
    Tile.__check_line_type(line)
    [xy_start, xy_end] = line
    corners = self.get_corners()
    if not xy_start in corners or not xy_end in corners:
      return False
    if xy_start[0] == xy_end[0] or xy_start[1] == xy_end[1]:
      return True
    return False

  def is_positive_edge_to_line(self, line):
    Tile.__check_line_type(line)
    if not self.is_edge_line(line):
      return False
    [xy_start, xy_end] = line
    line_vector = (xy_end[0] - xy_start[0], xy_end[1] - xy_start[1])
    tile_center = (self.x + self.size / 2.0, self.y + self.size / 2.0)
    vector_to_center = (tile_center[0] - xy_start[0],
                        tile_center[1] - xy_start[1])
    return is_positively_oriented_basis(vector_to_center, line_vector)


def main():
  test()
  print(SVG_HEADER)
  print('<polyline fill="none" stroke="black" points="')

  possible_vectors = [(1, 0), (0, 1), (-1, 0), (0, -1)]
  initial_vector = (1, 0)
  initial_x = min(TILES, key=lambda t: t[0])[0]
  initial_y = min(TILES, key=lambda t: t[1])[1]
  initial_point = (initial_x, initial_y)
  polyline = [initial_point]
  next_point = add_vector(initial_point, initial_vector)

  print('10,10 50,10 50,20 10,20 10,10')

  print('"/>')
  print('</svg>')
  return 0


def test():
  tile = Tile((1, 2), 5)
  corners = tile.get_corners()
  assert len(corners) == 4
  assert (1, 2) in corners
  assert (6, 2) in corners
  assert (1, 7) in corners
  assert (6, 7) in corners
  assert tile.is_edge_line([(1, 2), (1, 7)])
  assert tile.is_edge_line([(1, 7), (1, 2)])
  assert not tile.is_edge_line([(1, 8), (1, 2)])
  assert not tile.is_edge_line([(1, 2), (6, 7)])
  assert not tile.is_edge_line([(6, 7), (1, 2)])

  assert is_positively_oriented_basis((1, 0), (0, 1))
  assert not is_positively_oriented_basis((-1, 0), (0, 1))

  assert tile.is_positive_edge_to_line([(1, 2), (1, 7)])
  assert tile.is_positive_edge_to_line([(1, 7), (6, 7)])
  assert tile.is_positive_edge_to_line([(6, 7), (6, 2)])
  assert tile.is_positive_edge_to_line([(6, 2), (1, 2)])
  assert not tile.is_positive_edge_to_line([(1, 7), (1, 2)])

  eprint('Smoke tests passed.')


if __name__ == '__main__':
  sys.exit(main())

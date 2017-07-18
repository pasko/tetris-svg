#!/usr/bin/python2.7

from __future__ import print_function

import collections
import copy
import sys


TILES = [(1,1), (2,1), (3,1), (4,1)]
TILE_SIZE = 30
OFFSET_XY = (50, 50)
MARGIN = 5
assert MARGIN * 2 < TILE_SIZE


SVG_HEADER = (
"""
<svg version="1.1"
     baseProfile="full"
     width="800" height="800"
     xmlns="http://www.w3.org/2000/svg">
""")


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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
    return [(x, y), (x + size, y), (x + size, y + size), (x, y + size)]

  def get_opposite_corner(self, point):
    corners = self.get_corners()
    assert point in corners
    for corner in corners:
      if corner[0] != point[0] and corner[1] != point[1]:
        return corner
    return None

  def get_edges(self):
    result_lines = []
    corners = self.get_corners()
    prev_i = 0
    for i in xrange(1, 4):
      result_lines.append([corners[prev_i], corners[i]])
      prev_i = i
    result_lines.append([corners[3], corners[0]])
    return result_lines

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
    tile_center = (self.x + 1, self.y + 1)
    vector_to_center = (tile_center[0] - xy_start[0],
                        tile_center[1] - xy_start[1])
    return is_positively_oriented_basis(vector_to_center, line_vector)


class IndexableLine(object):
  """A line that can be put in a set, with direction ignored.

  Also has the ability to hold a reference to a Tile. When generating figures
  from tiles, each indexable line could reference the tile which it was
  generated from. This way all segments of an outline of a figure knows on which
  side the figure is.
  """
  def __init__(self, line, tile=None):
    [self.xy_start, self.xy_end] = line
    # The reference to the |tile| is ignored in hashing and comparisons.
    self.tile = tile

  def get_tile(self):
    assert self.tile is not None
    return self.tile

  def __eq__(self, other):
    if self.xy_start == other.xy_start and self.xy_end == other.xy_end:
      return True
    if self.xy_start == other.xy_end and self.xy_end == other.xy_start:
      return True
    return False

  def __hash__(self):
    return hash(self.xy_start) ^ hash(self.xy_end)

  def as_line(self):
    return [self.xy_start, self.xy_end]


def optimize_lines(indexable_lines):
  lines = set(indexable_lines)
  current_line = lines.pop()
  initial_point = current_line.xy_start
  path = []
  path.append(current_line)

  point_to_adjacent_lines = collections.defaultdict(list)
  for indexable_line in lines:
    point_to_adjacent_lines[indexable_line.xy_start].append(indexable_line)
    point_to_adjacent_lines[indexable_line.xy_end].append(indexable_line)

  # TODO: Allow converting to multiple paths if there is no single traversal
  # over the set of lines.
  while initial_point != current_line.xy_end:
    current_point = current_line.xy_end
    while True:
      current_line = point_to_adjacent_lines[current_point].pop()
      if current_line in lines:
        break
    lines.remove(current_line)
    if current_line.xy_start != current_point:
      assert current_line.xy_end == current_point
      opposite_direction_line = IndexableLine(
          [current_line.xy_end, current_line.xy_start], tile=line.get_tile())
      current_line = opposite_direction_line
    path.append(current_line)
  assert len(lines) == 0

  return path


def optimize_lines_to_path(indexable_lines):
  optimized = optimize_lines(indexable_lines)
  initial_line = optimized[0].as_line()
  initial_point = initial_line[0]
  current_point = initial_line[1]
  path = []
  path.append(initial_point)
  path.append(current_point)
  for indexable_line in optimized[1:]:
    assert indexable_line.as_line()[0] == current_point
    current_point = indexable_line.as_line()[1]
    path.append(current_point)
  return path


def check_path_is_equivalent_to_lines(path, lines):
  lines_as_set = set(lines)
  previous_point = path[0]
  for point in path[1:]:
    # Raise KeyError if the current segment is not present.
    lines_as_set.remove(IndexableLine([previous_point, point]))
    previous_point = point
  assert len(lines_as_set) == 0


def draw_simple_lines(lines):
  for line in lines:
    print('<line stroke="black" x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(
        line[0][0], line[0][1], line[1][0], line[1][1]))


class TiledFigure(object):
  def __init__(self, size, tile_starting_points):
    self.size = size
    self.tile_starting_points = copy.copy(tile_starting_points)
    self.outline_with_tiles = None

  def init_outline(self):
    all_lines = {}
    for point in self.tile_starting_points:
      tile = Tile(point, self.size)
      for edge in tile.get_edges():
        indexable_edge = IndexableLine(edge, tile=tile)
        if not indexable_edge in all_lines:
          all_lines[indexable_edge] = False
        else:
          all_lines[indexable_edge] = True

    self.outline_with_tiles = []
    for indexable_line, visited in all_lines.iteritems():
      if not visited:
        self.outline_with_tiles.append(indexable_line)

  def get_outline(self):
    """Returns an outline of the figure as a list of lines."""
    return [l.as_line() for l in self.outline_with_tiles]

  def get_outline_with_tiles(self):
    return self.outline_with_tiles

  @staticmethod
  def _connect_nested_dashes(previous, current):
    ret = []
    if previous[1] == current[0]:
      ret.append(current)
      return ret
    sign_vector = (cmp(previous[1][0] - previous[0][0], 0),
        cmp(previous[1][1] - previous[0][1], 0))
    next_dash = [previous[1],
        (previous[1][0] + sign_vector[0] * 2 * MARGIN,
         previous[1][1] + sign_vector[1] * 2 * MARGIN)]
    ret.append(next_dash)
    if next_dash[1] == current[0]:
      return ret
    ret.append([next_dash[1], current[0]])
    ret.append(current)
    return ret

  def get_nested_outline(self, margin):
    outline = optimize_lines(self.outline_with_tiles)
    nested = []
    previous_dash = None
    initial_dash = None
    for line in outline:
      tile = line.get_tile()
      xy_start = self._shift_point_towards_another(
          line.xy_start, tile.get_opposite_corner(line.xy_start), margin)
      xy_end = self._shift_point_towards_another(
          line.xy_end, tile.get_opposite_corner(line.xy_end), margin)
      current_dash = [xy_start, xy_end]
      if previous_dash is None:
        initial_dash = current_dash
        nested.append(current_dash)
      else:
        for l in self._connect_nested_dashes(previous_dash, current_dash):
          nested.append(l)
      previous_dash = current_dash
    for l in self._connect_nested_dashes(previous_dash, initial_dash):
      nested.append(l)
    return nested

  @staticmethod
  def _shift_point_towards_another(point, point2, margin):
    direction_step = cmp(point2[0] - point[0], 0) # sign
    ret_x = point[0] + margin * direction_step
    direction_step = cmp(point2[1] - point[1], 0) # sign
    ret_y = point[1] + margin * direction_step
    return [ret_x, ret_y]

  def draw_outline(self):
    for edge in self.get_outline():
      print('<polyline fill="none" stroke="black" points="')
      print('{},{} {},{}'.format(edge[0][0], edge[0][1], edge[1][0], edge[1][1]))
      print('"/>')


def has_segment_in_lines(segment, lines):
  indexed_lines = set()
  for l in lines:
    indexed_lines.add(IndexableLine(l))
  return (IndexableLine(segment) in indexed_lines)


def add_figure(figures, nested_outlines, move_to_xy, tile_starting_points):
  transformed_points = []
  for point in tile_starting_points:
    transformed_points.append(
        (OFFSET_XY[0] + TILE_SIZE * (move_to_xy[0] + point[0]),
         OFFSET_XY[1] + TILE_SIZE * (move_to_xy[1] + point[1])))
  figure = TiledFigure(TILE_SIZE, transformed_points)
  figure.init_outline()
  figures.append(figure)
  for line in figure.get_nested_outline(MARGIN):
    nested_outlines.append(line)


def main():
  test()
  print(SVG_HEADER)
  figures = []
  nested = []
  piece_t1 = [(0, 0), (0, 1), (1, 1), (0, 2)]
  add_figure(figures, nested, (0, 0), piece_t1)

  piece_i = [(i, 0) for i in xrange(0, 4)]
  add_figure(figures, nested, (1, 0), piece_i)

  piece_s = [(1, 0), (1, 1), (0, 1), (0, 2)]
  add_figure(figures, nested, (1, 1), piece_s)

  piece_o = [(0, 0), (0, 1), (1, 0), (1, 1)]
  add_figure(figures, nested, (3, 1), piece_o)

  piece_l1 = [(0, 0), (0, 1), (1, 1), (2, 1)]
  add_figure(figures, nested, (0, 3), piece_l1)

  piece_l2 = [(0, 0), (1, 0), (2, 0), (2, 1)]
  add_figure(figures, nested, (2, 3), piece_l2)

  piece_t2 = [(0, 1), (1, 0), (1, 1), (1, 2)]
  add_figure(figures, nested, (3, 5), piece_t2)

  add_figure(figures, nested, (2, 4), piece_s)
  add_figure(figures, nested, (0, 5), piece_o)
  add_figure(figures, nested, (0, 7), piece_i)

  for f in figures:
    f.draw_outline()

  draw_simple_lines(nested)
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
  assert tile.get_opposite_corner((1, 2)) == (6, 7)
  assert tile.get_opposite_corner((1, 7)) == (6, 2)

  for edge in tile.get_edges():
    assert tile.is_edge_line(edge)

  assert is_positively_oriented_basis((1, 0), (0, 1))
  assert not is_positively_oriented_basis((-1, 0), (0, 1))

  assert tile.is_positive_edge_to_line([(1, 2), (1, 7)])
  assert tile.is_positive_edge_to_line([(1, 7), (6, 7)])
  assert tile.is_positive_edge_to_line([(6, 7), (6, 2)])
  assert tile.is_positive_edge_to_line([(6, 2), (1, 2)])
  assert not tile.is_positive_edge_to_line([(1, 7), (1, 2)])

  # Check that exchanging xy_start and xy_end on an IndexableLine produces the
  # same key for indexing in a set.
  s = set()
  s.add(IndexableLine([(1, 2), (3, 4)]))
  s.add(IndexableLine([(3, 4), (1, 2)]))
  assert len(s) == 1
  s.add(IndexableLine([(2, 1), (3, 4)]))
  assert len(s) == 2

  figure = TiledFigure(10, [(10, 10), (20, 10), (30, 10), (40, 10)])
  figure.init_outline()
  outline = figure.get_outline()
  assert len(outline) == 10
  assert has_segment_in_lines([(10,10), (20,10)], outline)
  assert has_segment_in_lines([(10,10), (10,20)], outline)
  assert has_segment_in_lines([(50,10), (50,20)], outline)
  assert not has_segment_in_lines([(20,10), (20,20)], outline)
  assert not has_segment_in_lines([(30,10), (30,20)], outline)

  a = (0, 0)
  b = (0, 1)
  c = (1, 1)
  check_path_is_equivalent_to_lines([a, b, c], [
    IndexableLine([c, b]),
    IndexableLine([a, b])])

  indexable_lines = figure.get_outline_with_tiles()
  check_path_is_equivalent_to_lines(
      optimize_lines_to_path(indexable_lines), indexable_lines)

  piece_t1 = [(0, 0), (0, 1), (1, 1), (0, 2)]
  figure_from_t1 = TiledFigure(1, piece_t1)
  figure_from_t1.init_outline()
  indexable_lines_from_t1 = figure.get_outline_with_tiles()
  check_path_is_equivalent_to_lines(
      optimize_lines_to_path(indexable_lines_from_t1), indexable_lines_from_t1)

  eprint('Smoke tests passed.')


if __name__ == '__main__':
  sys.exit(main())

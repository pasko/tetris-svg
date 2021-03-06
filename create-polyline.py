#!/usr/bin/python2.7

from __future__ import print_function

import collections
import copy
import sys


# Let one unit of length be equal to 0.1mm.
OFFSET_XY = (30, 30)

# Parameters for currugated 6.7mm cardboard:
#     Maximum dimension: 940x565 mm
#     Minimum dimension: 15x15 mm
#     Minimum distance between two paths: 2mm
SHEET_X = 940
SHEET_Y = 565
TILE_SIZE = 100
MODULE_ROWS = 7
MODULE_COLUMNS = 9

MARGIN = 20
assert MARGIN * 2 < TILE_SIZE - 20

SVG_HEADER = (
"""
<svg version="1.1"
     baseProfile="full"
     width="{x}mm" height="{y}mm"
     viewBox="0 0 {x}0 {y}0"
     xmlns="http://www.w3.org/2000/svg">
""".format(x=SHEET_X, y=SHEET_Y))


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


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


def optimize_lines_to_traversals(indexable_lines):
  """Optimize the list of lines into several connected traversals.

  Each segment in the traversal is a line from the original list. Duplicates are
  removed.
  """
  lines = set(indexable_lines)
  current_line = lines.pop()
  path = []
  path.append(current_line)

  point_to_adjacent_lines = collections.defaultdict(list)
  for indexable_line in lines:
    point_to_adjacent_lines[indexable_line.xy_start].append(indexable_line)
    point_to_adjacent_lines[indexable_line.xy_end].append(indexable_line)

  traversals = []
  while len(lines) > 0:
    has_next_line = True
    while has_next_line:
      current_point = current_line.xy_end
      # TODO: 3 nested while loops is a little too much, move some.
      while True:
        remaining_adjacent_lines = point_to_adjacent_lines[current_point]
        if len(remaining_adjacent_lines) == 0:
          has_next_line = False
          break
        current_line = remaining_adjacent_lines.pop()
        if current_line in lines:
          has_next_line = True
          break
      if not has_next_line:
        if len(lines) > 0:
          traversals.append(path)
          path = []
          current_line = lines.pop()
          path.append(current_line)
        break  # From the outer loop
      else:
        lines.remove(current_line)
        if current_line.xy_start != current_point:
          assert current_line.xy_end == current_point
          opposite_direction_line = IndexableLine(
              [current_line.xy_end, current_line.xy_start],
              tile=current_line.get_tile())
          current_line = opposite_direction_line
        assert current_line.xy_start == current_point
        path.append(current_line)
  traversals.append(path)
  return traversals


def optimize_lines_to_paths(indexable_lines):
  traversals = optimize_lines_to_traversals(indexable_lines)
  paths = []
  for optimized in traversals:
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
    paths.append(path)
  return paths


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
  def __init__(self, size, tile_starting_points, margin):
    self.size = size
    self.margin = margin
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
  def _connect_nested_dashes(previous, current, margin):
    """Returns a list of lines connecting the |previous| and the |current|.

       If connected already, returns an empty list. Connects by continuing the
       |previous| until the line intersects with |current|. This should should
       happen at the length of 2 |margin|s.  If necessary, then connects with
       |current| adding the last line.
    """
    ret = []
    if previous[1] == current[0]:
      # Dashes share a point, hence no need to connect.
      return ret
    sign_vector = (cmp(previous[1][0] - previous[0][0], 0),
        cmp(previous[1][1] - previous[0][1], 0))
    next_dash = [previous[1],
        (previous[1][0] + sign_vector[0] * 2 * margin,
         previous[1][1] + sign_vector[1] * 2 * margin)]
    ret.append(next_dash)
    if next_dash[1] == current[0]:
      return ret
    ret.append([next_dash[1], current[0]])
    return ret

  def get_nested_outline(self):
    outlines = optimize_lines_to_traversals(self.outline_with_tiles)
    assert len(outlines) == 1  # Only singly-traversed figures are supported.
    outline = outlines[0]
    nested = []
    previous_dash = None
    initial_dash = None
    for line in outline:
      tile = line.get_tile()
      xy_start = self._shift_point_towards_another(
          line.xy_start, tile.get_opposite_corner(line.xy_start), self.margin)
      xy_end = self._shift_point_towards_another(
          line.xy_end, tile.get_opposite_corner(line.xy_end), self.margin)
      current_dash = [xy_start, xy_end]
      if previous_dash is None:
        initial_dash = current_dash
      else:
        for l in self._connect_nested_dashes(previous_dash, current_dash,
                                             self.margin):
          nested.append(l)
      nested.append(current_dash)
      previous_dash = current_dash

    # Make the final connection.
    for l in self._connect_nested_dashes(previous_dash, initial_dash,
                                         self.margin):
      nested.append(l)
    return nested

  @staticmethod
  def _shift_point_towards_another(point, point2, length):
    direction_step = cmp(point2[0] - point[0], 0) # sign
    ret_x = point[0] + length * direction_step
    direction_step = cmp(point2[1] - point[1], 0) # sign
    ret_y = point[1] + length * direction_step
    return (ret_x, ret_y)


def has_segment_in_lines(segment, lines):
  indexed_lines = set()
  for l in lines:
    indexed_lines.add(IndexableLine(l))
  return (IndexableLine(segment) in indexed_lines)


def add_figure(offset_xy, figures, nested_outlines, move_to_xy,
               tile_starting_points, margin):
  transformed_points = []
  for point in tile_starting_points:
    transformed_points.append(
        (offset_xy[0] + TILE_SIZE * (move_to_xy[0] + point[0]),
         offset_xy[1] + TILE_SIZE * (move_to_xy[1] + point[1])))
  figure = TiledFigure(TILE_SIZE, transformed_points, margin)
  figure.init_outline()
  figures.append(figure)
  if nested_outlines is not None:
    for line in figure.get_nested_outline():
      nested_outlines.append(line)


def add_figures_for_one_module(offset_xy, margin, figures, nested):
  if margin is not None:
    assert margin * 2 < TILE_SIZE - 20
  piece_t1 = [(0, 0), (0, 1), (1, 1), (0, 2)]
  add_figure(offset_xy, figures, nested, (0, 0), piece_t1, margin)

  piece_i = [(i, 0) for i in xrange(0, 4)]
  add_figure(offset_xy, figures, nested, (1, 0), piece_i, margin)

  piece_s = [(1, 0), (1, 1), (0, 1), (0, 2)]
  add_figure(offset_xy, figures, nested, (1, 1), piece_s, margin)

  piece_o = [(0, 0), (0, 1), (1, 0), (1, 1)]
  add_figure(offset_xy, figures, nested, (3, 1), piece_o, margin)

  piece_l1 = [(0, 0), (0, 1), (1, 1), (2, 1)]
  add_figure(offset_xy, figures, nested, (0, 3), piece_l1, margin)

  piece_l2 = [(0, 0), (1, 0), (2, 0), (2, 1)]
  add_figure(offset_xy, figures, nested, (2, 3), piece_l2, margin)

  piece_t2 = [(0, 1), (1, 0), (1, 1), (1, 2)]
  add_figure(offset_xy, figures, nested, (3, 5), piece_t2, margin)

  add_figure(offset_xy, figures, nested, (2, 4), piece_s, margin)
  add_figure(offset_xy, figures, nested, (0, 5), piece_o, margin)
  add_figure(offset_xy, figures, nested, (0, 7), piece_i, margin)


def draw_paths(paths):
  for path in paths:
    print('<polyline fill="none" stroke="black" points="')
    for point in path:
      print('{},{}'.format(point[0], point[1]))
    print('"/>')


def main():
  test()
  print(SVG_HEADER)
  figures = []
  nested = []

  # Draw a few of the figures in the row0 with margins.
  module_width = TILE_SIZE * 5
  module_height = TILE_SIZE * 8
  MODULES_WITH_NESTED_STUFF = 5
  assert MODULES_WITH_NESTED_STUFF <= MODULE_COLUMNS
  for i in xrange(MODULES_WITH_NESTED_STUFF):
    add_figures_for_one_module(
        (OFFSET_XY[0] + i * module_width, OFFSET_XY[1]),
        MARGIN + 2 * i,
        figures, nested)

  # Draw remaining figures without margins.
  for i in xrange(0, MODULE_ROWS * MODULE_COLUMNS):
    row = i / MODULE_COLUMNS
    column = i % MODULE_COLUMNS
    if row == 0 and column < MODULES_WITH_NESTED_STUFF:
      continue
    add_figures_for_one_module(
        (OFFSET_XY[0] + column * module_width,
         OFFSET_XY[1] + row * module_height),
        None,  # margin
        figures, None)

  # Gather all lines together, optimize them as a whole and draw.
  all_lines = []
  for f in figures:
    all_lines += f.get_outline_with_tiles()
  draw_paths(optimize_lines_to_paths(all_lines))

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
  assert tile.get_opposite_corner((1, 2)) == (6, 7)
  assert tile.get_opposite_corner((1, 7)) == (6, 2)

  # Check that exchanging xy_start and xy_end on an IndexableLine produces the
  # same key for indexing in a set.
  s = set()
  s.add(IndexableLine([(1, 2), (3, 4)]))
  s.add(IndexableLine([(3, 4), (1, 2)]))
  assert len(s) == 1
  s.add(IndexableLine([(2, 1), (3, 4)]))
  assert len(s) == 2

  # Check that a few segments are present in the outline of the "Piece I" - the
  # straight bar.
  figure = TiledFigure(10, [(10, 10), (20, 10), (30, 10), (40, 10)], margin=3)
  figure.init_outline()
  outline = figure.get_outline()
  assert len(outline) == 10
  assert has_segment_in_lines([(10,10), (20,10)], outline)
  assert has_segment_in_lines([(10,10), (10,20)], outline)
  assert has_segment_in_lines([(50,10), (50,20)], outline)
  assert not has_segment_in_lines([(20,10), (20,20)], outline)
  assert not has_segment_in_lines([(30,10), (30,20)], outline)

  # Check that the nested outline of the "Piece I" is as expected.
  nested_outline = figure.get_nested_outline()
  for l in nested_outline:
    l_x = l[0]
    l_y = l[1]
    assert l_x[0] >= 13 and l_x[1] <= 17
    assert l_y[0] >= 13 and l_y[1] <= 47
  assert len(nested_outline) == (10 +  # original segments
      2 * 3)  #  connections between segments
  assert has_segment_in_lines([(17, 13), (13, 13)], nested_outline)
  assert has_segment_in_lines([(27, 13), (33, 13)], nested_outline)

  a = (0, 0)
  b = (0, 1)
  c = (1, 1)
  check_path_is_equivalent_to_lines([a, b, c], [
    IndexableLine([c, b]),
    IndexableLine([a, b])])

  indexable_lines = figure.get_outline_with_tiles()
  paths = optimize_lines_to_paths(indexable_lines)
  assert len(paths) == 1  # This figure should be optimized to only one path.
  check_path_is_equivalent_to_lines(paths[0], indexable_lines)

  piece_t1 = [(0, 0), (0, 1), (1, 1), (0, 2)]
  figure_from_t1 = TiledFigure(1, piece_t1, margin=None)
  figure_from_t1.init_outline()
  indexable_lines_from_t1 = figure.get_outline_with_tiles()
  paths = optimize_lines_to_paths(indexable_lines_from_t1)
  assert len(paths) == 1  # This figure should be optimized to only one path.
  check_path_is_equivalent_to_lines(paths[0], indexable_lines_from_t1)

  eprint('Smoke tests passed.')


if __name__ == '__main__':
  sys.exit(main())

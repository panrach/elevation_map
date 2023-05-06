"""Assignment 2 functions."""

from copy import deepcopy

# Examples to use in doctests:

THREE_BY_THREE = [[1, 2, 1],
                  [4, 6, 5],
                  [7, 8, 9]]

FOUR_BY_FOUR = [[1, 2, 6, 5],
                [4, 5, 3, 2],
                [7, 9, 8, 1],
                [1, 2, 1, 4]]

UNIQUE_3X3 = [[1, 2, 3],
              [9, 8, 7],
              [4, 5, 6]]

UNIQUE_4X4 = [[10, 2, 3, 30],
              [9, 8, 7, 11],
              [4, 5, 6, 12],
              [13, 14, 15, 16]]


# Used to compare floats in doctests:
# If the difference between the expected return value and the actual return
# value is less than EPSILON, we will consider the test passed.
EPSILON = 0.005

# Used to check if an elevation is invalid
INVALID_ELEVATION = 0


# We provide a full docstring for this function as an example.
def compare_elevations_within_row(elevation_map: list[list[int]], map_row: int,
                                  level: int) -> list[int]:
    """
    Return a new list containing the three counts: the number of
    elevations from row number map_row of elevation map elevation_map
    that are less than, equal to, and greater than elevation level.

    Precondition: elevation_map is a valid elevation map.
                  0 <= map_row < len(elevation_map).

    >>> compare_elevations_within_row(THREE_BY_THREE, 1, 5)
    [1, 1, 1]
    >>> compare_elevations_within_row(FOUR_BY_FOUR, 1, 2)
    [0, 1, 3]

    """

    num_less = 0
    num_greater = 0
    num_equal = 0

    for elevation_level in elevation_map[map_row]:
        if elevation_level < level:
            num_less += 1
        elif elevation_level > level:
            num_greater += 1
        else:
            num_equal += 1

    return [num_less, num_equal, num_greater]


# We provide a partial doctest in this function as an example of
# testing a function that modifies its input. Note the use of deepcopy
# to create a copy of the nested list to use in the function call. We
# do this to make sure that different doctests do not affect each
# other.
def update_elevation(elevation_map: list[list[int]], start: list[int],
                     stop: list[int], delta: int) -> None:
    """
    Modify elevation_map so that the cells between start cell and stop cell
    inclusive change by delta elevation.

    Preconditions:
    - elevation_map is valid
    - start cell and stop cell are in the same column, row, or both
    - If they are in the same row, start cell column value is less than or
    equal to stop cell column value.
    - If they are in the same column, start cell row value is less than stop
    cell row value.
    - Delta will never cause the elevations to go below 1

    >>> THREE_BY_THREE_COPY = deepcopy(THREE_BY_THREE)
    >>> update_elevation(THREE_BY_THREE_COPY, [1, 0], [1, 1], -2)
    >>> THREE_BY_THREE_COPY
    [[1, 2, 1], [2, 4, 5], [7, 8, 9]]
    >>> FOUR_BY_FOUR_COPY = deepcopy(FOUR_BY_FOUR)
    >>> update_elevation(FOUR_BY_FOUR_COPY, [1, 2], [3, 2], 1)
    >>> FOUR_BY_FOUR_COPY
    [[1, 2, 6, 5], [4, 5, 4, 2], [7, 9, 9, 1], [1, 2, 2, 4]]

    """

    start_row = start[0]
    stop_row = stop[0]
    start_col = start[1]
    stop_col = stop[1]

    # changing one cell
    if start == stop:
        elevation_map[start_row][start_col] += delta

    # changing the row of cells
    elif start_row == stop_row:
        for col in range(start_col, stop_col + 1):
            elevation_map[start_row][col] += delta

    # changing column of cells
    elif start_col == stop_col:
        for row in range(start_row, stop_row + 1):
            elevation_map[row][stop_col] += delta


# We provide a partial doctest in this function as an example of
# testing a function that returns a float. Note the use of abs and
# EPSILON to check if two floats are "close enough". We do this to
# deal with inevitable errors that arise in floating point arithmetic.
def get_average_elevation(elevation_map: list[list[int]]) -> float:
    """
    Return the average elevation across all the cells in elevation_map.

    Precondition: elevation_map is valid.

    >>> abs(get_average_elevation(UNIQUE_3X3) - 5.0) < EPSILON
    True
    >>> abs(get_average_elevation(FOUR_BY_FOUR) - 3.8125) < EPSILON
    True

    """

    total = 0

    for mapp in elevation_map:
        for elevation in mapp:
            total += elevation

    return total/(len(elevation_map)) ** 2


def find_peak(elevation_map: list[list[int]]) -> list[int]:
    """
    Return the cell with the highest elevation in elevation_map.

    Preconditions: elevation_map is valid and all the elevations are unique.

    >>> find_peak(UNIQUE_3X3)
    [1, 0]
    >>> find_peak(UNIQUE_4X4)
    [0, 3]

    """

    highest = 0
    highest_cell = [[]]
    elevation = 0

    for i in range(len(elevation_map)):
        for j in range(len(elevation_map[i])):
            elevation = get_elevation(elevation_map, [i, j])
            if elevation > highest:
                highest = elevation
                highest_cell = [i, j]

    return highest_cell


def is_sink(elevation_map: list[list[int]], cell: list[int]) -> bool:
    """
    Return True if and only if cell is valid and a sink in elevation_map.

    Precondition: elevation_map is valid.

    >>> is_sink(UNIQUE_3X3, [0, 0])
    True
    >>> is_sink(UNIQUE_4X4, [3,2])
    False

    """

    adjacent_cells = get_adjacent_cells(cell, len(elevation_map))

    if not is_valid_cell(cell, len(elevation_map)):
        return False

    # if the adjacent elevation is greater than cell elevation,
    for adj_cell in adjacent_cells:
        if is_cell_lower(elevation_map, adj_cell, cell):
            return False

    return True


def find_local_sink(elevation_map: list[list[int]],
                    cell: list[int]) -> list[int]:
    """
    Return the local sink for cell in elevation_map.
    If cell is not a sink, then always return the adjacent cell with the lowest
    elevation.

    Preconditions:
    - elevation_map is valid
    - cell exists in elevation_map
    - all elevations in elevation_map are unique

    >>> find_local_sink(UNIQUE_3X3, [1, 1])
    [0, 0]
    >>> find_local_sink(UNIQUE_4X4, [0, 0])
    [0, 1]

    """

    lowest_cell = [cell[0], cell[1]]
    adjacent_cells = get_adjacent_cells(cell, len(elevation_map))

    if is_sink(elevation_map, cell):
        return cell

    for adj_cell in adjacent_cells:
        if is_cell_lower(elevation_map, adj_cell, lowest_cell):
            lowest_cell = adj_cell

    return lowest_cell


def can_hike_to(elevation_map: list[list[int]], start: list[int],
                dest: list[int], supplies: int) -> bool:
    """
    Return True if and only if start cell can be ewqual to dest cell
    in elevation_map elevation map with greater than or equal to 0 supplies.
    Otherwise, return False.

    The path taken from start to dest follows these rules:
    - the cell with the smallest change in elevation is the next direction
    - if both cells have the same change in elevation, then the next direction
    is North
    - the next direction never goes farther than dest
    - the next direction is a valid cell in elevation_map

    Preconditions:
    - dest is North-West of start
    - elevation_map is a valid elevation map
    - start and dest are valid cells in elevation_map
    - supplies is a non-negative integer

    >>> e_map = [[1, 6, 5, 6], [2, 5, 6, 8], [7, 2, 8, 1], [4, 4, 7, 3]]
    >>> can_hike_to(e_map, [3, 3], [2, 2], 10)
    True
    >>> can_hike_to(e_map, [3, 3], [2, 2], 8)
    False
    >>> can_hike_to(e_map, [3, 3], [3, 0], 7)
    True
    >>> can_hike_to(e_map, [3, 3], [3, 0], 6)
    False
    >>> can_hike_to(e_map, [3, 3], [0, 0], 18)
    True
    >>> can_hike_to(e_map, [3, 3], [0, 0], 17)
    False

    """

    current, dim = start, len(elevation_map)
    delta_north, delta_west, north, west = 0, 0, current, current

    while supplies >= 0 and current != dest:
        if current[0] == dest[0]:
            north = [dim + 1, dim + 1]
        else:
            north = [current[0] - 1, current[1]]
            delta_north = get_delta(elevation_map, current, north)
        if current[1] == dest[1]:
            west = [dim + 1, dim + 1]
        else:
            west = [current[0], current[1] - 1]
            delta_west = get_delta(elevation_map, current, west)
        if not is_valid_cell(north, dim) and not is_valid_cell(west, dim):
            return False
        delta = get_valid_delta([north, west], [delta_north, delta_west],
                                dim, current)
        supplies -= abs(delta)
    return current == dest and supplies >= 0


def get_lower_resolution(elevation_map: list[list[int]]) -> list[list[int]]:
    """
    Return elevation_map as a compressed map with each 2x2 entry as a single
    elevation point by taking the average of the original elevations. The
    average is always rounded down.

    If the dimension of elevation_map is odd, take the average of the remaining
    cells. For example, if there are two cells remaining, take the average of
    the two elevations. If one cell is remaining, take the average of the one
    elevation.

    >>> low_res_map = [[7, 9, 1], [4, 2, 1], [3, 2, 3]]
    >>> get_lower_resolution(low_res_map)
    [[5, 1], [2, 3]]
    >>> low_res_map = [[1, 6, 5, 6], [2, 5, 6, 8], [7, 2, 8, 1], [4, 4, 7, 3]]
    >>> get_lower_resolution(low_res_map)
    [[3, 6], [4, 4]]

    """

    new_list = []
    temp_list = []

    add_zeroes(elevation_map)

    for i in range(0, len(elevation_map), 2):
        for j in range(0, len(elevation_map), 2):
            temp_list.append(get_low_res_avg(elevation_map, i, j))
            if len(temp_list) == len(elevation_map) / 2:
                new_list.append(temp_list)
                temp_list = []

    return new_list


"""SUGGESTED HELPER FUNCTIONS
These functions are not required in the assignment. However, we believe it is
a great idea to define these functions and use them as helpers in the
required functions."""


def get_valid_delta(cells: list[list[int]], deltas: list[int],
                    dimension: int, valid_cell: list[int]) -> int:
    """
    Return delta1 if and only if the 0th cell of cells is a valid cell of
    elevation map with dimension x dimension. Return delta2 if and only if
    the 1st cell of cells is a valid cell of elevation map with dimension x
    dimension.

    Preconditions:
    - at least one cell in cells is valid for elevation map of
    dimension x dimension
    - all delta values in deltas are valid

    >>> list = [[1, 2], [3, 4]]
    >>> get_valid_delta(list, [2, -4], 8, [2, 3])
    -4
    >>> list = [[19, 7], [9, 10]]
    >>> get_valid_delta(list, [9, 10], 20, [4, 3])
    9

    """

    cell1, cell2, delta1, delta2 = cells[0], cells[1], deltas[0], deltas[1]

    if is_valid_cell(cell1, dimension) and is_valid_cell(cell2, dimension):
        is_cell1_valid = (delta1 <= delta2)
    else:
        is_cell1_valid = is_valid_cell(cell1, dimension)

    if is_cell1_valid:
        valid_cell[0] = cell1[0]
        valid_cell[1] = cell1[1]
        return delta1

    valid_cell[0] = cell2[0]
    valid_cell[1] = cell2[1]
    return delta2


def add_zeroes(elevation_map: list) -> None:
    """
    If the dimension of elevation_map map is odd, modify elevation_map so
    there is an extra row of 0's and an extra column of 0's

    >>> my_list = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    >>> add_zeroes(my_list)
    >>> my_list
    [[1, 2, 3, 0], [4, 5, 6, 0], [7, 8, 9, 0], [0, 0, 0, 0]]
    >>> my_list = [[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10], [2, 3, 4, 1]]
    >>> add_zeroes(my_list)
    >>> my_list
    [[1, 2, 3, 4], [4, 5, 6, 8], [7, 8, 9, 10], [2, 3, 4, 1]]

    """

    row_of_zeroes = []
    counter = 0

    if len(elevation_map) % 2 != 0:
        for sublist in elevation_map:
            sublist.append(0)

        while counter <= len(elevation_map):
            row_of_zeroes.append(0)
            counter += 1

        elevation_map.append(row_of_zeroes)


def get_low_res_avg(elevation_map: list[list[int]], i: int, j: int) -> int:
    """
    Return the average of the elevation_map with dimensions 2x2 in
    rows [i, i + 1] and columns [j, j + 1]

    Precondition: i and j are valid indices for elevation_map

    >>> get_low_res_avg([[2, 3], [4, 5]], 0, 0)
    3
    >>> get_low_res_avg([[5, 8], [9, 10]], 1, 1)
    10

    """

    valid_cell_count = 0
    total = 0

    for row in range(i, i + 2):
        for col in range(j, j + 2):
            elevation = get_elevation(elevation_map, [row, col])
            total += elevation
            if elevation > 0:
                valid_cell_count += 1

    return int(total / valid_cell_count)


def is_valid_elev(value: int) -> bool:
    """
    Return True if and only if value is greater than INVALID_ELEVATION.

    >>> is_valid_elev(5)
    True
    >>> is_valid_elev(0)
    False

    """
    return value > INVALID_ELEVATION


def get_elevation(elevation_map: list[list[int]], cell: list[int]) -> int:
    """
    Return the elevation of cell in elevation_map map. If cell is not valid in
    elevation_map, return INALID_ELEVATION.

    >>> get_elevation(FOUR_BY_FOUR, [0, 0])
    1
    >>> get_elevation(FOUR_BY_FOUR, [3, 3])
    4
    >>> get_elevation(FOUR_BY_FOUR, [4, 0])
    0
    >>> get_elevation(FOUR_BY_FOUR, [4, 4])
    0

    """

    if not is_valid_cell(cell, len(elevation_map)):
        return INVALID_ELEVATION

    return elevation_map[cell[0]][cell[1]]


def get_delta(elevation_map: list[list[int]],
              cell1: list[int],
              cell2: list[int]) -> int:
    """
    Return the change in elevation from cell1 to cell2 in elevation_map.
    If cell1 or cell2 is invalid for elevation_map, return None.

    >>> list = [[1, 2, 1], [4, 6, 5], [7, 8, 9]]
    >>> get_delta(list, [0, 0], [2, 2])
    8
    >>> list = [[1, 2, 6, 5], [4, 5, 3, 2], [7, 9, 8, 1], [1, 2, 1, 4]]
    >>> get_delta(list, [0, 0], [0, 2])
    5
    >>> list = [[1, 2, 6, 5], [4, 5, 3, 2], [7, 9, 8, 1], [1, 2, 1, 4]]
    >>> get_delta(list, [4, 0], [2, 2])

    """

    elevation_2 = get_elevation(elevation_map, cell2)
    elevation_1 = get_elevation(elevation_map, cell1)

    if not is_valid_elev(elevation_2) or not is_valid_elev(elevation_1):
        return None

    return elevation_2 - elevation_1


def is_valid_cell(cell: list[int], dimension: int) -> bool:
    """Return True if and only if cell is a valid cell in an elevation map
    of dimensions dimension x dimension.

    Precondition: cell is a list of length 2.

    >>> is_valid_cell([1, 1], 2)
    True
    >>> is_valid_cell([0, 2], 2)
    False
    >>> is_valid_cell([-1, 1], 2)
    False

    """

    return 0 <= cell[0] < dimension and 0 <= cell[1] < dimension


def is_cell_lower(elevation_map: list[list[int]], cell_1: list[int],
                  cell_2: list[int]) -> bool:
    """Return True iff cell_1 has a lower elevation than cell_2.

    Precondition: cell_1 and cell_2 are valid cells in elevation_map

    >>> map = [[0, 1], [2, 3]]
    >>> is_cell_lower(map, [0, 0], [1, 1])
    True
    >>> is_cell_lower(map, [1, 1], [0, 0])
    False
    >>> is_cell_lower(map, [1, 1], [1, 1])
    False
    >>> map = [[1, 2], [1, 4]]
    >>> is_cell_lower(map, [0, 0], [1, 0])
    False

    """

    elevation_1 = get_elevation(elevation_map, cell_1)
    elevation_2 = get_elevation(elevation_map, cell_2)

    return elevation_1 < elevation_2


def get_adjacent_cells(cell: list[int], dimension: int) -> list[list[int]]:
    """Return a list of cells adjacent to cell in an elevation map with
    dimensions dimension x dimension.

    Precondition: cell is a valid cell for an elevation map with
                  dimensions dimension x dimension.

    >>> adjacent_cells = get_adjacent_cells([1, 1], 3)
    >>> adjacent_cells.sort()
    >>> adjacent_cells
    [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2], [2, 0], [2, 1], [2, 2]]
    >>> adjacent_cells = get_adjacent_cells([1, 0], 3)
    >>> adjacent_cells.sort()
    >>> adjacent_cells
    [[0, 0], [0, 1], [1, 1], [2, 0], [2, 1]]

    """

    i = cell[0]
    j = cell[1]
    adjacent_cells = []

    for row in range(i - 1, i + 2):
        for col in range(j - 1, j + 2):
            if row != i or col != j:
                if is_valid_cell([row, col], dimension):
                    adjacent_cells.append([row, col])

    return adjacent_cells


if __name__ == '__main__':
    import doctest
    doctest.testmod()

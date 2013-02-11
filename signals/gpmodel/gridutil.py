import numpy as np

"""
n-dimension grid
"""


def grid2list(grid, n):
    grid = np.array(grid)
    truedim = grid.shape
    l = np.product(truedim[:n])  # length of the list
    list = np.zeros(np.append(l, truedim[n:]))
    grid2list.index = 0

    def iterate(addr):
        if len(addr) == n:
            list[grid2list.index] = grid[tuple(addr)]
            grid2list.index += 1
        else:
            for i in range(len(grid[tuple(addr)])):
                iterate(np.append(addr, i))

    iterate(np.array([]))
    return list


def list2grid(list, dim):
    list = np.array(list)
    dim = np.array(dim)

    if len(list) != np.product(dim):
        raise Exception("Dimension of grid must match the size of the list.")

    grid = np.zeros(np.append(dim, list.shape[1:]))
    list2grid.index = 0

    def iterate(addr):
        if len(addr) == len(dim):
            grid[tuple(addr)] = list[list2grid.index]
            list2grid.index += 1
        else:
            for i in range(len(grid[tuple(addr)])):
                iterate(np.append(addr, i))

    iterate(np.array([]))
    return grid


def makegrid(init, end, dim=None, res=101):
    init = np.array(init)
    end = np.array(end)

    d = len(init)
    if dim == None:
        dim = np.ones(d) * res

    factors = (end - init) / float(res - 1)
    grid = np.zeros(np.append(dim, len(dim)))

    def iterate(addr):
        if len(addr) == len(dim):
            grid[tuple(addr)] = init + addr * factors
        else:
            for i in range(len(grid[tuple(addr)])):
                iterate(np.append(addr, i))

    iterate(np.array([]))
    return grid

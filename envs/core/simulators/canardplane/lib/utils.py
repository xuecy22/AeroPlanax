import jax.numpy as jnp
from jax import jit


@jit
def linear_interp(grid_x, values, point):
    """
    一维线性插值函数

    参数:
    - grid_x: 网格坐标（升序排列），形状为 (nx,)
    - values: 网格点的值，形状为 (nx,)
    - point: 插值点的坐标，标量

    返回:
    - 插值后的值，标量
    """
    x = point
    # 找到插值点所在的索引
    ix = jnp.searchsorted(grid_x, x) - 1

    # 确保索引在有效范围内
    ix = jnp.clip(ix, 0, len(grid_x) - 2)

    # 获取两个邻近点的值
    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = values[ix]
    y1 = values[ix + 1]

    # 计算插值权重
    t = (x - x0) / (x1 - x0)

    # 执行线性插值
    y = y0 * (1 - t) + y1 * t

    return y

@jit
def fourd_linear_interp(grid_x, grid_y, grid_z, grid_w, values, point):
    """
    四维线性插值函数

    参数:
    - grid_x, grid_y, grid_z, grid_w: 四个维度的网格坐标（升序排列），形状分别为 (nx,), (ny,), (nz,), (nw,)
    - values: 网格点的值，形状为 (nx, ny, nz, nw)
    - point: 插值点的坐标，形状为 (4,)

    返回:
    - 插值后的值，形状为标量
    """
    x, y, z, w = point

    # 找到每个插值点所在的索引
    ix = jnp.searchsorted(grid_x, x) - 1
    iy = jnp.searchsorted(grid_y, y) - 1
    iz = jnp.searchsorted(grid_z, z) - 1
    iw = jnp.searchsorted(grid_w, w) - 1

    # 确保索引在有效范围内
    ix = jnp.clip(ix, 0, len(grid_x) - 2)
    iy = jnp.clip(iy, 0, len(grid_y) - 2)
    iz = jnp.clip(iz, 0, len(grid_z) - 2)
    iw = jnp.clip(iw, 0, len(grid_w) - 2)

    # 获取超立方体的16个顶点的值
    c0000 = values[ix    , iy    , iz    , iw    ]
    c1000 = values[ix + 1, iy    , iz    , iw    ]
    c0100 = values[ix    , iy + 1, iz    , iw    ]
    c1100 = values[ix + 1, iy + 1, iz    , iw    ]
    c0010 = values[ix    , iy    , iz + 1, iw    ]
    c1010 = values[ix + 1, iy    , iz + 1, iw    ]
    c0110 = values[ix    , iy + 1, iz + 1, iw    ]
    c1110 = values[ix + 1, iy + 1, iz + 1, iw    ]
    c0001 = values[ix    , iy    , iz    , iw + 1]
    c1001 = values[ix + 1, iy    , iz    , iw + 1]
    c0101 = values[ix    , iy + 1, iz    , iw + 1]
    c1101 = values[ix + 1, iy + 1, iz    , iw + 1]
    c0011 = values[ix    , iy    , iz + 1, iw + 1]
    c1011 = values[ix + 1, iy    , iz + 1, iw + 1]
    c0111 = values[ix    , iy + 1, iz + 1, iw + 1]
    c1111 = values[ix + 1, iy + 1, iz + 1, iw + 1]

    # 计算插值权重
    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = grid_y[iy]
    y1 = grid_y[iy + 1]
    z0 = grid_z[iz]
    z1 = grid_z[iz + 1]
    w0 = grid_w[iw]
    w1 = grid_w[iw + 1]

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)
    wd = (w - w0) / (w1 - w0)

    # 对 x 和 y 维度插值
    c00 = c0000 * (1 - xd) + c1000 * xd
    c01 = c0001 * (1 - xd) + c1001 * xd
    c10 = c0100 * (1 - xd) + c1100 * xd
    c11 = c0101 * (1 - xd) + c1101 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    # 对 z 和 w 维度插值
    c = c0 * (1 - zd) + c1 * zd

    result = c * (1 - wd) + c * wd
    return result

@jit
def trilinear_interp(grid_x, grid_y, grid_z, values, point):
    """
    三维线性插值函数

    参数:
    - grid_x, grid_y, grid_z: 三个维度的网格坐标（升序排列），形状分别为 (nx,), (ny,), (nz,)
    - values: 网格点的值，形状为 (nx, ny, nz)
    - points: 插值点的坐标，形状为 (3,)

    返回:
    - 插值后的值，形状为标量
    """
    x, y, z = point
    # 找到每个插值点所在的索引
    ix = jnp.searchsorted(grid_x, x) - 1
    iy = jnp.searchsorted(grid_y, y) - 1
    iz = jnp.searchsorted(grid_z, z) - 1

    # 确保索引在有效范围内
    ix = jnp.clip(ix, 0, len(grid_x) - 2)
    iy = jnp.clip(iy, 0, len(grid_y) - 2)
    iz = jnp.clip(iz, 0, len(grid_z) - 2)

    # 获取立方体的八个顶点的值
    c000 = values[ix    , iy    , iz    ]
    c100 = values[ix + 1, iy    , iz    ]
    c010 = values[ix    , iy + 1, iz    ]
    c110 = values[ix + 1, iy + 1, iz    ]
    c001 = values[ix    , iy    , iz + 1]
    c101 = values[ix + 1, iy    , iz + 1]
    c011 = values[ix    , iy + 1, iz + 1]
    c111 = values[ix + 1, iy + 1, iz + 1]

    # 计算插值权重
    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = grid_y[iy]
    y1 = grid_y[iy + 1]
    z0 = grid_z[iz]
    z1 = grid_z[iz + 1]

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)
    zd = (z - z0) / (z1 - z0)

    # 插值
    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd

    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    c = c0 * (1 - zd) + c1 * zd

    return c
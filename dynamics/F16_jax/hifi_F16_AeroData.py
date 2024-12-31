import os
import numpy as np
import jax.numpy as jnp
from jax import jit


path = os.path.dirname(os.path.realpath(__file__))

def safe_read_dat(dat_name):
    """
    安全读取数据文件，返回jnp数组。
    """
    data_path = os.path.join(path + '/data', dat_name)
    try:
        data = np.loadtxt(data_path)
        return data
    except OSError:
        print(f"Cannot find file {data_path} in current directory")
        return np.array([])


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


@jit
def bilinear_interp(grid_x, grid_y, values, point):
    """
    二维线性插值函数

    参数:
    - grid_x, grid_y: 两个维度的网格坐标（升序排列），形状分别为 (nx,), (ny,)
    - values: 网格点的值，形状为 (nx, ny)
    - point: 插值点的坐标，形状为 (2,)

    返回:
    - 插值后的值，标量
    """
    x, y = point
    # 找到每个插值点所在的索引
    ix = jnp.searchsorted(grid_x, x) - 1
    iy = jnp.searchsorted(grid_y, y) - 1

    # 确保索引在有效范围内
    ix = jnp.clip(ix, 0, len(grid_x) - 2)
    iy = jnp.clip(iy, 0, len(grid_y) - 2)

    # 获取四个顶点的值
    c00 = values[ix    , iy    ]
    c10 = values[ix + 1, iy    ]
    c01 = values[ix    , iy + 1]
    c11 = values[ix + 1, iy + 1]

    # 计算插值权重
    x0 = grid_x[ix]
    x1 = grid_x[ix + 1]
    y0 = grid_y[iy]
    y1 = grid_y[iy + 1]

    xd = (x - x0) / (x1 - x0)
    yd = (y - y0) / (y1 - y0)

    # 插值
    c0 = c00 * (1 - xd) + c10 * xd
    c1 = c01 * (1 - xd) + c11 * xd

    c = c0 * (1 - yd) + c1 * yd

    return c


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


ALPHA1 = safe_read_dat(r'ALPHA1.dat')
ALPHA2 = safe_read_dat(r'ALPHA2.dat')
BETA1 = safe_read_dat(r'BETA1.dat')
DH1 = safe_read_dat(r'DH1.dat')
DH2 = safe_read_dat(r'DH2.dat')

Cx = safe_read_dat(r'CX0120_ALPHA1_BETA1_DH1_201.dat')
Cx = Cx.reshape((DH1.shape[0], BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cx(point):
    DH1_jnp = jnp.array(DH1)
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cx_jnp = jnp.array(Cx)
    return trilinear_interp(DH1_jnp, BETA1_jnp, ALPHA1_jnp, Cx_jnp, point)

Cz = safe_read_dat(r'CZ0120_ALPHA1_BETA1_DH1_301.dat')
Cz = Cz.reshape((DH1.shape[0], BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cz(point):
    DH1_jnp = jnp.array(DH1)
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cz_jnp = jnp.array(Cz)
    return trilinear_interp(DH1_jnp, BETA1_jnp, ALPHA1_jnp, Cz_jnp, point)

Cm = safe_read_dat(r'CM0120_ALPHA1_BETA1_DH1_101.dat')
Cm = Cm.reshape((DH1.shape[0], BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cm(point):
    DH1_jnp = jnp.array(DH1)
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cm_jnp = jnp.array(Cm)
    return trilinear_interp(DH1_jnp, BETA1_jnp, ALPHA1_jnp, Cm_jnp, point)

Cy = safe_read_dat(r'CY0320_ALPHA1_BETA1_401.dat')
Cy = Cy.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cy(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cy_jnp = jnp.array(Cy)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cy_jnp, point)

Cn = safe_read_dat(r'CN0120_ALPHA1_BETA1_DH2_501.dat')
Cn = Cn.reshape((DH2.shape[0], BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cn(point):
    DH2_jnp = jnp.array(DH2)
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cn_jnp = jnp.array(Cn)
    return trilinear_interp(DH2_jnp, BETA1_jnp, ALPHA1_jnp, Cn_jnp, point)

Cl = safe_read_dat(r'CL0120_ALPHA1_BETA1_DH2_601.dat')
Cl = Cl.reshape((DH2.shape[0], BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cl(point):
    DH2_jnp = jnp.array(DH2)
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cl_jnp = jnp.array(Cl)
    return trilinear_interp(DH2_jnp, BETA1_jnp, ALPHA1_jnp, Cl_jnp, point)

Cx_lef = safe_read_dat(r'CX0820_ALPHA2_BETA1_202.dat')
Cx_lef = Cx_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cx_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cx_lef_jnp = jnp.array(Cx_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cx_lef_jnp, point)

Cz_lef = safe_read_dat(r'CZ0820_ALPHA2_BETA1_302.dat')
Cz_lef = Cz_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cz_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cz_lef_jnp = jnp.array(Cz_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cz_lef_jnp, point)

Cm_lef = safe_read_dat(r'CM0820_ALPHA2_BETA1_102.dat')
Cm_lef = Cm_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cm_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cm_lef_jnp = jnp.array(Cm_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cm_lef_jnp, point)

Cy_lef = safe_read_dat(r'CY0820_ALPHA2_BETA1_402.dat')
Cy_lef = Cy_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cy_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cy_lef_jnp = jnp.array(Cy_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cy_lef_jnp, point)

Cn_lef = safe_read_dat(r'CN0820_ALPHA2_BETA1_502.dat')
Cn_lef = Cn_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cn_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cn_lef_jnp = jnp.array(Cn_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cn_lef_jnp, point)

Cl_lef = safe_read_dat(r'CL0820_ALPHA2_BETA1_602.dat')
Cl_lef = Cl_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cl_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cl_lef_jnp = jnp.array(Cl_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cl_lef_jnp, point)

CXq = safe_read_dat(r'CX1120_ALPHA1_204.dat')
@jit
def _CXq(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CXq_jnp = jnp.array(CXq)
    return linear_interp(ALPHA1_jnp, CXq_jnp, point)

CZq = safe_read_dat(r'CZ1120_ALPHA1_304.dat')
@jit
def _CZq(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CZq_jnp = jnp.array(CZq)
    return linear_interp(ALPHA1_jnp, CZq_jnp, point)

CMq = safe_read_dat(r'CM1120_ALPHA1_104.dat')
@jit
def _CMq(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CMq_jnp = jnp.array(CMq)
    return linear_interp(ALPHA1_jnp, CMq_jnp, point)

CYp = safe_read_dat(r'CY1220_ALPHA1_408.dat')
@jit
def _CYp(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CYp_jnp = jnp.array(CYp)
    return linear_interp(ALPHA1_jnp, CYp_jnp, point)

CYr = safe_read_dat(r'CY1320_ALPHA1_406.dat')
@jit
def _CYr(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CYr_jnp = jnp.array(CYr)
    return linear_interp(ALPHA1_jnp, CYr_jnp, point)

CNr = safe_read_dat(r'CN1320_ALPHA1_506.dat')
@jit
def _CNr(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CNr_jnp = jnp.array(CNr)
    return linear_interp(ALPHA1_jnp, CNr_jnp, point)

CNp = safe_read_dat(r'CN1220_ALPHA1_508.dat')
@jit
def _CNp(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CNp_jnp = jnp.array(CNp)
    return linear_interp(ALPHA1_jnp, CNp_jnp, point)

CLp = safe_read_dat(r'CL1220_ALPHA1_608.dat')
@jit
def _CLp(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CLp_jnp = jnp.array(CLp)
    return linear_interp(ALPHA1_jnp, CLp_jnp, point)

CLr = safe_read_dat(r'CL1320_ALPHA1_606.dat')
@jit
def _CLr(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    CLr_jnp = jnp.array(CLr)
    return linear_interp(ALPHA1_jnp, CLr_jnp, point)

delta_CXq_lef = safe_read_dat(r'CX1420_ALPHA2_205.dat')
@jit
def _delta_CXq_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CXq_lef_jnp = jnp.array(delta_CXq_lef)
    return linear_interp(ALPHA2_jnp, delta_CXq_lef_jnp, point)

delta_CYr_lef = safe_read_dat(r'CY1620_ALPHA2_407.dat')
@jit
def _delta_CYr_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CYr_lef_jnp = jnp.array(delta_CYr_lef)
    return linear_interp(ALPHA2_jnp, delta_CYr_lef_jnp, point)

delta_CYp_lef = safe_read_dat(r'CY1520_ALPHA2_409.dat')
@jit
def _delta_CYp_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CYp_lef_jnp = jnp.array(delta_CYp_lef)
    return linear_interp(ALPHA2_jnp, delta_CYp_lef_jnp, point)

delta_CZq_lef = safe_read_dat(r'CZ1420_ALPHA2_305.dat')
@jit
def _delta_CZq_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CZq_lef_jnp = jnp.array(delta_CZq_lef)
    return linear_interp(ALPHA2_jnp, delta_CZq_lef_jnp, point)

delta_CLr_lef = safe_read_dat(r'CL1620_ALPHA2_607.dat')
@jit
def _delta_CLr_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CLr_lef_jnp = jnp.array(delta_CLr_lef)
    return linear_interp(ALPHA2_jnp, delta_CLr_lef_jnp, point)

delta_CLp_lef = safe_read_dat(r'CL1520_ALPHA2_609.dat')
@jit
def _delta_CLp_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CLp_lef_jnp = jnp.array(delta_CLp_lef)
    return linear_interp(ALPHA2_jnp, delta_CLp_lef_jnp, point)

delta_CMq_lef = safe_read_dat(r'CM1420_ALPHA2_105.dat')
@jit
def _delta_CMq_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CMq_lef_jnp = jnp.array(delta_CMq_lef)
    return linear_interp(ALPHA2_jnp, delta_CMq_lef_jnp, point)

delta_CNr_lef = safe_read_dat(r'CN1620_ALPHA2_507.dat')
@jit
def _delta_CNr_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CNr_lef_jnp = jnp.array(delta_CNr_lef)
    return linear_interp(ALPHA2_jnp, delta_CNr_lef_jnp, point)

delta_CNp_lef = safe_read_dat(r'CN1520_ALPHA2_509.dat')
@jit
def _delta_CNp_lef(point):
    ALPHA2_jnp = jnp.array(ALPHA2)
    delta_CNp_lef_jnp = jnp.array(delta_CNp_lef)
    return linear_interp(ALPHA2_jnp, delta_CNp_lef_jnp, point)

Cy_r30 = safe_read_dat(r'CY0720_ALPHA1_BETA1_405.dat')
Cy_r30 = Cy_r30.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cy_r30(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cy_r30_jnp = jnp.array(Cy_r30)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cy_r30_jnp, point)


Cn_r30 = safe_read_dat(r'CN0720_ALPHA1_BETA1_503.dat')
Cn_r30 = Cn_r30.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cn_r30(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cn_r30_jnp = jnp.array(Cn_r30)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cn_r30_jnp, point)

Cl_r30 = safe_read_dat(r'CL0720_ALPHA1_BETA1_603.dat')
Cl_r30 = Cl_r30.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cl_r30(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cl_r30_jnp = jnp.array(Cl_r30)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cl_r30_jnp, point)

Cy_a20 = safe_read_dat(r'CY0620_ALPHA1_BETA1_403.dat')
Cy_a20 = Cy_a20.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cy_a20(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cy_a20_jnp = jnp.array(Cy_a20)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cy_a20_jnp, point)

Cy_a20_lef = safe_read_dat(r'CY0920_ALPHA2_BETA1_404.dat')
Cy_a20_lef = Cy_a20_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cy_a20_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cy_a20_lef_jnp = jnp.array(Cy_a20_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cy_a20_lef_jnp, point)

Cn_a20 = safe_read_dat(r'CN0620_ALPHA1_BETA1_504.dat')
Cn_a20 = Cn_a20.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cn_a20(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cn_a20_jnp = jnp.array(Cn_a20)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cn_a20_jnp, point)

Cn_a20_lef = safe_read_dat(r'CN0920_ALPHA2_BETA1_505.dat')
Cn_a20_lef = Cn_a20_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cn_a20_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cn_a20_lef_jnp = jnp.array(Cn_a20_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cn_a20_lef_jnp, point)

Cl_a20 = safe_read_dat(r'CL0620_ALPHA1_BETA1_604.dat')
Cl_a20 = Cl_a20.reshape((BETA1.shape[0], ALPHA1.shape[0]))
@jit
def _Cl_a20(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA1_jnp = jnp.array(ALPHA1)
    Cl_a20_jnp = jnp.array(Cl_a20)
    return bilinear_interp(BETA1_jnp, ALPHA1_jnp, Cl_a20_jnp, point)

Cl_a20_lef = safe_read_dat(r'CL0920_ALPHA2_BETA1_605.dat')
Cl_a20_lef = Cl_a20_lef.reshape((BETA1.shape[0], ALPHA2.shape[0]))
@jit
def _Cl_a20_lef(point):
    BETA1_jnp = jnp.array(BETA1)
    ALPHA2_jnp = jnp.array(ALPHA2)
    Cl_a20_lef_jnp = jnp.array(Cl_a20_lef)
    return bilinear_interp(BETA1_jnp, ALPHA2_jnp, Cl_a20_lef_jnp, point)

delta_CNbeta = safe_read_dat(r'CN9999_ALPHA1_brett.dat')
@jit
def _delta_CNbeta(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    delta_CNbeta_jnp = jnp.array(delta_CNbeta)
    return linear_interp(ALPHA1_jnp, delta_CNbeta_jnp, point)

delta_CLbeta = safe_read_dat(r'CL9999_ALPHA1_brett.dat')
@jit
def _delta_CLbeta(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    delta_CLbeta_jnp = jnp.array(delta_CLbeta)
    return linear_interp(ALPHA1_jnp, delta_CLbeta_jnp, point)

delta_Cm = safe_read_dat(r'CM9999_ALPHA1_brett.dat')
@jit
def _delta_Cm(point):
    ALPHA1_jnp = jnp.array(ALPHA1)
    delta_Cm_jnp = jnp.array(delta_Cm)
    return linear_interp(ALPHA1_jnp, delta_Cm_jnp, point)

eta_el = safe_read_dat(r'ETA_DH1_brett.dat')
@jit
def _eta_el(point):
    DH1_jnp = jnp.array(DH1)
    eta_el_jnp = jnp.array(eta_el)
    return linear_interp(DH1_jnp, eta_el_jnp, point)

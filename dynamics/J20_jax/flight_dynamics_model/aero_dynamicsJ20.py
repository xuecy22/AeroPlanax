import os
import jax
import jax.numpy as jnp
from jax import jit
import scipy.io as scio
from scipy.interpolate import RegularGridInterpolator
from . import plane_params
from .plane_params import J20PlaneParams
from ..lib.attitude import attitude as att


curr_path = os.path.dirname(__file__)
data_path = os.path.join(curr_path, "coefJ20.mat")
coefJ20 = scio.loadmat(data_path)

a1 = jnp.array(coefJ20['staticCoefJ20']['alpha1'][0][0][0])
b1 = jnp.array(coefJ20['staticCoefJ20']['beta1'][0][0][0])
b2 = jnp.array(coefJ20['staticCoefJ20']['beta2'][0][0][0])
crnd = jnp.array(coefJ20['staticCoefJ20']['crnd'][0][0][0])
ael = jnp.array(coefJ20['staticCoefJ20']['ael'][0][0][0])
aer = jnp.array(coefJ20['staticCoefJ20']['aer'][0][0][0])
rd = jnp.array(coefJ20['staticCoefJ20']['rud'][0][0][0])

CD_EA = jnp.array(coefJ20['staticCoefJ20']['CD_EA'][0][0])
CL_EA = jnp.array(coefJ20['staticCoefJ20']['CL_EA'][0][0])
CC_EA = jnp.array(coefJ20['staticCoefJ20']['CC_EA'][0][0])
Cm_EA = jnp.array(coefJ20['staticCoefJ20']['Cm_EA'][0][0])
Cl_EA = jnp.array(coefJ20['staticCoefJ20']['Cl_EA'][0][0])
Cn_EA = jnp.array(coefJ20['staticCoefJ20']['Cn_EA'][0][0])

CD_VT = jnp.array(coefJ20['staticCoefJ20']['CD_VT'][0][0])
CL_VT = jnp.array(coefJ20['staticCoefJ20']['CL_VT'][0][0])
CC_VT = jnp.array(coefJ20['staticCoefJ20']['CC_VT'][0][0])
Cm_VT = jnp.array(coefJ20['staticCoefJ20']['Cm_VT'][0][0])
Cl_VT = jnp.array(coefJ20['staticCoefJ20']['Cl_VT'][0][0])
Cn_VT = jnp.array(coefJ20['staticCoefJ20']['Cn_VT'][0][0])

CD_CR = jnp.array(coefJ20['staticCoefJ20']['CD_CR'][0][0])
CL_CR = jnp.array(coefJ20['staticCoefJ20']['CL_CR'][0][0])
CC_CR = jnp.array(coefJ20['staticCoefJ20']['CC_CR'][0][0])
Cm_CR = jnp.array(coefJ20['staticCoefJ20']['Cm_CR'][0][0])
Cl_CR = jnp.array(coefJ20['staticCoefJ20']['Cl_CR'][0][0])
Cn_CR = jnp.array(coefJ20['staticCoefJ20']['Cn_CR'][0][0])

CCp = coefJ20['aeroDerivativesJ20']['CCp'][0][0][0][0]
CCr = coefJ20['aeroDerivativesJ20']['CCr'][0][0][0][0]
CLq = coefJ20['aeroDerivativesJ20']['CLq'][0][0][0][0]
CLalphadot = coefJ20['aeroDerivativesJ20']['CLalphadot'][0][0][0][0]
Clp = coefJ20['aeroDerivativesJ20']['Clp'][0][0][0][0]
Clr = coefJ20['aeroDerivativesJ20']['Clr'][0][0][0][0]
Cmq = coefJ20['aeroDerivativesJ20']['Cmq'][0][0][0][0]
Cmalphadot = coefJ20['aeroDerivativesJ20']['Cmalphadot'][0][0][0][0]
Cnp = coefJ20['aeroDerivativesJ20']['Cnp'][0][0][0][0]
Cnr = coefJ20['aeroDerivativesJ20']['Cnr'][0][0][0][0]


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

@jit
def _CDae(point):
    return fourd_linear_interp(a1, b1, ael, aer, CD_EA, point)

@jit
def _CLae(point):
    return fourd_linear_interp(a1, b1, ael, aer, CL_EA, point)

@jit
def _CCae(point):
    return fourd_linear_interp(a1, b1, ael, aer, CC_EA, point)

@jit
def _Cmae(point):
    return fourd_linear_interp(a1, b1, ael, aer, Cm_EA, point)

@jit
def _Clae(point):
    return fourd_linear_interp(a1, b1, ael, aer, Cl_EA, point)

@jit
def _Cnae(point):
    return fourd_linear_interp(a1, b1, ael, aer, Cn_EA, point)

@jit
def _CDrd(point):
    return trilinear_interp(a1, b2, rd, CD_VT, point)

@jit
def _CLrd(point):
    return trilinear_interp(a1, b2, rd, CL_VT, point)

@jit
def _CCrd(point):
    return trilinear_interp(a1, b2, rd, CC_VT, point)
@jit
def _Cmrd(point):
    return trilinear_interp(a1, b2, rd, Cm_VT, point)

@jit
def _Clrd(point):
    return trilinear_interp(a1, b2, rd, Cl_VT, point)

@jit
def _Cnrd(point):
    return trilinear_interp(a1, b2, rd, Cn_VT, point)

@jit
def _CDcr(point):
    return trilinear_interp(a1, b2, crnd, CD_CR, point)

@jit
def _CLcr(point):
    return trilinear_interp(a1, b2, crnd, CL_CR, point)

@jit
def _CCcr(point):
    return trilinear_interp(a1, b2, crnd, CC_CR, point)

@jit
def _Cmcr(point):
    return trilinear_interp(a1, b2, crnd, Cm_CR, point)

@jit
def _Clcr(point):
    return trilinear_interp(a1, b2, crnd, Cl_CR, point)

@jit
def _Cncr(point):
    return trilinear_interp(a1, b2, crnd, Cn_CR, point)


@jit
def Static_Coefficient(alpha, beta, delta):
    """静导数

    Args:
        alpha (float): Unit in radians, range -30~30°
        beta (float): Unit in radians, range -30~30°
        delta (np.array (4,)): [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
            delta_LEA, range -15~15°
            delta_REA, range -15~15°
            delta_VT, range -6~6°
            delta_CR, range -6~6°

    Returns:
        CF (np.array (3,)): Static force coefficients [CD; CC; CL] in wind frame
        CM (np.array (3,)): Static momentum coefficients [Cl; Cm; Cn] in body frame
    """

    flagAE = 1
    flagBeta = 1
    ai = alpha * 180 / jnp.pi   # aoa in degrees
    bi = beta * 180 / jnp.pi    # sideslip angle in degrees
    # ---- TODO ----
    ai = jnp.clip(ai, -30, 30)
    bi = jnp.clip(bi, -30, 30)
    # ------

    aeli = delta[0]   # left elevon setting in degrees.
    aeri = delta[1]   # right elevon setting in degrees.
    rudi = delta[2]   # rudder setting in degrees.
    crdi = delta[3]   # carnard setting in degrees.

    aeli = jax.lax.select(aeli > aeri, aeri, aeli)
    aeri = jax.lax.select(aeli > aeri, aeli, aeri)
    bi = jax.lax.select(aeli > aeri, -bi, bi)
    rudi = jax.lax.select(aeli > aeri, -rudi, rudi)
    flagAE = jax.lax.select(aeli > aeri, -1, 1)
    biae = bi

    flagBeta = jax.lax.select(bi > 0, -1, 1)
    bi = jax.lax.select(bi > 0, -bi, bi)
    rudi = jax.lax.select(bi > 0, -rudi, rudi)

    CDae = _CDae((ai, biae, aeli, aeri))
    CLae = _CLae((ai, biae, aeli, aeri))
    CCae = _CCae((ai, biae, aeli, aeri))
    Cmae = _Cmae((ai, biae, aeli, aeri))
    Clae = _Clae((ai, biae, aeli, aeri))
    Cnae = _Cnae((ai, biae, aeli, aeri))
    CDrd = _CDrd((ai, bi, rudi))
    CLrd = _CLrd((ai, bi, rudi))
    CCrd = _CCrd((ai, bi, rudi))
    Cmrd = _Cmrd((ai, bi, rudi))
    Clrd = _Clrd((ai, bi, rudi))
    Cnrd = _Cnrd((ai, bi, rudi))
    CDcr = _CDcr((ai, bi, crdi))
    CLcr = _CLcr((ai, bi, crdi))
    CCcr = _CCcr((ai, bi, crdi))
    Cmcr = _Cmcr((ai, bi, crdi))
    Clcr = _Clcr((ai, bi, crdi))
    Cncr = _Cncr((ai, bi, crdi))

    CD = CDae + CDrd + CDcr
    CL = CLae + CLrd + CLcr
    CC = flagAE*(CCae + flagBeta*CCrd + flagBeta*CCcr)
    Cm = Cmae + Cmrd + Cmcr
    Cl = flagAE*(Clae + flagBeta*Clrd + flagBeta*Clcr)
    Cn = flagAE*(Cnae + flagBeta*Cnrd + flagBeta*Cncr)

    CF = jnp.array([CD, CC, CL]).reshape(-1)
    CM = jnp.array([Cl, Cm, Cn]).reshape(-1)
    return CF, CM

@jit
def Aero_Coefficient(alpha, beta, alphadot, VTAS, omega, delta, airframe: J20PlaneParams):
    """动导数

    Args:
        dynamics (_type_): fdm6dof
        delta (np.array (4,)): [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
            delta_LEA, range -15~15°
            delta_REA, range -15~15°
            delta_VT, range -6~6°
            delta_CR, range -6~6°
        airframe (_type_): PlaneParams

    Returns:
        CF (np.array (3,)): Static force coefficients [CD; CC; CL] in wind frame
        CM (np.array (3,)): Static momentum coefficients [Cl; Cm; Cn] in body frame
    """
    CF, CM = Static_Coefficient(alpha, beta, delta)
    p, q, r = omega
    CF = CF + 1.0 / (2 * VTAS + 1e-8)*jnp.array([0,
                                        (CCp*p +
                                        CCr*r)*plane_params.wingspan,
                                        (CLq*q + CLalphadot*alphadot)*plane_params.chord])

    CM = CM + 1.0/(2 * VTAS + 1e-8)*jnp.array([(Clp*p + Clr*r)*plane_params.wingspan,
                                        (Cmq*q +
                                        Cmalphadot*alphadot)*plane_params.chord,
                                        (Cnp*p + Cnr*r)*plane_params.wingspan])
    return CF, CM

@jit
def Aero_Forces_Torques(alpha, beta, alphadot, VTAS, q, omega, delta, airframe: J20PlaneParams):
    """_summary_

    Args:
        delta (np.array (4,)): Control surface deviation [delta_LEA;delta_REA;delta_VT;delta_CR], unit in degrees
            delta_LEA, range -15~15°
            delta_REA, range -15~15°
            delta_VT, range -6~6°
            delta_CR, range -6~6°
        airframe (_type_): PlaneParams
        q (float): Dynamic pressure, unit in Pascal

    Returns:
        _type_: _description_
    """
    CF, CM = Aero_Coefficient(alpha, beta, alphadot, VTAS, omega, delta, airframe)
    C_wind_body = att.aerodynamicAngle2DCM_Wind2Body(alpha, beta)
    Fb = C_wind_body @ (-q * plane_params.S_wing*CF)
    Mb = q*plane_params.S_wing*jnp.array([plane_params.wingspan, plane_params.chord, plane_params.wingspan])*CM
    # Compensate for CG offset
    Mb = Mb + jnp.cross(plane_params.emptyInertia.rCG-airframe.inertia.rCG, Fb)
    return Fb, Mb


# if __name__ == '__main__':
#     J20 = AeroDynamicsJ20()
#     print(J20.Static_Coefficient(np.radians(0), np.radians(0), [0, 0, 0, 0]))

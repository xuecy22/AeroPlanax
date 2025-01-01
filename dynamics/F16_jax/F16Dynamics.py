import jax.numpy as jnp
from jax import jit
from . import hifi_F16_AeroData as hifi_F16


@jit
def atmos(alt, vt):
    # 根据高度和速度计算动压、马赫数
    rho0 = 2.377e-3
    tfac = 1 - .703e-5 * (alt)
    temp = 519.0 * tfac
    temp = (alt >= 35000.0) * 390 + (alt < 35000.0) * temp
    rho = rho0 * jnp.pow(tfac, 4.14)
    mach = (vt) / jnp.sqrt(1.4 * 1716.3 * temp)
    qbar = .5 * rho * jnp.pow(vt, 2)
    ps = 1715.0 * rho * temp

    ps = (ps == 0) * 1715 + (ps != 0) * ps

    return (mach, qbar, ps)

@jit
def accels(roll, pitch, alpha, beta, vt, alpha_dot, beta_dot, vt_dot, P, Q, R):
    # 根据飞行状态结算三轴过载
    grav = 32.174
    sina = jnp.sin(alpha)
    cosa = jnp.cos(alpha)
    sinb = jnp.sin(beta)
    cosb = jnp.cos(beta)
    vel_u = vt * cosb * cosa
    vel_v = vt * sinb
    vel_w = vt * cosb * sina
    u_dot = cosb * cosa * vt_dot - vt * sinb * cosa * beta_dot - vt * cosb * sina * alpha_dot
    v_dot = sinb * vt_dot + vt * cosb * beta_dot
    w_dot = cosb * sina * vt_dot - vt * sinb * sina * beta_dot + vt * cosb * cosa * alpha_dot
    nx_cg = 1.0 / grav * (u_dot + Q * vel_w - R * vel_v) + jnp.sin(pitch)
    ny_cg = 1.0 / grav * (v_dot + R * vel_u - P * vel_w) - jnp.cos(pitch) * jnp.sin(roll)
    nz_cg = -1.0 / grav * (w_dot + P * vel_v - Q * vel_u) + jnp.cos(pitch) * jnp.cos(roll)
    return (nx_cg, ny_cg, nz_cg)

@jit
def nlplant(xu):
    xdot = jnp.zeros_like(xu)
    g = 32.17
    m = 636.94
    B = 30.0
    S = 300.0
    cbar = 11.32
    xcgr = 0.35
    xcg = 0.30
    Heng = 0.0
    pi = jnp.pi

    Jy = 55814.0
    Jxz = 982.0
    Jz = 63100.0
    Jx = 9496.0

    r2d = 180.0 / pi

    # States
    alt = xu[2]
    phi = xu[3]
    theta = xu[4]
    psi = xu[5]

    vt = xu[6]
    alpha = xu[7] * r2d
    beta = xu[8] * r2d
    P = xu[9]
    Q = xu[10]
    R = xu[11]

    sa = jnp.sin(xu[7])
    ca = jnp.cos(xu[7])
    sb = jnp.sin(xu[8])
    cb = jnp.cos(xu[8])

    st = jnp.sin(theta)
    ct = jnp.cos(theta)
    tt = jnp.tan(theta)
    sphi = jnp.sin(phi)
    cphi = jnp.cos(phi)
    spsi = jnp.sin(psi)
    cpsi = jnp.cos(psi)

    vt = (vt <= 0.01) * 0.01 + (vt > 0.01) * vt

    # Control inputs

    T = xu[12]
    el = xu[13]
    ail = xu[14]
    rud = xu[15]
    lef = xu[16]

    dail = ail / 21.5
    drud = rud / 30.0
    dlef = (1 - lef / 25.0)

    # Atmospheric effects
    # sets dynamic pressure and mach number

    temp = atmos(alt, vt)
    mach = temp[0]
    qbar = temp[1]
    ps = temp[2]

    # Dynamics
    # Navigation Equations

    U = vt * ca * cb
    V = vt * sb
    W = vt * sa * cb

    xdot = xdot.at[0].set(U * (ct * cpsi) + V * (sphi * cpsi * st - cphi * spsi) + W * (cphi * st * cpsi + sphi * spsi))
    xdot = xdot.at[1].set(U * (ct * spsi) + V * (sphi * spsi * st + cphi * cpsi) + W * (cphi * st * spsi - sphi * cpsi))
    xdot = xdot.at[2].set(U * st - V * (sphi * ct) - W * (cphi * ct))
    xdot = xdot.at[3].set(P + tt * (Q * sphi + R * cphi))
    xdot = xdot.at[4].set(Q * cphi - R * sphi)
    xdot = xdot.at[5].set((Q * sphi + R * cphi) / ct)

    Cx = hifi_F16._Cx((el, beta, alpha))
    Cz = hifi_F16._Cz((el, beta, alpha))
    Cm = hifi_F16._Cm((el, beta, alpha))
    Cy = hifi_F16._Cy((beta, alpha))
    Cn = hifi_F16._Cn((el, beta, alpha))
    Cl = hifi_F16._Cl((el, beta, alpha))

    Cxq = hifi_F16._CXq(alpha)
    Cyr = hifi_F16._CYr(alpha)
    Cyp = hifi_F16._CYp(alpha)
    Czq = hifi_F16._CZq(alpha)
    Clr = hifi_F16._CLr(alpha)
    Clp = hifi_F16._CLp(alpha)
    Cmq = hifi_F16._CMq(alpha)
    Cnr = hifi_F16._CNr(alpha)
    Cnp = hifi_F16._CNp(alpha)

    delta_Cx_lef = hifi_F16._Cx_lef((alpha, beta)) - hifi_F16._Cx((alpha, beta, 0))
    delta_Cz_lef = hifi_F16._Cz_lef((alpha, beta)) - hifi_F16._Cz((alpha, beta, 0))
    delta_Cm_lef = hifi_F16._Cm_lef((alpha, beta)) - hifi_F16._Cm((alpha, beta, 0))
    delta_Cy_lef = hifi_F16._Cy_lef((alpha, beta)) - hifi_F16._Cy((alpha, beta))
    delta_Cn_lef = hifi_F16._Cn_lef((alpha, beta)) - hifi_F16._Cn((alpha, beta, 0))
    delta_Cl_lef = hifi_F16._Cl_lef((alpha, beta)) - hifi_F16._Cl((alpha, beta, 0))

    delta_Cxq_lef = hifi_F16._delta_CXq_lef(alpha)
    delta_Cyr_lef = hifi_F16._delta_CYr_lef(alpha)
    delta_Cyp_lef = hifi_F16._delta_CYp_lef(alpha)
    # delta_Czq_lef = hifi_F16._delta_CZq_lef(alpha)
    delta_Clr_lef = hifi_F16._delta_CLr_lef(alpha)
    delta_Clp_lef = hifi_F16._delta_CLp_lef(alpha)
    delta_Cmq_lef = hifi_F16._delta_CMq_lef(alpha)
    delta_Cnr_lef = hifi_F16._delta_CNr_lef(alpha)
    delta_Cnp_lef = hifi_F16._delta_CNp_lef(alpha)

    delta_Cy_r30 = hifi_F16._Cy_r30((alpha, beta)) - hifi_F16._Cy((alpha, beta))
    delta_Cn_r30 = hifi_F16._Cn_r30((alpha, beta)) - hifi_F16._Cn((alpha, beta, 0))
    delta_Cl_r30 = hifi_F16._Cl_r30((alpha, beta)) - hifi_F16._Cl((alpha, beta, 0))

    delta_Cy_a20 = hifi_F16._Cy_a20((alpha, beta)) - hifi_F16._Cy((alpha, beta))
    delta_Cy_a20_lef = hifi_F16._Cy_a20_lef((alpha, beta)) - hifi_F16._Cy_lef((alpha, beta)) -\
        (hifi_F16._Cy_a20((alpha, beta)) - hifi_F16._Cy((alpha, beta)))
    delta_Cn_a20 = hifi_F16._Cn_a20((alpha, beta)) - hifi_F16._Cn((alpha, beta, 0))
    delta_Cn_a20_lef = hifi_F16._Cn_a20_lef((alpha, beta)) - hifi_F16._Cn_lef((alpha, beta)) -\
        (hifi_F16._Cn_a20((alpha, beta)) - hifi_F16._Cn((alpha, beta, 0)))
    delta_Cl_a20 = hifi_F16._Cl_a20((alpha, beta)) - hifi_F16._Cl((alpha, beta, 0))
    delta_Cl_a20_lef = hifi_F16._Cl_a20_lef((alpha, beta)) - hifi_F16._Cl_lef((alpha, beta)) -\
        (hifi_F16._Cl_a20((alpha, beta)) - hifi_F16._Cl((alpha, beta, 0)))

    delta_Cnbeta = hifi_F16._delta_CNbeta(alpha)
    delta_Clbeta = hifi_F16._delta_CLbeta(alpha)
    delta_Cm = hifi_F16._delta_Cm(alpha)
    eta_el = hifi_F16._eta_el(el)
    delta_Cm_ds = 0
    # compute Cx_tot, Cz_tot, Cm_tot, Cy_tot, Cn_tot, and Cl_tot
    # (as on NASA report p37-40)

    dXdQ = (cbar / (2 * vt)) * (Cxq + delta_Cxq_lef * dlef)
    Cx_tot = Cx + delta_Cx_lef * dlef + dXdQ * Q
    dZdQ = (cbar / (2 * vt)) * (Czq + delta_Cz_lef * dlef)
    Cz_tot = Cz + delta_Cz_lef * dlef + dZdQ * Q
    dMdQ = (cbar / (2 * vt)) * (Cmq + delta_Cmq_lef * dlef)
    Cm_tot = Cm * eta_el + Cz_tot * (xcgr - xcg) + delta_Cm_lef * dlef + dMdQ * Q + delta_Cm + delta_Cm_ds
    dYdail = delta_Cy_a20 + delta_Cy_a20_lef * dlef
    dYdR = (B / (2 * vt)) * (Cyr + delta_Cyr_lef * dlef)
    dYdP = (B / (2 * vt)) * (Cyp + delta_Cyp_lef * dlef)
    
    Cy_tot = Cy + delta_Cy_lef * dlef + dYdail * dail + delta_Cy_r30 * drud + dYdR * R + dYdP * P
    dNdail = delta_Cn_a20 + delta_Cn_a20_lef * dlef
    dNdR = (B / (2 * vt)) * (Cnr + delta_Cnr_lef * dlef)
    dNdP = (B / (2 * vt)) * (Cnp + delta_Cnp_lef * dlef)
    Cn_tot = Cn + delta_Cn_lef * dlef - Cy_tot * (xcgr - xcg) * (cbar / B) + dNdail * dail + delta_Cn_r30 * drud + dNdR * R + dNdP * P + delta_Cnbeta * beta
    dLdail = delta_Cl_a20 + delta_Cl_a20_lef * dlef
    dLdR = (B / (2 * vt)) * (Clr + delta_Clr_lef * dlef)
    dLdP = (B / (2 * vt)) * (Clp + delta_Clp_lef * dlef)
    Cl_tot = Cl + delta_Cl_lef * dlef + dLdail * dail + delta_Cl_r30 * drud + dLdR * R + dLdP * P + delta_Clbeta * beta
    Udot = R * V - Q * W - g * st + qbar * S * Cx_tot / m + T / m
    Vdot = P * W - R * U + g * ct * sphi + qbar * S * Cy_tot / m
    Wdot = Q * U - P * V + g * ct * cphi + qbar * S * Cz_tot / m
    xdot = xdot.at[6].set((U * Udot + V * Vdot + W * Wdot) / vt)
    xdot = xdot.at[7].set((U * Wdot - W * Udot) / (U * U + W * W))
    xdot = xdot.at[8].set((Vdot * vt - V * xdot[6]) / (vt * vt * cb))
    L_tot = Cl_tot * qbar * S * B
    M_tot = Cm_tot * qbar * S * cbar
    N_tot = Cn_tot * qbar * S * B
    denom = Jx * Jz - Jxz * Jxz
    xdot = xdot.at[9].set((Jz * L_tot + Jxz * N_tot - (Jz * (Jz - Jy) + Jxz * Jxz) * Q * R + Jxz * (Jx - Jy + Jz) * P * Q + Jxz * Q * Heng) / denom)
    xdot = xdot.at[10].set((M_tot + (Jz - Jx) * P * R - Jxz * (P * P - R * R) - R * Heng) / Jy)
    xdot = xdot.at[11].set((Jx * N_tot + Jxz * L_tot + (Jx * (Jx - Jy) + Jxz * Jxz) * P * Q - Jxz * (Jx - Jy + Jz) * Q * R + Jx * Q * Heng) / denom)

    return xdot


@jit
def update(x, u, dt):
    xu = jnp.hstack((x, u))
    xdot = nlplant(xu)
    next_x = x + xdot[:12] * dt
    return next_x
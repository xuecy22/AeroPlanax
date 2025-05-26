import jax


@jax.jit
def TAS2IAS(VTAS, rho, Ps, Ts):
    """Calculate Indicated airspeed, dynamic pressure and mach number by true air speed and air density

    Args:
        VTAS (_type_): true air speed, unit in m/s
        rho (_type_): Air density, unit in kg/m^3
        Ps (_type_): Static pressure, unit in Pascal
        Ts (_type_): Static temperature, unit in Celcius

    Returns:
        VIAS: Indicated airspeed, unit in m/s
        qbar: Dynamic pressure, unit in Pascal
        Mach: Mach number
    """
    gama = 1.4             # 比热容比
    R = 287.05287        # J/(kg*K)
    p0 = 101325  # Pa    Standard sea level atmospheric pressure
    # qbar = ρ*V*V*1/2
    # VTas^2 = ((2*gama*Ps)/(gama-1)*rho)*((qbar/Ps+1)^((gama-1)/gama))-1)
    # qbar = (((VTAS^2*(gama-1)*rho)/(2*gama*Ps)+1)^(gama/(gama-1))-1)*Ps
    # gama = 1.4 ==> qbar = (((VTAS^2*(0.4)*rho)/(2*1.4*Ps)+1)^(1.4/(1.4-1))-1)*Ps;
    qbar = (((VTAS ^ 2*(0.4)*rho)/(2.8*Ps)+1) ^ (1.4/0.4)-1)*Ps
    # 使用校正空速作为指示空速，VCAS = VIAS + Vdelta，Vdelta为位置误差或静压源误差，静压准确，忽略不计。
    # ((2*gama*p0)/(gamga-1)*rho0)^(1/2) = ((2*1.4*p0)/(0.4*1.225))^(1/2) = 760.9205;
    VIAS = 760.9205 * ((qbar / p0 + 1) ^ ((gama - 1) / gama) - 1) ^ (1 / 2)
    # Mach = VTAS/(gama*R*T)^(1/2);(gama*R*T)^(1/2)为当地音速。
    Mach = VTAS/ (gama * R * Ts) ^ (1/2)
    return VIAS, qbar, Mach

from jax import jit

@jit
def airdensity(Ts, Ps):
    """Calculate airdensity of current position

    Args:
        Ts (_type_): 静温 摄氏度
        Ps (_type_): 静压 帕

    Returns:
        rho: Air density, unit in kg/m^3
    """
    R = 287.05287        # J/(kg*K)
    rho = Ps / (R * (Ts + 273.15) + 1e-8)
    return rho

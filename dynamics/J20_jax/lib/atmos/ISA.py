import jax
import jax.numpy as jnp


@jax.jit
def ISA(alt):
    """atmos  Standard ISA model

    Args:
        alt (float): Mean sea level altitude, unit in meters

    Returns:
        T: Static temperature, unit in Kelvin
        rho: Air density, unit in kg/m^3
        ps: Static pressure, unit in Pascal
    """

    r0 = 6356.766         # km    effective earth radius
    p0 = 101325         # Pa    Standard sea level atmospheric pressure
    g0 = 9.80665          # m/s^2 Standard sea level gravity acceleration
    R = 287.05287         # J/(kg*K)
    a = - g0 / R
    # H 为位势高度。与alt换算公式为： H = (r0*alt)/(r0+alt),
    # r0为有效地球半径，6356.766km
    H = (r0 * 1000 * alt) / (r0 * 1000 + alt)
    Hb, Tb, Lb, pb = get_table(H)
    # alt-H-T
    T = Tb + Lb * (H - Hb)
    # alt-H-Ps
    Ps = jax.lax.select(Lb == 0, pb*jnp.exp((a/Tb)*(H-Hb)), pb*((1+(Lb/Tb)*(H-Hb))**(a/Lb)))
    # alt-H-T-PS-rho
    # Ps = rho* R * T
    # rho = Ps/(t*T)
    rho = Ps/(R*T)
    return T, rho, Ps


@jax.jit
def get_table(H):
    """atmos  Standard ISA model

    Hb  
    Tb
    Lb 
    pb 
    """
    a = 0.0001  # You might need to define the value of 'a'

    def branch_0():
        # H < 11000
        Hb = 0.0
        Tb = 288.15
        Lb = -0.0065
        pb = 101325.0
        return Hb, Tb, Lb, pb
    
    def branch_1():
        # 11000 <= H < 20000
        Hb = 11000.0
        Tb = 216.65
        Lb = 0.0
        pb = 22632.0
        return Hb, Tb, Lb, pb
    
    def branch_2():
        # 20000 <= H < 32000
        Hb = 20000.0
        Tb = 216.65
        Lb = 0.001
        pb = 5474.9
        return Hb, Tb, Lb, pb

    def branch_3():
        # 32000 <= H < 47000
        Hb = 32000.0
        Tb = 228.65
        Lb = 0.0028
        pb = 868.0194
        return Hb, Tb, Lb, pb

    def branch_4():
        # 47000 <= H < 51000
        Hb = 47000.0
        Tb = 270.65
        Lb = 0.0
        pb = 110.9062
        return Hb, Tb, Lb, pb
    
    def branch_5():
        # 51000 <= H < 71000
        Hb = 51000.0
        Tb = 270.65
        Lb = 0.0028
        pb = 66.9388
        return Hb, Tb, Lb, pb

    def branch_6():
        # 71000 <= H < 84852
        Hb = 71000.0
        Tb = 214.65
        Lb = 0.0028
        pb = 3.9564
        return Hb, Tb, Lb, pb

    def branch_7():
        # H >= 84852
        Hb = 84852.0
        Tb = 186.87
        Lb = 0.0
        pb = 0.3734
        return Hb, Tb, Lb, pb

    # Define a list of conditions
    conditions = jnp.array([
        H < 11000,          # Condition for branch_0
        (11000 <= H) & (H < 20000),  # Condition for branch_1
        (20000 <= H) & (H < 32000),  # Condition for branch_2
        (32000 <= H) & (H < 47000),  # Condition for branch_3
        (47000 <= H) & (H < 51000),  # Condition for branch_4
        (51000 <= H) & (H < 71000),  # Condition for branch_5
        (71000 <= H) & (H < 84852),  # Condition for branch_6
        H >= 84852           # Condition for branch_7
    ], dtype=jnp.bool)

    # Use lax.switch to select the correct branch based on the conditions
    selected_branch = jax.lax.switch(
        jnp.argmax(conditions), 
        [branch_0, branch_1, branch_2, branch_3, branch_4, branch_5, branch_6, branch_7]
    )

    return selected_branch


# if __name__ == "__main__":
#     import matplotlib.pyplot as plt
#     alt = jnp.linspace(0, 10000, 1000)
#     ISA_vmap = jax.jit(jax.vmap(ISA, in_axes=(0)))
#     T, rho, Ps = ISA_vmap(alt)
    
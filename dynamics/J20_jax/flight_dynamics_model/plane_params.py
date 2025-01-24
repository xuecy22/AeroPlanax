import flax.struct
import jax.numpy as jnp
from ..lib import rigid_body
from . import fuel_tank
import flax
import jax


S_wing = 1.28
wingspan = 1.804
chord = 0.856
# e = 0.9
AR = wingspan**2/S_wing
# Plane origin set to nose tip
# Estimation of inertia matrix, method 1
emptyInertia = rigid_body.createRigidbody(m=18.4265, Jx=1.043028,
                            Jy=9.135626, Jz=9.633677, Jxz=0.0, rCG=jnp.array([-1.62219, -0.00554, -0.0046142]))
# Estimation of inertia matrix, method 2
# self.Jx = 0.6442618
# self.Jy = 4.5940122
# self.Jz = 6.715574632
rFGR = jnp.array([-0.808, 0.0, 0.15])
rLMGR = jnp.array([-1.703, -0.15, 0.15])
rRMGR = jnp.array([-1.703, 0.15, 0.15])

@flax.struct.dataclass
class CanardPlaneParams:
    fueltank: fuel_tank.FuelTank
    inertia: rigid_body.RigidBody

def createPlaneParams(fuel=-1):
    # Center of gravity of empty plane in Body frame, unit in meters
    # self.rCG_empty = np.array([-1.595, 0, 0])
    # self.rCG = np.array([-1.595, 0, 0])
    fueltank = jax.lax.cond(fuel > 0,
                            lambda: fuel_tank.createFueltank(volume=fuel),
                            lambda: fuel_tank.createFueltank())
    inertia = rigid_body.createCombination(
        emptyInertia, jnp.array([0, 0, 0]), fueltank.inertia, fuel_tank.rFuel)
    state = CanardPlaneParams(
        fueltank=fueltank,
        inertia=inertia
    )
    return state
    # print("Total mass:%.2f, plane mass:%.2f, fuel mass:%.2f" %
    #       (self.inertia.mass, self.emptyInertia.mass, self.fueltank.inertia.mass))

def updatePlaneInertia(state, deltaT, SFC):
    """Update plane inertia using current fuel consumption rate

    Args:
        deltaT      Time step, unit in seconds
        SFC         Fuel consumption rate, unit in kg/h
    """
    # Update fuel tank inertia
    fueltank = fuel_tank.consumpFuel(state.fueltank, deltaT, SFC)
    # Combine empty plane inertia with fuel tank inertia
    inertia = rigid_body.rigidCombine(
        state.inertia, emptyInertia, jnp.array([0, 0, 0]), state.fueltank.inertia, fuel_tank.rFuel)
    state = state.replace(fueltank=fueltank, inertia=inertia)
    return state

import flax.struct
import jax.numpy as jnp
from ..lib import rigid_body
import flax
import jax

Length = 0.3   # fuel tank length, unit in meters
Width = 0.18   # fuel tank width, unit in meters
Height = 0.13   # fuel tank height, unit in meters
Capacity = Length*Width*Height*1000   # fuel tank capacity, unit in liters
# Relative position from plane origin to fuel tank origin in Body frame, unit in meters
rFuel = jnp.array([-1.4778, 0, 0])
mShell = 0.02   # fuel tank shell mass, unit in kg
# fuel tank shell inertia, unit in kg.m^2
S_LW = Length*Width
S_WH = Width*Height
S_HL = Height*Length
Jx_Shell = mShell*S_WH / (S_LW+S_WH+S_HL)*(Width**2+Height**2)/12.0
Jy_Shell = mShell*S_HL / (S_LW+S_WH+S_HL)*(Length**2+Height**2)/12.0
Jz_Shell = mShell*S_LW / (S_LW+S_WH+S_HL)*(Length**2+Width**2)/12.0

@flax.struct.dataclass
class FuelTank:
    """_summary_
    """
    density: float
    volume: float
    M: float
    Jx: float
    Jy: float
    Jz: float
    fullInertia: rigid_body.RigidBody
    percent: float
    inertia: rigid_body.RigidBody
    

def createFueltank(volume=Capacity, fuel_density=0.85):
    """FuelTank

    Args:
        volume (float, optional): Fuel tank volume, unit in L. Defaults to Capacity.
        fuel_density (float, optional): Fuel density, unit in kg/L. Defaults to 0.85.
    """
    # Fuel density, unit in kg/L
    density = fuel_density
    volume = jnp.clip(volume, 0.0, Capacity)

    # Calculate inertia of fuel at full tank state
    M = Capacity*density
    Jx = M*(Width**2+Height**2)/12.0
    Jy = M*(Length**2+Height**2)/12.0
    Jz = M*(Length**2+Width**2)/12.0

    # Set fuel tank origin to tank center, thus rCG = [0, 0, 0]
    fullInertia = rigid_body.createRigidbody(m=M, Jx=Jx, Jy=Jy, Jz=Jz)
    percent = volume/Capacity
    # Calculate inertia of fuel tank at current fuel volume
    inertia = rigid_body.createRigidbody(m=M*percent+mShell, Jx=Jx*percent+Jx_Shell,
                                Jy=Jy*percent+Jy_Shell, Jz=Jz*percent+Jz_Shell)
    percent *= 100.0
    state = FuelTank(
        density=density,
        volume=volume,
        M=M,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        fullInertia=fullInertia,
        percent=percent,
        inertia=inertia
    )
    return state

def consumpFuel(state, deltaT, SFC):
    """Update fuel tank volume and mass providing fuel consumption rate

    Args:
        deltaT (float): simulation physical delta time, unit in second
        SFC (float): specific fuel consumption, unit in kg/h
    """
    volume = state.volume - deltaT * SFC / (state.density * 3600)
    volume = jax.lax.select(volume < 0, 0.0, volume)
    percent = volume / Capacity
    # self.mFuel = self.volume*self.density
    # self.mass = self.mFuel + FuelTank.mShell
    # Update inertia
    mass = state.fullInertia.mass * percent + mShell
    J = state.fullInertia._J * percent \
        + jnp.array([[Jx_Shell, 0, 0],
                     [0, Jy_Shell, 0],
                     [0, 0, Jz_Shell]])
    percent *= 100.0
    state = state.replace(volume=volume, percent=percent,
                         inertia=state.inertia.replace(mass=mass))
    state = state.replace(inertia=rigid_body.setJ(state.inertia, J))
    return state

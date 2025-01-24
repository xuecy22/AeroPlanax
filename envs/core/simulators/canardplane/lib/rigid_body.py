import flax.struct
import jax.numpy as jnp
import flax
import jax


@flax.struct.dataclass
class RigidBody:
    mass: float
    Jx: float
    Jy: float
    Jz: float
    Jxy: float
    Jxz: float
    Jyz: float
    rCG: jnp.ndarray
    _J: jnp.ndarray
    _Jinv: jnp.ndarray
    _Jchanged: bool


def createRigidbody(m=1.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=jnp.zeros(3)):
    state = RigidBody(
        mass=jax.lax.select(m <= 0.0, 0.0, m),             # mass, kg
        # Innertia matrix relative to CG, unit in kg*m^2
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        Jxy=Jxy,
        Jxz=Jxz,
        Jyz=Jyz,
        # center of gravity relative to origin, unit in meters
        rCG=rCG,
        _J=jnp.array([[Jx, -Jxy, -Jxz],
                      [-Jxy, Jy, -Jyz],
                      [-Jxz, -Jyz, Jz]]),
        _Jinv=jnp.array([[Jx, -Jxy, -Jxz],
                         [-Jxy, Jy, -Jyz],
                         [-Jxz, -Jyz, Jz]]),
        _Jchanged=False
    )
    state = state.replace(_Jinv=jnp.linalg.inv(state._J))
    return state

def J(state):
    return state._J

def setJ(state, value):
    _J = 0.5*(value + value.transpose())  # make sure it is symmetric
    Jx = value[0, 0]
    Jy = value[1, 1]
    Jz = value[2, 2]
    Jxy = -value[0, 1]
    Jxz = -value[0, 2]
    Jyz = -value[1, 2]
    _Jchanged = True
    state = state.replace(Jx=Jx, Jy=Jy, Jz=Jz, Jxy=Jxy, Jxz=Jxz, Jyz=Jyz, _J=_J, _Jchanged=_Jchanged)
    return state

def Jinv(state):
    state = state.replace(_Jinv=jax.lax.select(state._Jchanged, jnp.linalg.inv(state._J), state._Jinv))
    state = state.replace(_Jchanged=jax.lax.select(state._Jchanged, False, True))
    return state._Jinv

def createCombination(rigid1, r1, rigid2, r2):
    """New Object, Combine two rigid bodies to a new rigid body object
    Args:
        rigid1      Rigid body 1
        r1          Relative position from new origin to rigid1's origin, unit in meters
        rigid2      Rigid body 2
        r2          Relative position from new origin to rigid2's origin, unit in meters

    Returns:
        RigidBody: new RigidBody object
    """
    mass, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG = cal_rigidCombine(rigid1, r1, rigid2, r2)
    return createRigidbody(m=mass, Jx=Jx, Jy=Jy, Jz=Jz, Jxy=Jxy, Jxz=Jxz, Jyz=Jyz, rCG=rCG)

def rigidCombine(state, rigid1, r1, rigid2, r2):
    """Update by combining two rigid bodies
    Args:
        rigid1      Rigid body 1
        r1          Relative position from new origin to rigid1's origin, unit in meters
        rigid2      Rigid body 2
        r2          Relative position from new origin to rigid2's origin, unit in meters
    """
    mass, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG = cal_rigidCombine(rigid1, r1, rigid2, r2)
    state = state.replace(mass=mass, Jx=Jx, Jy=Jy, Jz=Jz, Jxy=Jxy, Jxz=Jxz, Jyz=Jyz, rCG=rCG)
    return state

def cal_rigidCombine(rigid1, r1: jnp.ndarray, rigid2, r2: jnp.ndarray):
    """Calculate inertia of two rigid bodies combination
    Args:
        rigid1      Rigid body 1
        r1          Relative position from new origin to rigid1's origin, unit in meters
        rigid2      Rigid body 2
        r2          Relative position from new origin to rigid2's origin, unit in meters

    Returns:
        (m, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG): RigidBody Init Params
    """
    m = rigid1.mass + rigid2.mass
    # CG of combined rigid body relative to new origin, unit in meters
    rCG = jax.lax.select(m == 0.0,
                            ((r1+rigid1.rCG) + (r2+rigid2.rCG))/2.0,
                            (rigid1.mass*(r1+rigid1.rCG) + rigid2.mass*(r2+rigid2.rCG))/m)
    r1CG = r1 + rigid1.rCG - rCG
    r2CG = r2 + rigid2.rCG - rCG
    r1square = r1CG**2
    r2square = r2CG**2
    Jx = rigid1.Jx + rigid2.Jx + rigid1.mass * \
        (r1square[1]+r1square[2]) + rigid2.mass*(r2square[1]+r2square[2])
    Jy = rigid1.Jy + rigid2.Jy + rigid1.mass * \
        (r1square[0]+r1square[2]) + rigid2.mass*(r2square[0]+r2square[2])
    Jz = rigid1.Jz + rigid2.Jz + rigid1.mass * \
        (r1square[0]+r1square[1]) + rigid2.mass*(r2square[0]+r2square[1])
    Jxy = rigid1.Jxy + rigid2.Jxy + rigid1.mass * \
        r1CG[0]*r1CG[1] + rigid2.mass*r2CG[0]*r2CG[1]
    Jxz = rigid1.Jxz + rigid2.Jxz + rigid1.mass * \
        r1CG[0]*r1CG[2] + rigid2.mass*r2CG[0]*r2CG[2]
    Jyz = rigid1.Jyz + rigid2.Jyz + rigid1.mass * \
        r1CG[1]*r1CG[2] + rigid2.mass*r2CG[1]*r2CG[2]

    return m, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG

# if __name__ == "__main__":
#     a = create_rigidbody(m=10.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=jnp.array([1, 2, 3]))
#     ra = jnp.array([0.1, 0.2, 0.3])
#     b = create_rigidbody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=jnp.array([4, 5, 6]))
#     rb = jnp.array([0.18, 2, -1.4])
#     c = create_rigidbody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=jnp.array([4, 5, 6]))
#     rc = jnp.array([1, -5, 10])
#     d = create_rigidbody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=jnp.array([4, 5, 6]))
#     rd = jnp.array([0.8, 2, -30])
#     res = cal_rigidCombine(a, ra, b, rb)
#     res = cal_rigidCombine(create_rigidbody(*res), jnp.zeros(3), c, rc)
#     res1 = cal_rigidCombine(create_rigidbody(*res), jnp.zeros(3), d, rd)
#     print(res1)

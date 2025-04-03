import flax.struct
import jax.numpy as jnp
from . import geocal
import flax
import jax


@flax.struct.dataclass
class PositionLLA:
    Latitude: float
    Longitude: float
    Altitude: float
    x: float
    y: float
    z: float
    OriginLatitude: float
    OriginLongitude: float
    OriginAltitude: float
    # LLA: Latitude(deg), Longitude(deg), Altitude(deg)


def createPositionLLA(lat=0.0, lon=0.0, alt=0.0, latOrigin=0.0, lonOrigin=0.0, altOrigin=0.0):
    Latitude = jnp.float32(lat)
    Longitude = jnp.float32(lon)
    Altitude = jnp.float32(alt)
    x = jnp.float32(0.0)
    y = jnp.float32(0.0)
    z = jnp.float32(0.0)
    OriginLatitude = jax.lax.select(latOrigin != 0,
                                    jnp.float32(latOrigin),
                                    Latitude)
    OriginLongitude = jax.lax.select(lonOrigin != 0,
                                     jnp.float32(lonOrigin),
                                     Longitude)
    OriginAltitude = jax.lax.select(altOrigin != 0,
                                    jnp.float32(altOrigin),
                                    Altitude)
    state = PositionLLA(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        x=x,
        y=y,
        z=z,
        OriginLatitude=OriginLatitude,
        OriginLongitude=OriginLongitude,
        OriginAltitude=OriginAltitude
    )
    return state

def setCurLLA(state, lat, lon, alt):
    Latitude = jnp.float32(lat)
    Longitude = jnp.float32(lon)
    Altitude = jnp.float32(alt)
    x, y, z = geocal.LLA_to_localNED(
        Latitude, Longitude, Altitude, state.OriginLatitude, state.OriginLongitude, state.OriginAltitude)
    state = state.replace(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        x=jnp.float32(x),
        y=jnp.float32(y),
        z=jnp.float32(z),
    )
    return state

def setCurXYZ(state, x, y, z):
    x = jnp.float32(x)
    y = jnp.float32(y)
    z = jnp.float32(z)

    Latitude, Longitude, Altitude = geocal.localNED_to_LLA(
        x, y, z,
        state.OriginLatitude,
        state.OriginLongitude,
        state.OriginAltitude
    )
    state = state.replace(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        x=jnp.float32(x),
        y=jnp.float32(y),
        z=jnp.float32(z),
    )
    return state

def setNewOrigin(state, ori_lat, ori_lon, ori_alt):
    OriginLatitude = jnp.float32(ori_lat)
    OriginLongitude = jnp.float32(ori_lon)
    OriginAltitude = jnp.float32(ori_alt)
    x, y, z = geocal.LLA_to_localNED(
        state.Latitude, state.Longitude, state.Altitude, OriginLatitude, OriginLongitude, OriginAltitude)
    state = state.replace(
        x=jnp.float32(x),
        y=jnp.float32(y),
        z=jnp.float32(z),
        OriginLatitude=OriginLatitude,
        OriginLongitude=OriginLongitude,
        OriginAltitude=OriginAltitude
    )
    return state

def rebaseNewOrigin(state, ori_lat, ori_lon, ori_alt):
    OriginLatitude = jnp.float32(ori_lat)
    OriginLongitude = jnp.float32(ori_lon)
    OriginAltitude = jnp.float32(ori_alt)
    Latitude, Longitude, Altitude = geocal.localNED_to_LLA(
        state.x, state.y, state.z, OriginLatitude, OriginLongitude, OriginAltitude)
    state = state.replace(
        Latitude=jnp.float32(Latitude),
        Longitude=jnp.float32(Longitude),
        Altitude=jnp.float32(Altitude),
        OriginLatitude=OriginLatitude,
        OriginLongitude=OriginLongitude,
        OriginAltitude=OriginAltitude
    )
    return state


def getCurrentDCM(state):
    return geocal.cosinematrix_ECEF_to_NED(
        state.Latitude.astype(jnp.float64),
        state.Longitude.astype(jnp.float64),
    )


def getOriginDCM(state):
    return geocal.cosinematrix_ECEF_to_NED(
        state.OriginLatitude.astype(jnp.float64),
        state.OriginLongitude.astype(jnp.float64),
    )

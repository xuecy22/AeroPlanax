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


def createPositionLLA(self, lat=30.0, lon=118.0, alt=0.0, latOrigin=0.0, lonOrigin=0.0, altOrigin=0.0):
    Latitude = lat
    Longitude = lon
    Altitude = alt
    x = 0
    y = 0
    z = 0
    OriginLatitude = jax.lax.select(latOrigin != 0, latOrigin, lat)
    OriginLongitude = jax.lax.select(lonOrigin != 0, lonOrigin, lon)
    OriginAltitude = jax.lax.select(altOrigin != 0, altOrigin, alt)
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
    Latitude = lat
    Longitude = lon
    Altitude = alt
    x, y, z = geocal.LLA_to_localNED(
        Latitude, Longitude, Altitude, state.OriginLatitude, state.OriginLongitude, state.OriginAltitude)
    state = state.replace(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        x=x,
        y=y,
        z=z
    )
    return state

def setCurXYZ(state, x, y, z):
    Latitude, Longitude, Altitude = geocal.localNED_to_LLA(
        x, y, z, state.OriginLatitude, state.OriginLongitude, state.OriginAltitude)
    state = state.replace(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        x=x,
        y=y,
        z=z
    )
    return state

def setNewOrigin(state, ori_lat, ori_lon, ori_alt):
    OriginLatitude = ori_lat
    OriginLongitude = ori_lon
    OriginAltitude = ori_alt
    x, y, z = geocal.LLA_to_localNED(
        state.Latitude, state.Longitude, state.Altitude, OriginLatitude, OriginLongitude, OriginAltitude)
    state = state.replace(
        x=x,
        y=y,
        z=z,
        OriginLatitude=OriginLatitude,
        OriginLongitude=OriginLongitude,
        OriginAltitude=OriginAltitude
    )
    return state

def rebaseNewOrigin(state, ori_lat, ori_lon, ori_alt):
    OriginLatitude = ori_lat
    OriginLongitude = ori_lon
    OriginAltitude = ori_alt
    Latitude, Longitude, Altitude = geocal.localNED_to_LLA(
        state.x, state.y, state.z, OriginLatitude, OriginLongitude, OriginAltitude)
    state = state.replace(
        Latitude=Latitude,
        Longitude=Longitude,
        Altitude=Altitude,
        OriginLatitude=OriginLatitude,
        OriginLongitude=OriginLongitude,
        OriginAltitude=OriginAltitude
    )
    return state

def getCurrentDCM(state):
    return geocal.cosinematrix_ECEF_to_NED(state.Latitude, state.Longitude)

def getOriginDCM(state):
    return geocal.cosinematrix_ECEF_to_NED(state.OriginLatitude, state.OriginLongitude)

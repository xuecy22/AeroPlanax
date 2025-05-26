import jax.numpy as jnp
import jax


def LLA_to_ECEF(latitude, longitude, altitude):
    # latitude:纬度 longitude:经度 altitude:海拔
    # 经纬度的余弦值
    cosLat = jnp.cos(latitude * jnp.pi / 180)
    sinLat = jnp.sin(latitude * jnp.pi / 180)
    cosLon = jnp.cos(longitude * jnp.pi / 180)
    sinLon = jnp.sin(longitude * jnp.pi / 180)

    # WGS84坐标系的参数
    a = 6378137.0  # 地球参考椭球体半长轴
    f = 1.0 / 298.257223563   # 参考椭球体扁率 :f = (a-b)/a
    C = 1.0 / jnp.sqrt(cosLat * cosLat + (1-f) * (1-f) *
                        sinLat * sinLat)   # 主垂直曲率半径 N = a * C
    S = (1-f) * (1-f) * C
    h = altitude

    # 计算该位置在ECEF坐标系下的XYZ坐标
    X = (a * C + h) * cosLat * cosLon
    Y = (a * C + h) * cosLat * sinLon
    Z = (a * S + h) * sinLat

    return jnp.array([X, Y, Z])


def ECEF_to_LLA(X, Y, Z):
    # WGS84坐标系的参数
    a = 6378137.0        # 椭球长半轴
    b = 6356752.314245   # 椭球短半轴
    ea = jnp.sqrt((a ** 2 - b ** 2) / a ** 2)
    eb = jnp.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = jnp.sqrt(X ** 2 + Y ** 2)
    theta = jnp.arctan2(Z * a, p * b)

    # 计算经纬度及海拔
    longitude = jnp.arctan2(Y, X)
    latitude = jnp.arctan2(Z + eb ** 2 * b * jnp.sin(theta)
                          ** 3, p - ea ** 2 * a * jnp.cos(theta) ** 3)
    N = a / jnp.sqrt(1 - ea ** 2 * jnp.sin(latitude) ** 2)
    altitude = p / jnp.cos(latitude) - N

    return jnp.degrees(latitude), jnp.degrees(longitude), altitude


def cosinematrix_ECEF_to_NED(latitude, longitude):
    cosLat = jnp.cos(jnp.radians(latitude))
    sinLat = jnp.sin(jnp.radians(latitude))
    cosLon = jnp.cos(jnp.radians(longitude))
    sinLon = jnp.sin(jnp.radians(longitude))
    return jnp.array([[-sinLat * cosLon, -sinLat * sinLon, cosLat],
                      [-sinLon, cosLon, 0],
                      [-cosLat * cosLon, -cosLat * sinLon, -sinLat]])


def get_distance(lat1, lon1, lat2, lon2, alt1, alt2):
    # 根据经度、纬度、海拔计算两地距离
    v1 = LLA_to_ECEF(lat1, lon1, alt1)
    v2 = LLA_to_ECEF(lat2, lon2, alt2)
    distance = jnp.linalg.norm(v1 - v2)

    return distance


def get_distance_rough(lat1, lon1, lat2, lon2, alt1, alt2):
    lat1 = jnp.radians(lat1)
    lon1 = jnp.radians(lon1)
    lat2 = jnp.radians(lat2)
    lon2 = jnp.radians(lon2)

    dlon = lon2 - lon1
    cos_d_lon = jnp.cos(dlon)
    sin_d_lon = jnp.sin(dlon)

    sin_lat1 = jnp.sin(lat1)
    cos_lat1 = jnp.cos(lat1)
    sin_lat2 = jnp.sin(lat2)
    cos_lat2 = jnp.cos(lat2)

    a = 6378137.0
    c = jnp.acos(jnp.clip(sin_lat1 * sin_lat2 + cos_lat1 *
                  cos_lat2 * cos_d_lon, -1.0,  1.0))
    k = 1/jnp.sinc(c/jnp.pi)
    y = k * a * sin_d_lon * cos_lat2
    x = k * a * cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_d_lon
    z = alt1 - alt2

    return jnp.sqrt(x * x + y * y + z * z)


def get_bearing(lat1, lon1, lat2, lon2):
    lat1 = jnp.radians(lat1)
    lon1 = jnp.radians(lon1)
    lat2 = jnp.radians(lat2)
    lon2 = jnp.radians(lon2)
    dlon = lon2 - lon1

    y = jnp.sin(dlon) * jnp.cos(lat2)
    x = jnp.cos(lat1) * jnp.sin(lat2) - jnp.sin(lat1) * \
        jnp.cos(lat2) * jnp.cos(dlon)

    bearing = jnp.atan2(y, x)
    bearing = jnp.degrees(bearing)
    bearing = (bearing + 360.0) % 360.0

    return bearing


def LLA_to_localNED(latTarget, lonTarget, altTarget, latOrigin, lonOrigin, altOrigin):
    # latitude:纬度 longitude:经度 altitude:海拔
    dlon = jnp.radians(lonTarget - lonOrigin)
    cos_d_lon = jnp.cos(dlon)
    sin_d_lon = jnp.sin(dlon)

    sin_lat1 = jnp.sin(jnp.radians(latOrigin))
    cos_lat1 = jnp.cos(jnp.radians(latOrigin))
    sin_lat2 = jnp.sin(jnp.radians(latTarget))
    cos_lat2 = jnp.cos(jnp.radians(latTarget))

    cos_c = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_d_lon
    # if the two points are close(within 145km), use the simplified formula
    def branch_0():
        c = jnp.acos(jnp.clip(cos_c, -1.0,  1.0))
        k = 1/jnp.sinc(c/jnp.pi)
        a = 6378137.0
        x = k * a * (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_d_lon)
        y = k * a * sin_d_lon * cos_lat2
        z = altOrigin - altTarget
        return jnp.array([x, y, z])
    
    def branch_1():
        r1 = LLA_to_ECEF(latOrigin, lonOrigin, altOrigin)
        r2 = LLA_to_ECEF(latTarget, lonTarget, altTarget)
        r21 = r2 - r1
        C_ECEF_to_NED = cosinematrix_ECEF_to_NED(latOrigin, lonOrigin)
        return jnp.dot(C_ECEF_to_NED, r21)
    
    result = jax.lax.cond(cos_c > 0.99974, branch_0(), branch_1())
    return result


def localNED_to_LLA(x, y, z, latOrigin, lonOrigin, altOrigin):
    # latitude:纬度 longitude:经度 altitude:海拔
    RADIUS_OF_EARTH = 6371000
    x_rad = x / RADIUS_OF_EARTH
    y_rad = y / RADIUS_OF_EARTH
    c = jnp.sqrt(x_rad * x_rad + y_rad * y_rad)

    def branch_0():
        return latOrigin, lonOrigin, altOrigin - z
    
    def branch_1():
        sin_c = jnp.sin(c)
        cos_c = jnp.cos(c)
        _ref_lon = jnp.radians(lonOrigin)
        _ref_sin_lat = jnp.sin(jnp.radians(latOrigin))
        _ref_cos_lat = jnp.cos(jnp.radians(latOrigin))
        lat_rad = jnp.arcsin(cos_c * _ref_sin_lat +
                            (x_rad * sin_c * _ref_cos_lat) / c)
        lon_rad = (_ref_lon + jnp.arctan2(y_rad * sin_c, c *
                   _ref_cos_lat * cos_c - x_rad * _ref_sin_lat * sin_c))

        return jnp.degrees(lat_rad), jnp.degrees(lon_rad), altOrigin - z
    
    def branch_else():
        rECEF = LLA_to_ECEF(latOrigin, lonOrigin, altOrigin)
        rECEF += jnp.dot(cosinematrix_ECEF_to_NED(latOrigin,
                        lonOrigin).transpose(), jnp.array([x, y, z]))
        return ECEF_to_LLA(rECEF[0], rECEF[1], rECEF[2])
    
    conditions = jnp.array([
        c < 1e-9,          # Condition for branch_0
        ((c > 1e-9) & (c < 0.02276)),  # Condition for branch_1
    ], dtype=jnp.bool)

    result = jax.lax.switch(
        jnp.argmax(conditions), 
        [branch_0, branch_1, branch_else]
    )
    return result


def perspective(angle):
    return (angle + 180) % 360 - 180


def get_azimuth_elevation_distance(lat1, lon1, lat2, lon2, alt1, alt2, east_speed=0, north_speed=0, ground_speed=0, dt=0):
    x, y, z = LLA_to_localNED(lat2, lon2, alt2, lat1, lon1, alt1)
    x += north_speed * dt
    y += east_speed * dt
    # z += ground_speed * dt

    azimuth = jnp.degrees(jnp.atan2(y, x))
    azimuth = (azimuth + 360.0) % 360.0
    elevation = jnp.degrees(jnp.atan2(-z, jnp.sqrt(x * x + y * y)))
    distance = jnp.sqrt(x * x + y * y + z * z)

    return perspective(azimuth), perspective(elevation), distance


def test():
    lat1, lon1, alt1 = (59.99999999999999, 119.99999999999999, 0.5)
    lat2, lon2, alt2 = (60.200000000000024, 119.99999999999999, 6095)
    lat_ori, lon_ori, alt_ori = -35.2358049, 149.1556262, 0.5
    lat1, lon1, alt1 = lat_ori, lon_ori, alt_ori
    lat1, lon1, alt1 = (34.216132, 118.231272, 33.8+1.4)
    lat2, lon2, alt2 = (34.215756, 118.231256, 34.0)
    tmp = get_azimuth_elevation_distance(lat1, lon1, lat2, lon2, alt1, alt2)
    print(tmp)


if __name__ == "__main__":
    test()

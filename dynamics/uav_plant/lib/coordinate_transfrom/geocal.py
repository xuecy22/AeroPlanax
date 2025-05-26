#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-14 09:56:09
@ LastEditors: Yega
@ Description: Calculate Coordinate conversion from longitude and latitude, calculate azimuth
'''
import math
import numpy as np


def LLA_to_ECEF(latitude, longitude, altitude):
    # latitude:纬度 longitude:经度 altitude:海拔
    # 经纬度的余弦值
    cosLat = math.cos(latitude * math.pi / 180)
    sinLat = math.sin(latitude * math.pi / 180)
    cosLon = math.cos(longitude * math.pi / 180)
    sinLon = math.sin(longitude * math.pi / 180)

    # WGS84坐标系的参数
    a = 6378137.0  # 地球参考椭球体半长轴
    f = 1.0 / 298.257223563   # 参考椭球体扁率 :f = (a-b)/a
    C = 1.0 / math.sqrt(cosLat * cosLat + (1-f) * (1-f) *
                        sinLat * sinLat)   # 主垂直曲率半径 N = a * C
    S = (1-f) * (1-f) * C
    h = altitude

    # 计算该位置在ECEF坐标系下的XYZ坐标
    X = (a * C + h) * cosLat * cosLon
    Y = (a * C + h) * cosLat * sinLon
    Z = (a * S + h) * sinLat

    return np.array([X, Y, Z])


def LLA_to_ECEF_to_List(latitude, longitude, altitude):
    return LLA_to_ECEF(latitude, longitude, altitude).tolist()


def ECEF_to_LLA(X, Y, Z):
    # WGS84坐标系的参数
    a = 6378137.0        # 椭球长半轴
    b = 6356752.314245   # 椭球短半轴
    ea = np.sqrt((a ** 2 - b ** 2) / a ** 2)
    eb = np.sqrt((a ** 2 - b ** 2) / b ** 2)
    p = np.sqrt(X ** 2 + Y ** 2)
    theta = np.arctan2(Z * a, p * b)

    # 计算经纬度及海拔
    longitude = np.arctan2(Y, X)
    latitude = np.arctan2(Z + eb ** 2 * b * np.sin(theta)
                          ** 3, p - ea ** 2 * a * np.cos(theta) ** 3)
    N = a / np.sqrt(1 - ea ** 2 * np.sin(latitude) ** 2)
    altitude = p / np.cos(latitude) - N

    return np.array([np.degrees(latitude), np.degrees(longitude), altitude])


def cosinematrix_ECEF_to_NED(latitude, longitude):
    cosLat = math.cos(math.radians(latitude))
    sinLat = math.sin(math.radians(latitude))
    cosLon = math.cos(math.radians(longitude))
    sinLon = math.sin(math.radians(longitude))
    return np.array([[-sinLat * cosLon, -sinLat * sinLon, cosLat],
                     [-sinLon, cosLon, 0],
                     [-cosLat * cosLon, -cosLat * sinLon, -sinLat]])


def get_distance(lat1, lon1, lat2, lon2, alt1, alt2):
    # 根据经度、纬度、海拔计算两地距离
    v1 = LLA_to_ECEF(lat1, lon1, alt1)
    v2 = LLA_to_ECEF(lat2, lon2, alt2)
    distance = np.linalg.norm(v1 - v2)

    return distance


def get_distance_rough(lat1, lon1, lat2, lon2, alt1, alt2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    dlon = lon2 - lon1
    cos_d_lon = math.cos(dlon)
    sin_d_lon = math.sin(dlon)

    sin_lat1 = math.sin(lat1)
    cos_lat1 = math.cos(lat1)
    sin_lat2 = math.sin(lat2)
    cos_lat2 = math.cos(lat2)

    a = 6378137.0
    c = math.acos(np.clip(sin_lat1 * sin_lat2 + cos_lat1 *
                  cos_lat2 * cos_d_lon, -1.0,  1.0))
    k = 1/np.sinc(c/np.pi)
    y = k * a * sin_d_lon * cos_lat2
    x = k * a * cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_d_lon
    z = alt1 - alt2

    return math.sqrt(x * x + y * y + z * z)


def get_bearing(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1

    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * \
        math.cos(lat2) * math.cos(dlon)

    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360.0) % 360.0

    return bearing


def LLA_to_localNED(latTarget, lonTarget, altTarget, latOrigin, lonOrigin, altOrigin):
    # latitude:纬度 longitude:经度 altitude:海拔
    dlon = math.radians(lonTarget - lonOrigin)
    cos_d_lon = math.cos(dlon)
    sin_d_lon = math.sin(dlon)

    sin_lat1 = math.sin(math.radians(latOrigin))
    cos_lat1 = math.cos(math.radians(latOrigin))
    sin_lat2 = math.sin(math.radians(latTarget))
    cos_lat2 = math.cos(math.radians(latTarget))

    cos_c = sin_lat1 * sin_lat2 + cos_lat1 * cos_lat2 * cos_d_lon
    # if the two points are close(within 145km), use the simplified formula
    if cos_c > 0.99974:
        c = math.acos(np.clip(cos_c, -1.0,  1.0))
        k = 1/np.sinc(c/np.pi)
        a = 6378137.0
        x = k * a * (cos_lat1 * sin_lat2 - sin_lat1 * cos_lat2 * cos_d_lon)
        y = k * a * sin_d_lon * cos_lat2
        z = altOrigin - altTarget
        return np.array([x, y, z])
    else:   # calculate r21 in ECEF coordinate system then convert to NED
        r1 = LLA_to_ECEF(latOrigin, lonOrigin, altOrigin)
        r2 = LLA_to_ECEF(latTarget, lonTarget, altTarget)
        r21 = r2 - r1
        C_ECEF_to_NED = cosinematrix_ECEF_to_NED(latOrigin, lonOrigin)
        return np.dot(C_ECEF_to_NED, r21)


def localNED_to_LLA(x, y, z, latOrigin, lonOrigin, altOrigin):
    # latitude:纬度 longitude:经度 altitude:海拔
    RADIUS_OF_EARTH = 6371000
    x_rad = x / RADIUS_OF_EARTH
    y_rad = y / RADIUS_OF_EARTH
    c = np.sqrt(x_rad * x_rad + y_rad * y_rad)

    if c < 1e-9:
        return latOrigin, lonOrigin, altOrigin - z
    # if the two points are close(within 145km), use the simplified formula
    elif c < 0.02276:
        sin_c = np.sin(c)
        cos_c = np.cos(c)
        _ref_lon = np.radians(lonOrigin)
        _ref_sin_lat = np.sin(np.radians(latOrigin))
        _ref_cos_lat = np.cos(np.radians(latOrigin))
        lat_rad = np.arcsin(cos_c * _ref_sin_lat +
                            (x_rad * sin_c * _ref_cos_lat) / c)
        lon_rad = (_ref_lon + np.arctan2(y_rad * sin_c, c *
                   _ref_cos_lat * cos_c - x_rad * _ref_sin_lat * sin_c))

        return np.degrees(lat_rad), np.degrees(lon_rad), altOrigin - z
    else:
        rECEF = LLA_to_ECEF(latOrigin, lonOrigin, altOrigin)
        rECEF += np.dot(cosinematrix_ECEF_to_NED(latOrigin,
                        lonOrigin).transpose(), np.array([x, y, z]))
        return ECEF_to_LLA(rECEF[0], rECEF[1], rECEF[2])


def perspective(angle):
    return (angle + 180) % 360 - 180


def get_azimuth_elevation_distance(lat1, lon1, lat2, lon2, alt1, alt2, east_speed=0, north_speed=0, ground_speed=0, dt=0):
    x, y, z = LLA_to_localNED(lat2, lon2, alt2, lat1, lon1, alt1)
    x += north_speed * dt
    y += east_speed * dt
    # z += ground_speed * dt

    azimuth = math.degrees(math.atan2(y, x))
    azimuth = (azimuth + 360.0) % 360.0
    elevation = math.degrees(math.atan2(-z, math.sqrt(x * x + y * y)))
    distance = math.sqrt(x * x + y * y + z * z)

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

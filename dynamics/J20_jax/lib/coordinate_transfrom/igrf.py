from geopack import geopack
import numpy as np
import time


def LLA_to_ECEF(latitude, longitude, altitude):
    # latitude:纬度 longitude:经度 altitude:海拔
    # 经纬度的余弦值
    cosLat = np.cos(latitude * np.pi / 180)
    sinLat = np.sin(latitude * np.pi / 180)
    cosLon = np.cos(longitude * np.pi / 180)
    sinLon = np.sin(longitude * np.pi / 180)

    # WGS84坐标系的参数
    a = 6378137.0  # 地球参考椭球体半长轴
    f = 1.0 / 298.257223563   # 参考椭球体扁率 :f = (a-b)/a
    C = 1.0 / np.sqrt(cosLat * cosLat + (1-f) * (1-f) *
                      sinLat * sinLat)   # 主垂直曲率半径 N = a * C
    S = (1-f) * (1-f) * C
    h = altitude

    # 计算该位置在ECEF坐标系下的XYZ坐标
    X = (a * C + h) * cosLat * cosLon
    Y = (a * C + h) * cosLat * sinLon
    Z = (a * S + h) * sinLat

    return np.array([X, Y, Z])


def LLA_cal_mag(lat, lon, alt, ut: int | None = None):
    ut = ut if ut is not None else time.time()
    xyz_ecef = LLA_to_ECEF(lat, lon, alt)
    r = np.linalg.norm(xyz_ecef)/1e3/6371.2
    geopack.recalc(int(ut))
    magU, magS, magE = geopack.igrf_geo(
        r, np.radians(lat), np.radians(lon))
    magNED = np.array([-magS, magE, -magU])/1e2
    declination = np.arctan2(magNED[1], magNED[0])
    magNED = magNED.tolist()
    return magNED[0], magNED[1], magNED[2], np.degrees(declination), np.linalg.norm(magNED)

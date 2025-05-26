#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-14 09:59:05
@ LastEditors: Yega
@ Description: Do edit!
'''
import numpy as np
from . import geocal


class PositionLLA:
    # LLA: Latitude(deg), Longitude(deg), Altitude(deg)
    def __init__(self, lat=30, lon=118, alt=0, latOrigin=None, lonOrigin=None, altOrigin=None):
        self.Latitude = lat
        self.Longitude = lon
        self.Altitude = alt
        self.x = 0
        self.y = 0
        self.z = 0
        self.OriginLatitude = latOrigin if latOrigin is not None else lat
        self.OriginLongitude = lonOrigin if lonOrigin is not None else lon
        self.OriginAltitude = altOrigin if altOrigin is not None else alt

    def setCurLLA(self, lat, lon, alt):
        self.Latitude = lat
        self.Longitude = lon
        self.Altitude = alt
        self.x, self.y, self.z = geocal.LLA_to_localNED(
            self.Latitude, self.Longitude, self.Altitude, self.OriginLatitude, self.OriginLongitude, self.OriginAltitude)

    def setCurXYZ(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.Latitude, self.Longitude, self.Altitude = geocal.localNED_to_LLA(
            self.x, self.y, self.z, self.OriginLatitude, self.OriginLongitude, self.OriginAltitude)

    def setNewOrigin(self, ori_lat, ori_lon, ori_alt):
        self.OriginLatitude = ori_lat
        self.OriginLongitude = ori_lon
        self.OriginAltitude = ori_alt
        self.x, self.y, self.z = geocal.LLA_to_localNED(
            self.Latitude, self.Longitude, self.Altitude, self.OriginLatitude, self.OriginLongitude, self.OriginAltitude)

    def rebaseNewOrigin(self, ori_lat, ori_lon, ori_alt):
        self.OriginLatitude = ori_lat
        self.OriginLongitude = ori_lon
        self.OriginAltitude = ori_alt
        self.Latitude, self.Longitude, self.Altitude = geocal.localNED_to_LLA(
            self.x, self.y, self.z, self.OriginLatitude, self.OriginLongitude, self.OriginAltitude)

    def getCurrentDCM(self):
        return geocal.cosinematrix_ECEF_to_NED(self.Latitude, self.Longitude)

    def getOriginDCM(self):
        return geocal.cosinematrix_ECEF_to_NED(self.OriginLatitude, self.OriginLongitude)

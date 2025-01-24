from . import geocal
import numpy as np


class changePoseLLA:
    # LLA: Latitude(deg), Longitude(deg), Altitude(deg), yaw offset(deg)
    def __init__(self, sourceLLA, targetLLA, yawOffset=0):
        self.sourceOrigin = sourceLLA
        self.targetOrigin = targetLLA
        # Source direction to target direction, clockwise is positive
        self.yawOffset = (yawOffset + 360.0) % 360.0
        self.rotateSrctoTrgt = self.getRotateMat_SrctoTrgt()

    def getRotateMat_SrctoTrgt(self):
        cosy = np.cos(np.radians(self.yawOffset))
        siny = np.sin(np.radians(self.yawOffset))
        return np.array([[cosy, siny, 0],
                         [-siny, cosy, 0],
                         [0, 0, 1]])

    def setYawOffset(self, yawOffset):
        self.yawOffset = (yawOffset + 360.0) % 360.0
        self.rotateSrctoTrgt = self.getRotateMat_SrctoTrgt()

    def setYawOffsetByLLA(self, sourceA, sourceB, targetA, targetB):
        yawOffset = geocal.get_bearing(sourceA[0], sourceA[1], sourceB[0], sourceB[1]) - \
            geocal.get_bearing(targetA[0], targetA[1], targetB[0], targetB[1])
        self.setYawOffset(yawOffset)

    def getLLAY(self, Loc, Yaw):
        relativePos = geocal.LLA_to_localNED(
            Loc[0], Loc[1], Loc[2], self.sourceOrigin[0], self.sourceOrigin[1], self.sourceOrigin[2])
        relativePos = np.dot(self.rotateSrctoTrgt, relativePos)
        targetLoc = geocal.localNED_to_LLA(
            relativePos[0], relativePos[1], relativePos[2], self.targetOrigin[0], self.targetOrigin[1], self.targetOrigin[2])
        Yaw -= self.yawOffset
        targetLoc_rad = np.radians(targetLoc)
        return targetLoc_rad[0], targetLoc_rad[1], targetLoc_rad[2], np.radians((Yaw + 360.0) % 360.0)

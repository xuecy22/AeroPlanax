#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: QiYang
@ Date: 2024-03-30 12:22:05
@ LastEditors: Yega
@ Description: Do edit!
'''
import numpy as np


class RigidBody:
    def __init__(self, m=1.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=np.zeros(3)) -> None:
        self.mass = 0.0 if m <= 0.0 else m             # mass, kg
        # Innertia matrix relative to CG, unit in kg*m^2
        self.Jx = Jx
        self.Jy = Jy
        self.Jz = Jz
        self.Jxy = Jxy
        self.Jxz = Jxz
        self.Jyz = Jyz
        self._J = np.array([[self.Jx, -self.Jxy, -self.Jxz],
                           [-self.Jxy, self.Jy, -self.Jyz],
                           [-self.Jxz, -self.Jyz, self.Jz]])
        self._Jinv = None if m <= 0.0 else np.linalg.inv(self._J)
        self._Jchanged = False
        # center of gravity relative to origin, unit in meters
        self.rCG = rCG

    @property
    def J(self):
        return self._J

    @property
    def Jinv(self):
        if self._Jchanged:
            self._Jinv = np.linalg.inv(self._J)
            self._Jchanged = False
        return self._Jinv

    @J.setter
    def J(self, value):
        self._J = 0.5*(value+value.transpose())  # make sure it is symmetric
        self.Jx = value[0, 0]
        self.Jy = value[1, 1]
        self.Jz = value[2, 2]
        self.Jxy = -value[0, 1]
        self.Jxz = -value[0, 2]
        self.Jyz = -value[1, 2]
        self._Jchanged = True

    def scale(self, percent):
        if percent < 0.0:
            percent = 0.0
        self.mass *= percent
        self.Jx *= percent
        self.Jy *= percent
        self.Jz *= percent
        self.Jxy *= percent
        self.Jxz *= percent
        self.Jyz *= percent

    def __mul__(self, percent):
        if percent < 0.0:
            percent = 0.0
        mass = self.mass * percent
        Jx = self.Jx * percent
        Jy = self.Jy * percent
        Jz = self.Jz * percent
        Jxy = self.Jxy * percent
        Jxz = self.Jxz * percent
        Jyz = self.Jyz * percent
        return RigidBody(mass, Jx, Jy, Jz, Jxy, Jxz, Jyz, self.rCG)

    @staticmethod
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
        return RigidBody(*RigidBody.cal_rigidCombine(rigid1, r1, rigid2, r2))

    @staticmethod
    def createCombination_multiple(*rigids: list[object, np.ndarray]):
        """New Object, Combine multiple rigid bodies to a new rigid body object
        Args:
            rigids      list[rigid:RigidBody, origin offset:np.array(3)]
                rigid                  RigidBody
                origin offset          Relative position from new origin to rigid's origin, unit in meters

        Returns:
            RigidBody: new RigidBody object
        """
        return RigidBody(*RigidBody.cal_rigidCombine_multiple(*rigids))

    def rigidCombine(self, rigid1, r1, rigid2, r2) -> None:
        """Update by combining two rigid bodies
        Args:
            rigid1      Rigid body 1
            r1          Relative position from new origin to rigid1's origin, unit in meters
            rigid2      Rigid body 2
            r2          Relative position from new origin to rigid2's origin, unit in meters
        """
        self.mass, self.Jx, self.Jy, self.Jz, self.Jxy, self.Jxz, self.Jyz, self.rCG = RigidBody.cal_rigidCombine(
            rigid1, r1, rigid2, r2)

    def rigidCombine_multiple(self, *rigids: list[object, np.ndarray]):
        """Update by combining multiple rigid bodies
        Args:
            rigids      list[rigid:RigidBody, origin offset:np.array(3)]
                rigid                  RigidBody
                origin offset          Relative position from new origin to rigid's origin, unit in meters
        """
        self.mass, self.Jx, self.Jy, self.Jz, self.Jxy, self.Jxz, self.Jyz, self.rCG = RigidBody.cal_rigidCombine_multiple(
            *rigids)

    @staticmethod
    def cal_rigidCombine(rigid1, r1: np.ndarray, rigid2, r2: np.ndarray):
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
        if m == 0.0:
            rCG = ((r1+rigid1.rCG) + (r2+rigid2.rCG))/2.0
        else:
            rCG = (rigid1.mass*(r1+rigid1.rCG) + rigid2.mass*(r2+rigid2.rCG))/m
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

    @staticmethod
    def cal_rigidCombine_multiple(*rigids: list[object, np.ndarray]):
        """Calculate inertia of multiple rigid bodies combination
        Args:
            rigids      list[rigid:RigidBody, origin offset:np.array(3)]
                rigid                  RigidBody
                origin offset          Relative position from new origin to rigid's origin, unit in meters

        Returns:
            (m, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG): RigidBody Init Params
        """
        m_array, Jx, Jy, Jz, Jxy, Jxz, Jyz = [np.zeros(len(rigids)) for _ in range(7)]
        r_origin_offset, rCG_to_origin = [np.zeros((len(rigids), 3)) for _ in range(2)]
        for i, rigid_sub in enumerate(rigids):
            rigid, r_offset = rigid_sub
            m_array[i] = rigid.mass
            r_origin_offset[i, :] = r_offset
            rCG_to_origin[i, :] = rigid.rCG
            Jx[i] = rigid.Jx
            Jy[i] = rigid.Jy
            Jz[i] = rigid.Jz
            Jxy[i] = rigid.Jxy
            Jxz[i] = rigid.Jxz
            Jyz[i] = rigid.Jyz

        Jx = np.sum(Jx)
        Jy = np.sum(Jy)
        Jz = np.sum(Jz)
        Jxy = np.sum(Jxy)
        Jxz = np.sum(Jxz)
        Jyz = np.sum(Jyz)
        m = np.sum(m_array)
        # CG of combined rigid body relative to new origin, unit in meters
        if m == 0.0:
            rCG = np.sum((r_origin_offset + rCG_to_origin), axis=0)/len(rigids)
        else:
            rCG = np.sum((r_origin_offset + rCG_to_origin).T*(m_array/m), axis=1)
        rCG_to_new_origin = r_origin_offset + rCG_to_origin - rCG
        rCG_to_new_origin_square = rCG_to_new_origin ** 2
        Jx += np.sum((rCG_to_new_origin_square[:, [1, 2]]).T, axis=0)@m_array
        Jy += np.sum((rCG_to_new_origin_square[:, [0, 2]]).T, axis=0)@m_array
        Jz += np.sum((rCG_to_new_origin_square[:, [0, 1]]).T, axis=0)@m_array
        Jxy += (rCG_to_new_origin[:, 0]*rCG_to_new_origin[:, 1])@m_array
        Jxz += (rCG_to_new_origin[:, 0]*rCG_to_new_origin[:, 2])@m_array
        Jyz += (rCG_to_new_origin[:, 1]*rCG_to_new_origin[:, 2])@m_array
        return m, Jx, Jy, Jz, Jxy, Jxz, Jyz, rCG


if __name__ == "__main__":
    a = RigidBody(m=10.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=np.array([1, 2, 3]))
    ra = np.array([0.1, 0.2, 0.3])
    b = RigidBody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=np.array([4, 5, 6]))
    rb = np.array([0.18, 2, -1.4])
    c = RigidBody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=np.array([4, 5, 6]))
    rc = np.array([1, -5, 10])
    d = RigidBody(m=5224.0, Jx=1.0, Jy=1.0, Jz=1.0, Jxy=0, Jxz=0, Jyz=0, rCG=np.array([4, 5, 6]))
    rd = np.array([0.8, 2, -30])
    res = RigidBody.cal_rigidCombine(a, ra, b, rb)
    res = RigidBody.cal_rigidCombine(RigidBody(*res), np.zeros(3), c, rc)
    res1 = RigidBody.cal_rigidCombine(RigidBody(*res), np.zeros(3), d, rd)
    print(res1)
    res2 = RigidBody.cal_rigidCombine_multiple([a, ra], [b, rb], [c, rc], [d, rd])
    print(res2)

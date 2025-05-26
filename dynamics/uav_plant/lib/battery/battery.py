#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
@ Author: Yega
@ Date: 2024-03-13 17:31:28
@ LastEditors: Style
@ Description: Battery: SOC-OCV Mapping curve
'''
import os
import scipy.io as scio
from scipy.interpolate import interp1d


class Battery:
    curr_path = os.path.dirname(__file__)
    data_path = os.path.join(curr_path, "OCV_SOC.mat")
    OCVSOC = scio.loadmat(data_path)
    soc_ocv_func = interp1d(OCVSOC['soc'][0], OCVSOC['ocv'][0], kind='linear')

    def __init__(self, serial=3, parallel=1, soc=1.0, cellcapacity=3000.0) -> None:
        self.innerResPerCell = 0.21                             # Unit in Ohm
        self.innerRes = (serial*self.innerResPerCell)/parallel  # Unit in Ohm
        self.ser = serial                                       # Number of cells in series
        self.par = parallel                                     # Number of cells in parallel
        self.cells = serial*parallel                            # Number of total cells
        self.OCVoltage = Battery.soc_ocv_func(soc)*0.001              # Open circuit voltage per cell, unit in V
        self.Current = 0                                        # Unit in A
        self.cellVoltage = self.OCVoltage - self.innerResPerCell*self.Current / \
            parallel            # Output voltage per cell, unit in V
        self.Voltage = serial*self.cellVoltage                  # Total output voltage, unit in V
        self.SOC = soc                                          # State of charge, 1.0 for full capacity
        self.cellCapacity = cellcapacity                        # Unit in mAh
        self.totalCapacity = cellcapacity*parallel              # Unit in mAh
        self.battCapacity = self.SOC*self.totalCapacity         # Unit in mAh

    def consume_batt(self, deltaT, current):
        '''Consume battery capacity
            deltaT: Time step, unit in seconds
            current: Current, output positive, unit in A
        '''
        self.Current = current
        self.battCapacity = self.battCapacity-(1000.0/3600.0)*self.Current*deltaT
        if self.battCapacity < 0:
            self.battCapacity = 0
        elif self.battCapacity > self.totalCapacity:
            self.battCapacity = self.totalCapacity

        self.SOC = self.battCapacity/self.totalCapacity
        self.OCVoltage = Battery.soc_ocv_func(self.SOC)*0.001
        self.cellVoltage = self.OCVoltage-self.Current*self.innerResPerCell/self.par
        self.Voltage = self.ser*self.cellVoltage


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    battery1 = Battery()
    battery2 = Battery(parallel=2)
    battery3 = Battery(serial=2)
    t = np.linspace(0, 120000, 1000)
    volt1 = np.zeros_like(t)
    volt2 = np.zeros_like(t)
    volt3 = np.zeros_like(t)
    volt1[0] = battery1.Voltage
    volt2[0] = battery2.Voltage
    volt3[0] = battery3.Voltage
    for i in range(1, len(t)):
        battery1.consume_batt(t[i]-t[i-1], 0.200)
        battery2.consume_batt(t[i]-t[i-1], 0.200)
        battery3.consume_batt(t[i]-t[i-1], 0.200)
        volt1[i] = battery1.Voltage
        volt2[i] = battery2.Voltage
        volt3[i] = battery3.Voltage
    plt.plot(t, volt1, t, volt2, t, volt3)
    plt.show()

from __future__ import annotations

import cantera as ct
import numpy as np
from numba import double, njit

# Type signatures for numba
double1D = double[:]
double2D = double[:, :]
double3D = double[:, :, :]


@njit(double1D(double2D, double1D))
def get_specific_gas_constants_compiled(Y, molecularWeights):
    """
    Function used by the thermoTable class to find the gas constant. This
    function is compiled for speed-up.
        inputs:
            Y: scalar [nX,nSp]
            molecularWeights: species molecular weights [nSp]
        output:
            R: gas constants [nX]
    """
    # find dimensions
    nX = len(Y[:, 0])
    nSp = len(Y[0, :])
    # determine R
    R = np.zeros(nX)
    for iX in range(nX):
        molecularWeight = 0.0
        for iSp in range(nSp):
            molecularWeight += Y[iX, iSp] / molecularWeights[iSp]
        molecularWeight = 1.0 / molecularWeight
        R[iX] = ct.gas_constant / molecularWeight
    return R


# @njit(double1D(double1D, double2D, double1D, double2D, double2D))
def get_cp_compiled(T, Y, TTable, a, b):
    """
    Function used by the thermoTable class to find the constant pressure
    specific heats. This function is compiled for speed-up.
        inputs:
            T: Temperatures [nX]
            Y: scalar [nX,nSp]
            TTable: table of temperatures [nT]
            a: first order coefficient for cp [nT]
            b: zeroth order coefficient for cp [nT]
        output:
            cp: constant pressure specific heat ratios [nX]
    """
    # find dimensions
    nX = len(Y[:, 0])
    nSp = len(Y[0, :])
    # find table extremes
    TMin = TTable[0]
    dT = TTable[1] - TTable[0]  # assume constant steps in table
    TMax = TTable[-1] + dT
    # determine the indices
    indices = np.zeros(nX, dtype=np.int64)
    for iX in range(nX):
        indices[iX] = int((T[iX] - TMin) / dT)
    # determine cp
    cp = np.zeros((nX,))
    for iX in range(nX):
        if (T[iX] < TMin) or (T[iX] > TMax):
            msg = f"Temperature out of bounds: {T[iX]} not in range [{TMin}, {TMax}]"
            raise ValueError(msg)
        index = indices[iX]
        bbar = 0.0
        for iSp in range(nSp):
            bbar += Y[iX, iSp] * (
                a[index, iSp] / 2.0 * (T[iX] + TTable[index]) + b[index, iSp]
            )
        cp[iX] = bbar
    return cp


class ThermoTable:
    """
    This is a class defined to encapsulate the temperature table with the
    relevant methods
    """

    def __init__(self, gas: ct.Solution):
        """
        This method initializes the temperature table. The table uses a
        piecewise linear function for the constant pressure specific heat
        coefficients. The coefficients are selected to retain the exact
        enthalpies at the table points.
        """
        nSp = gas.n_species
        self.TMin = 50.0
        self.dT = 100.0
        self.TMax = 9950.0
        self.T = np.arange(
            self.TMin, self.TMax, self.dT
        )  # vector of temperatures assuming thermal equilibrium between species
        nT = len(self.T)
        self.h = np.zeros((nT, nSp))  # matrix of species enthalpies per temperature
        # cpk = ak*T+bk for T in [Tk,Tk+1], k in {0,1,2,...,nT-1}
        self.a = np.zeros((nT, nSp))  # matrix of species first order coefficients
        self.b = np.zeros((nT, nSp))  # matrix of species zeroth order coefficients
        self.molecularWeights = gas.molecular_weights
        # determine the coefficients
        for kSp, species in enumerate(gas.species()):
            # initialize with actual cp
            cpk = species.thermo.cp(self.T[0]) / self.molecularWeights[kSp]
            hk = species.thermo.h(self.T[0]) / self.molecularWeights[kSp]
            for kT, Tk in enumerate(self.T):
                # compute next
                Tkp1 = Tk + self.dT
                hkp1 = species.thermo.h(Tkp1) / self.molecularWeights[kSp]
                dh = hkp1 - hk
                # store
                self.h[kT, kSp] = hk
                self.a[kT, kSp] = 2.0 / self.dT * (dh / self.dT - cpk)
                self.b[kT, kSp] = cpk - self.a[kT, kSp] * Tk
                # update
                cpk = self.a[kT, kSp] * (Tkp1) + self.b[kT, kSp]
                hk = hkp1

    def get_specific_gas_constants(self, Y):
        """
        This method computes the mixture-specific gas constat
            inputs:
                Y: matrix of mass fractions [n,nSp]
            outputs:
                R: vector of mixture-specific gas constants [n]
        """
        return get_specific_gas_constants_compiled(Y, self.molecularWeights)

    def get_cp(self, T, Y):
        """
        This method computes the constant pressure specific heat as determined
        by Billet and Abgrall (2003) for the double flux method.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        """
        return get_cp_compiled(T, Y, self.T, self.a, self.b)

    def get_frozen_enthalpy(self, T, Y):
        """
        This method computes the enthalpy according to Billet and Abgrall (2003).
        This is the enthalpy that is frozen over the time step
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                cp: vector of constant pressure specific heats
        """
        if any(np.logical_or(self.TMin > T, self.TMax < T)):
            msg = "Temperature not within table"
            raise ValueError(msg)
        nT = len(T)
        indices = [int((Tk - self.TMin) / self.dT) for Tk in T]
        h0 = np.zeros(nT)
        for k, index in enumerate(indices):
            bbar = self.a[index, :] / 2.0 * (T[k] + self.T[index]) + self.b[index, :]
            h0[k] = np.dot(Y[k, :], self.h[index] - bbar * self.T[index])
        return h0

    def get_gamma(self, T, Y):
        """
        This method computes the specific heat ratio, gamma.
            inputs:
                T: vector of temperatures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                gamma: vector of specific heat ratios
        """
        cp = self.get_cp(T, Y)
        R = self.get_specific_gas_constants(Y)
        return cp / (cp - R)

    def get_temperature(self, r, p, Y):
        """
        This method applies the ideal gas law to compute the temperature
            inputs:
                r: vector of densities [n]
                p: vector of pressures [n]
                Y: matrix of mass fractions [n,nSp]
            outputs:
                T: vector of temperatures
        """
        R = self.get_specific_gas_constants(Y)
        return p / (r * R)

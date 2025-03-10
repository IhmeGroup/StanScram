from __future__ import annotations

import numpy as np

# Global variables (parameters) used by the solver
mn = 3  # number of 1D Euler equations


def viscous_flux(domain, rLR, uLR, pLR, YLR):
    """
    This method computes the viscous flux at each interface
        inputs:
            rLR=array containing left and right density states [nLR,nFaces]
            uLR=array containing left and right velocity states [nLR,nFaces]
            pLR=array containing left and right pressure states [nLR,nFaces]
            YLR=array containing left and right scalar states
                [nLR,nFaces,nSp]
        return:
            f=modeled viscous fluxes [nFaces,mn+nSp]
    """
    # get the temperature, pressure, and composition for each cell (including the two ghosts)
    nT = domain.n + 2
    T = np.zeros(nT)
    T[:-1] = domain.get_temperature(rLR[0, :], pLR[0, :], YLR[0, :, :])
    T[[-1]] = domain.get_temperature(
        np.array([rLR[1, -1]]),
        np.array([pLR[1, -1]]),
        np.array([YLR[1, -1, :]]).reshape((1, -1)),
    )
    p, F, Y = np.zeros(nT), np.ones(nT), np.zeros((nT, domain.n_scalars))
    p[:-1], p[-1] = pLR[0, :], pLR[1, -1]
    F[1:-1] = domain.F
    F[0], F[-1] = domain.F[0], domain.F[-1]  # no gradient in F at boundary
    Y[:-1, :], Y[-1, :] = YLR[0, :, :], YLR[1, -1, :]
    mu = domain.get_mu(T, p, Y)
    cp = domain.get_cp(T, Y)
    k = domain.get_lambda_over_cv(T, p, Y) * cp * F
    diff = np.zeros((nT, domain.n_scalars))
    if domain.physics == "FPV":
        diff_ = k / (domain.r * cp)
        diff = diff_.reshape(-1, 1)
    elif domain.physics == "FRC":
        for i, Ti in enumerate(T):
            domain.gas.TP = Ti, p[i]
            if domain.gas.n_species > 1:
                domain.gas.Y = Y[i, :]
            diff[i, :] = domain.gas.mix_diff_coeffs * F[i]
    # compute the gas properties at the face
    viscosity = (mu[1:] + mu[:-1]) / 2.0
    conductivity = (k[1:] + k[:-1]) / 2.0
    diffusivities = (diff[1:, :] + diff[:-1, :]) / 2.0
    r = ((rLR[0, :] + rLR[1, :]) / 2.0).reshape(-1, 1)
    # get the central differences
    dudx = (uLR[1, :] - uLR[0, :]) / domain.dx
    dTdx = (T[1:] - T[:-1]) / domain.dx
    dYdx = (YLR[1, :, :] - YLR[0, :, :]) / domain.dx
    # compute the fluxes
    f = np.zeros((nT - 1, mn + domain.n_scalars))
    f[:, 1] = 4.0 / 3.0 * viscosity * dudx
    f[:, 2] = conductivity * dTdx
    f[:, mn:] = r * diffusivities * dYdx
    return f

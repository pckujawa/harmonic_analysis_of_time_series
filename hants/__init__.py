#!/usr/local/env python
import numpy as np

# Computing diagonal for each row of a 2d array. See: http://stackoverflow.com/q/27214027/2459096
def makediag3d(M):
    b = np.zeros((M.shape[0], M.shape[1] * M.shape[1]))
    b[:, ::M.shape[1] + 1] = M
    return b.reshape(M.shape[0], M.shape[1], M.shape[1])


# Function to apply the Harmonic analysis of time series applied to arrays
# import profilehooks
# @profilehooks.profile(sort='time')
def HANTS(ni, y, nf=3, HiLo='Hi', low=0., high=255, fet=5, delta=0.1):
    """
    ni    = nr. of images (total number of actual samples of the time series)
    nb    = length of the base period, measured in virtual samples
            (days, dekads, months, etc.)
    nf    = number of frequencies to be considered above the zero frequency
    y     = array of input sample values (e.g. NDVI values)
    ts    = array of size ni of time sample indicators
            (indicates virtual sample number relative to the base period);
            numbers in array ts maybe greater than nb
            If no aux file is used (no time samples), we assume ts(i)= i,
            where i=1, ..., ni
    HiLo  = 2-character string indicating rejection of high or low outliers
            select from 'Hi', 'Lo' or 'None'
    low   = valid range minimum
    high  = valid range maximum (values outside the valid range are rejeced
            right away)
    fet   = fit error tolerance (points deviating more than fet from curve
            fit are rejected)
    dod   = degree of overdeterminedness (iteration stops if number of
            points reaches the minimum required for curve fitting, plus
            dod). This is a safety measure
    delta = small positive number (e.g. 0.1) to suppress high amplitudes
    """

    # define some parameters
    nb = ni  #
    ts = np.arange(ni)
    dod = 1  # (2*nf-1)

    # create empty arrays to fill
    mat = np.zeros(shape=(min(2 * nf + 1, ni), ni))

    yr = np.zeros(shape=(y.shape[0], ni))

    # check which setting to set for outlier filtering
    if HiLo == 'Hi':
        sHiLo = -1
    elif HiLo == 'Lo':
        sHiLo = 1
    else:
        sHiLo = 0

    # initiate parameters
    nr = min(2 * nf + 1, ni)  # number of 2*+1 frequecies, or number of input images
    noutmax = ni - nr - dod  # number of input images - number of 2*+1 frequencies - degree of overdeterminedness
    mat[0, :] = 1
    ang = 2 * np.pi * np.arange(nb) / nb
    cs = np.cos(ang)
    sn = np.sin(ang)

    # create some standard sinus and cosinus functions and put in matrix
    i = np.arange(1, nf + 1)
    for j in np.arange(ni):
        index = np.mod(i * ts[j], nb)
        mat[2 * i - 1, j] = cs.take(index)
        mat[2 * i, j] = sn.take(index)

    # repeat the mat array over the number of arrays in y
    # and create arrays with ones with shape y where high and low values are set to 0
    mat = np.tile(mat[None].T, (1, y.shape[0])).T
    p = np.ones_like(y)
    p[(low >= y) | (y > high)] = 0
    nout = np.sum(p == 0, axis=-1)  # count the outliers for each timeseries

    # prepare for while loop
    ready = np.zeros((y.shape[0]), dtype=bool)  # all timeseries set to false
    a = np.arange(ni)
    it = np.nditer(a)

    while ((not it.finished) & (not ready.all())):

        # print '--------*-*-*-*',it.value, '*-*-*-*--------'
        # multipy outliers with timeseries
        za = np.einsum('ijk,ik->ij', mat, p * y)

        # multiply mat with the multiplication of multiply diagonal of p with transpose of mat
        diag = makediag3d(p)
        A = np.einsum('ajk,aki->aji', mat, np.einsum('aij,jka->ajk', diag, mat.T))
        # add delta to supress high amplitudes but not for [0,0]
        A = A + np.tile(np.diag(np.ones(nr))[None].T, (1, y.shape[0])).T * delta
        A[:, 0, 0] = A[:, 0, 0] - delta

        # solve linear matrix equation and define reconstructed timeseries
        zr = np.linalg.solve(A, za)
        yr = np.einsum('ijk,kj->ki', mat.T, zr)

        # calculate error and sort err by index
        err = p * (sHiLo * (yr - y))
        rankVec = np.argsort(err, axis=1, )

        # select maximum error and compute new ready status
        maxerr = np.diag(err.take(rankVec[:, ni - 1], axis=-1))
        ready = (maxerr <= fet) | (nout == noutmax)

        # if ready is still false
        if (not all(ready)):
            i = ni  # i is number of input images
            j = rankVec.take(i - 1, axis=-1)

            p.T[j.T, np.indices(j.shape)] = p.T[j.T, np.indices(j.shape)] * ready.astype(
                int)  #*check
            nout += 1
            i -= 1

        it.iternext()
    return yr


# Compute semi-random time series array with numb standing for number of timeseries
def array_in(numb):
    y = np.array([5.0, 2.0, 10.0, 12.0, 18.0, 23.0, 27.0, 40.0, 60.0, 70.0, 90.0, 160.0, 190.0,
                  210.0, 104.0, 90.0, 170.0, 50.0, 120.0, 60.0, 40.0, 30.0, 28.0, 24.0, 15.0,
                  10.0])
    y = np.tile(y[None].T, (1, numb)).T
    kl = (np.random.randint(2, size=(numb, 26)) *
          np.random.randint(2, size=(numb, 26)) + 1)
    kl[kl == 2] = 0
    y = y * kl
    return y


if __name__ == '__main__':
    y = array_in(10000)
    HANTS(ni=26, y=y, nf=3, HiLo='Lo')

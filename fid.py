from __future__ import division
import pygamma as pg
import numpy as np
import matplotlib.pyplot as plt
import pdb

def hz_to_tesla(hz):
    # for 1H neuclei
    return hz / 42.58

def load_sys(infile='./sys/gaba.sys'):
    sys = pg.spin_system()
    sys.read(infile)
    return sys

def fid(sys, b0=123.23):
    sys.OmegaAdjust(b0)
    H = pg.Hcs(sys) + pg.HJ(sys)
    D = pg.Fm(sys)
    ACQ = pg.acquire1D(pg.gen_op(D), H, 0.1)
    sigma = pg.sigma_eq(sys)
    sigma0 = pg.Ixpuls(sys, sigma, 90.0)
    mx = pg.TTable1D(ACQ.table(sigma0))
    return mx


def binning(mx, b0=123.23, high_ppm=-3.5, low_ppm=-1.5, dt=5e-4, npts=4096, linewidth=1):
    # add some broadening and decay to the model
    mx.broaden(linewidth)
    mx.Iscale(0.1, 0)

    ADC = []
    acq = mx.T(npts, dt)
    for ii in range(0, acq.pts()):
        ADC.extend([acq.get(ii).real() + (1j * acq.get(ii).imag())])

    FFT = []
    acq = mx.F(npts, b0 * high_ppm, b0 * low_ppm)
    for ii in range(0, acq.pts()):
        FFT.extend([acq.get(ii).real() + (1j * acq.get(ii).imag())])

    ADC = normalise_complex_signal(np.array(ADC))
    FFT = normalise_complex_signal(np.array(FFT))
    nu = np.linspace(high_ppm, low_ppm, npts)

    return ADC, FFT, nu


def plot_fft(sys, ADC, FFT, nu):
    fig, axes = plt.subplots(2, 2, sharex='col', sharey=True,  figsize=(19.2, 10.8), dpi=85)
    # pulse sequence name, system name and magnetic field strength
    plt.suptitle('FID: %s B0: %.2fT' %(sys.name(), hz_to_tesla(sys.Omega())))

    # Plot the reported FFT from PyGamma
    axes[0, 0].set_title('FFT Magnitude spectrum')
    axes[0, 0].plot(nu, np.abs(FFT)/max(np.abs(FFT)))
    axes[0, 0].set_xlim([min(nu), max(nu)])

    axes[1, 0].set_title('FFT Real & Imag. spectrum')
    axes[1, 0].plot(nu, np.real(FFT), label='Real')
    axes[1, 0].plot(nu, np.imag(FFT), label='Imag')
    axes[1, 0].set_xlabel('PPM')
    axes[1, 0].legend(['Real', 'Imaginary'])

    # Plot the ADC
    axes[0, 1].set_title('ADC Real')
    axes[0, 1].plot(np.linspace(0, len(ADC), len(ADC)), np.real(ADC))
    axes[0, 1].set_xlim([0, len(ADC)])

    axes[1, 1].set_title('ADC Imaginary')
    axes[1, 1].plot(np.linspace(0, len(ADC), len(ADC)),np.imag(ADC))
    axes[1, 1].set_xlabel('T')

    plt.show()


def normalise_complex_signal(signal):
    return signal / np.max([np.abs(np.real(signal)), np.abs(np.imag(signal))])


if __name__ == '__main__':
    sys = sys=load_sys()
    mx = fid(sys)
    ADC, FFT, nu = binning(mx)
    plot_fft(sys, ADC, FFT, nu)

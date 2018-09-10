from __future__ import division
import pygamma as pg
import numpy as np
import matplotlib.pyplot as plt


def fid(infile='./sys/gaba.sys', b0=123.23):
    sys = pg.spin_system()
    sys.read(infile)
    sys.OmegaAdjust(b0)
    H = pg.Hcs(sys) + pg.HJ(sys)
    D = pg.Fm(sys)
    ACQ = pg.acquire1D(pg.gen_op(D), H, 0.1)
    sigma = pg.sigma_eq(sys)
    sigma0 = pg.Ixpuls(sys, sigma, 90.0)
    mx = pg.TTable1D(ACQ.table(sigma0))
    return mx


def binning(mx, b0=123.23, high_ppm=-5, low_ppm=-1, dt=5e-4, npts=4096, linewidth=1):
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
    ADC_FFT = normalise_complex_signal(np.fft.fftshift(np.fft.fft(ADC)))
    ADC_nu = -(np.linspace(-1, 1, npts) * ((1 / dt) / 2)) / b0
    FFT = normalise_complex_signal(np.array(FFT))
    FFT_nu = np.linspace(high_ppm, low_ppm, npts)

    return ADC, ADC_FFT, ADC_nu, FFT, FFT_nu


def plot_fft(ADC, ADC_FFT, ADC_nu, FFT, FFT_nu, high_ppm=-5, low_ppm=-1):
    fig, axes = plt.subplots(2, 3, sharex='col', sharey=True,  figsize=(19.2, 10.8), dpi=85)
    # pulse sequence name
    plt.suptitle('FID')

    # Plot the reported FFT from PyGamma
    axes[0, 0].set_title('FFT Magnitude spectrum')
    axes[0, 0].plot(FFT_nu, np.abs(FFT)/max(np.abs(FFT)))
    axes[0, 0].set_xlim([high_ppm, low_ppm])

    axes[1, 0].set_title('FFT Real & Imag. spectrum')
    axes[1, 0].plot(FFT_nu, np.real(FFT), label='Real')
    axes[1, 0].plot(FFT_nu, np.imag(FFT), label='Imag')
    axes[1, 0].set_xlabel('PPM')

    # Plot the FFT(ADC)
    axes[0, 1].set_title('ADC Magnitude spectrum')
    axes[0, 1].plot(ADC_nu, np.abs(ADC_FFT)/max(np.abs(ADC_FFT)))
    axes[0, 1].set_xlim([high_ppm, low_ppm])
    axes[1, 1].legend(loc='upper left')

    axes[1, 1].set_title('FFT Real & Imag. spectrum')
    axes[1, 1].plot(ADC_nu, np.real(ADC_FFT), label='Real')
    axes[1, 1].plot(ADC_nu, np.imag(ADC_FFT), label='Imag')
    axes[1, 1].set_xlabel('PPM')

    # Plot the ADC
    axes[0, 2].set_title('ADC Real')
    axes[0, 2].plot(np.linspace(0, len(ADC), len(ADC)), np.real(ADC))
    axes[0, 2].set_xlim([0, len(ADC)])

    axes[1, 2].set_title('ADC Imaginary')
    axes[1, 2].plot(np.linspace(0, len(ADC), len(ADC)),np.imag(ADC))
    axes[1, 2].set_xlabel('T')

    plt.show()


def normalise_complex_signal(signal):
    return signal / np.max([np.abs(np.real(signal)), np.abs(np.imag(signal))])


if __name__ == '__main__':
    mx = fid()
    ADC, ADC_FFT, ADC_nu, FFT, FFT_nu = binning(mx)
    plot_fft(ADC, ADC_FFT, ADC_nu, FFT, FFT_nu)

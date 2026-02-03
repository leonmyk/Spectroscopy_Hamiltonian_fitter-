import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob
import h5py
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import time
from datetime import *
from matplotlib.collections import LineCollection
import os, sys
from functions import load_h5_to_dic
from functions import complex_ramsey_fit

dirpath = 'Z:/SPIN-001/Run2 [BFp4]/RamanManuRamsey9o2_interleaved/'
dirpath = 'Z:/SPIN-001/Run2 [BFp4]/RamanEchoJaimeTrixV3/'
# dirpath = 'Z:/SPIN-001/Run2 [BFp4]/RamanRamsey9o2_interleaved/'
filename = '20260126173853__RamanManuRamsey9o2_interleaved.hdf5' 
filename = '20260202121414__RamanEchoJaimeTrixV3.hdf5' 
# filename = '20260119184019__RamanRamsey9o2_interleaved.hdf5'
signal = load_h5_to_dic(dirpath + filename)[0]

def plot(
    data_click,
    N_RO,
    wait_multiplier,
    threshold,
    transition,
    time,
    plot_guess: bool = False,
    nuclear_detuning : int = None,
    artificial_detuning: int = None,
    drive_freq: int = None,
    plot: bool = True,
    total_detuning: float = None,
    
):

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data_I = ((data_click[:, :, 0, 0] > threshold)).mean(0)
    data_Q = ((data_click[:, :, 1, 0] > threshold)).mean(0)
    if total_artificial_detuning == None:   
        total_artificial_detuning = nuclear_detuning + artificial_detuning
    deviations = []

    # time = (time/4/wait_multiplier).astype(int)*4*wait_multiplier/1e6

    # Plot the data and fitted function

    complex_Ramsey_signal = data_I + 1j * data_Q
    fft_data = abs(np.fft.fft(complex_Ramsey_signal-complex_Ramsey_signal.mean()))
    freqfft = np.fft.fftfreq(len(time), time[1] - time[0])
    freqmax = freqfft[np.argmax((fft_data))]
    Z = np.concatenate([data_I, data_Q])
    # Initial parameter guesses for the curve fit
    # complex_ramsey_fit(t,f,T,phi,A,B)
    guess = [
        freqmax, # Frequency in Hz
        5,  # Decay time constant T [ms]
        -1* np.pi,
        (np.max(data_I) - np.min(data_I)) / 2,  # Amplitude
        (np.average(data_I) + np.average(data_Q)) / 2,  # offset
    ]
    try:
        # Perform curve fitting with initial guesses
        params, params_covariance = curve_fit(complex_ramsey_fit, time, Z, p0=guess)
    except Exception as e:
        print("Fit failed:", e)
        params = guess  # Use initial guess if fit fails
        params_covariance = []
    # Extract fit parameters for display
    f1_fit, T_fit, phi1_fit, A1_fit, offset_fit = params
    
    
    std = Bootstrap_analysis(time, data_I,data_Q, params)
    print(f"Transition {transition}: Fitted Frequency = {1e3 * f1_fit:.2f} Hz,, Std = {std:.4f} kHz")
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(321)

    # Generate a smooth line to overlay the fitted function
    x = np.linspace(time[0], time[-1], 1001)
    
    if plot:
        if plot_guess:
            plt.plot(
                x,
                complex_ramsey_fit(x, *guess)[:len(x)],
                label="Guess: Dual Cosine with Decay",
                linestyle="--",
                color=colors[transition],
                linewidth=2,
            )
        plt.plot(
            x,
            complex_ramsey_fit(x, *params)[:len(x)],  # Ensure `params` matches the expected parameter count
            color=colors[transition],
            label = r"$f_{ground}$ =" +
            f"{1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz"
        )
        plt.plot(
            x,
            complex_ramsey_fit(x, *params)[len(x):],  # Ensure `params` matches the expected parameter count
            '--',
            color='black',
            alpha = 0.2
        )
        plt.plot(time, data_I, "o", label=transition, color=colors[transition], markeredgecolor = 'black',)
        plt.plot(time, data_Q, "o", label='Ramsey quadrature', color='black', markeredgecolor = 'black', alpha = 0.2)
        plt.ylabel("Population")
        plt.ylim(0, 1)
        plt.legend()
        plt.xlabel("Ramsey time (ms)")
        
        # Summary of fitted parameters for display
        fitted_info = (
            f"Frequency 1 = {1e3 * f1_fit:.1f} Hz\n"
            # f"Frequency 2 = {1e3 * f2_fit:.1f} Hz\n"
            f"$T_2^*$ = {T_fit:.2f} ms\n"
            rf"Artificial detuning $\delta$ = {1e9 * (artificial_detuning+nuclear_detuning):.1f} Hz = {1e9 * nuclear_detuning:.1f} Hz + {1e9 * artificial_detuning:.1f} Hz"
            '\n'
            rf"$\omega_\downarrow$ = $\delta$ + f + $f_d$ =" +
            f"{1e9 * total_artificial_detuning + 1e3 * f1_fit:.1f} Hz + {drive_freq:.1f} Hz" +
            f"= {1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz\n"
            f"Averages = {data_click.shape[0]}" + f" (meas time = {meas_time:.1f} h)"

        )

        # Plot mean clicks over time
        plt.subplot(323)

        plt.plot(time, data_click[:,:,0,0].mean(0), label=f"{transition}", color=colors[transition])
        plt.legend()
        plt.xlabel("Ramsey time (ms)")
        plt.title("Mean Clicks Over Time")
        
        plt.subplot(322)
        complex_Ramsey_signal = data_I + 1j * data_Q
        fft_data = np.fft.fft(complex_Ramsey_signal-complex_Ramsey_signal.mean())
        freqfft = np.fft.fftfreq(len(time), time[1] - time[0])
        freqmax = freqfft[np.argmax(abs(fft_data))]
        plt.plot(np.fft.fftshift(freqfft), np.fft.fftshift(fft_data))
        plt.axvline(freqmax, ls = '--', color = 'black')
        plt.xlabel(r'Frequency (kHz)')
        plt.ylabel('Fourier Transform Signal')

        # Histogram of clicks
        plt.subplot(324)

        plt.hist(
            np.concatenate(data_click[:,:,0,0]), label=f"{transition}", bins=np.arange(0, N_RO / 2, 1), alpha=0.5, color=colors[transition]
        )
        plt.axvline(threshold)
        plt.legend()
        plt.xlabel("Number of occurrences")
        plt.title("Histogram of Clicks")

        # Experiment metadata in the title
        plt.suptitle(
            f"{fitted_info}"
        )

        # Adjust layout for the title and save the figure
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return f1_fit, std

def Bootstrap_analysis(time,x,y,guess,plot=False):
    n_bootstrap = 1000
    f1_bootstrapped = []
    N = len(y)
    for _ in range(n_bootstrap):
        indices = np.random.randint(0, N, N)
        time_sampled = time[indices]
        x_sampled = x[indices]
        y_sampled = y[indices]
        z_sampled = np.concatenate([x_sampled, y_sampled])
        try:
            params, _ = curve_fit(complex_ramsey_fit, time_sampled, z_sampled, p0=guess)
            f1_bootstrapped.append(params[0])
        except:
            continue
    f1_bootstrapped = np.array(f1_bootstrapped)
    f1_means = np.mean(f1_bootstrapped)
    f1_std = np.std(f1_bootstrapped)
    if plot:
        h = 3.5 * f1_std / (n_bootstrap ** (1/3))
        bins = int((f1_bootstrapped.max() - f1_bootstrapped.min()) / h)
        plt.hist(f1_bootstrapped, bins=30, alpha=0.7, color='blue')
        plt.title(rf'Bootstrap Analysis of Fitted Frequency std is {f1_std:.4f} KHz')
        plt.xlabel('Fitted Frequency (KHz)')
        plt.ylabel('Occurrences')
        plt.axvline(f1_means, color='red', linestyle='--')
        plt.show()
    return f1_std
     
    

nuclear_frequency  = [7_559_790,   6_893_756,    6_226_376,  5_556_583,    4_882_806,  4_200_214,    3_500_709,    2_768_085,  1_813_939]

end_transition = 1 #signal["end_transition"]
f1_fits = np.zeros(end_transition)
stds = np.zeros(end_transition)
print(signal.keys())

for transition in range(end_transition):






    data_click = signal["data_click"] #[:, transition, :, :, :]
    threshold = signal["threshold"]
    meas_time = signal["time"]
    meas_time_hours = signal["meas_time_hours"]
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    nuclear_detuning = signal["nuclear_detuning"][transition]
    artificial_detuning = signal["artificial_detuning"][transition]
    # wait_multiplier = signal["wait_multiplier"]
    N_RO = signal["N_RO"]


    start_time=10000
    end_time=3_010_000
    n_steps=41

    # start_time = 10000
    # end_time = 1000000000
    # n_steps = 41

    ramsey_times = np.linspace(start_time,end_time,n_steps)*1e-6  # Convert to ms
    f1_fits[transition], stds[transition] = plot(data_click, N_RO, 1, threshold, transition, ramsey_times, plot_guess=False,plot = True, nuclear_detuning=nuclear_detuning, artificial_detuning=artificial_detuning, drive_freq=nuclear_frequency[transition])


print("Fitted frequencies (kHz):", f1_fits)
print("Fitted frequencies std (kHz):", stds)
import matplotlib.pyplot as plt
import numpy as np
import time
from pathlib import Path
import glob
import h5py
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
from functions import complex_ramsey_fit
from functions import load_h5_to_dic
from functions import complex_ramsey_fit_n


def Chunk_Data(signal,chunk_size = None,nb_chunks = None):

    chunk_size = 136
    chunked_signals = []

    # =========================
    # Raw data
    # =========================
    data_click = signal["data_click"] # (N_avg, N_time, I/Q, ...)

    n_avg, n_time = data_click.shape[:2]
    if nb_chunks == None:
        n_chunks = n_avg // chunk_size
    elif chunk_size == None:
        chunk_size = n_avg//n_chunks
    else :
        print("need chunk size or number of chunks at least...")
    

    for k in range(n_chunks):
        sl = slice(k * chunk_size, (k + 1) * chunk_size)
        signal_to_add = signal.copy()
        signal_to_add["data_click"] = signal["data_click"][sl,:,:,:]
        signal_to_add["meas_time_hours"] = signal["meas_time_hours"]/n_chunks
        signal_to_add["iteration"] = signal["iteration"]/n_chunks
        
        chunked_signals.append(signal_to_add)
        
    return chunked_signals

def plot(
    data_click,
    N_RO,
    threshold,
    transition,
    time_,
    plot_guess: bool = False,
    nuclear_detuning : int = 0,
    artificial_detuning: int = 0,
    drive_freq: int = None,
    plot: bool = True,
    meas_time: float = 0,
    plot_bootstrap:bool = False,
    decay_time: float = 3000
    
):
    
    
    data_I_before_averaging = ((data_click[:, :, 0, 0] > threshold))
    data_Q_before_averaging = ((data_click[:, :, 1, 0] > threshold))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data_I = data_I_before_averaging.mean(0)
    data_Q = data_Q_before_averaging.mean(0)


    total_artificial_detuning = nuclear_detuning + artificial_detuning


    # time = (time/4/wait_multiplier).astype(int)*4*wait_multiplier/1e6

    # Plot the data and fitted function

    complex_Ramsey_signal = data_I + 1j * data_Q
    fft_data = abs(np.fft.fft(complex_Ramsey_signal-complex_Ramsey_signal.mean()))
    freqfft = np.fft.fftfreq(len(time_), time_[1] - time_[0])
    freqmax = freqfft[np.argmax((fft_data))]
    Z = np.concatenate([data_I, data_Q])
    # Initial parameter guesses for the curve fit
    # complex_ramsey_fit(t,f,T,phi,A,B)
    guess = [
        freqmax, # Frequency in Hz
        decay_time,  # Decay time constant T [ms]
        -1* np.pi,
        (np.max(data_I) - np.min(data_I)) / 2,  # Amplitude
        (np.average(data_I) + np.average(data_Q)) / 2,  # offset
    ]
    try:
        # Perform curve fitting with initial guesses
        params, params_covariance = curve_fit(complex_ramsey_fit, time_, Z, p0=guess)
    except Exception as e:
        print("Fit failed:", e)
        params = guess  # Use initial guess if fit fails
        params_covariance = []
    # Extract fit parameters for display
    f1_fit, T_fit, phi1_fit, A1_fit, offset_fit = params
    
    std = Bootstrap_analysis(time_, data_I_before_averaging,data_Q_before_averaging, params,plot=plot_bootstrap)
    print(f"Transition {transition}: Fitted Frequency = {1e3 * f1_fit:.2f} Hz,, Std = {std*1e3:.4f} Hz")
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(321)

    # Generate a smooth line to overlay the fitted function
    x = np.linspace(time_[0], time_[-1], 1001)
    
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
            rf"{1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz"
        )
        plt.plot(
            x,
            complex_ramsey_fit(x, *params)[len(x):],  # Ensure `params` matches the expected parameter count
            '--',
            color='black',
            alpha = 0.2
        )
        plt.plot(time_, data_I, "o", label=transition, color=colors[transition], markeredgecolor = 'black',)
        plt.plot(time_, data_Q, "o", label='Ramsey quadrature', color='black', markeredgecolor = 'black', alpha = 0.2)
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

        plt.plot(time_, data_click[:,:,0,0].mean(0), label=f"{transition}", color=colors[transition])
        plt.legend()
        plt.xlabel("Ramsey time (ms)")
        plt.title("Mean Clicks Over Time")
        
        plt.subplot(322)
        complex_Ramsey_signal = data_I + 1j * data_Q
        fft_data = np.fft.fft(complex_Ramsey_signal-complex_Ramsey_signal.mean())
        freqfft = np.fft.fftfreq(len(time_), time_[1] - time_[0])
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

def Bootstrap_analysis(time_,x,y,guess,plot=False):
    n_bootstrap = 1000
    f1_bootstrapped = []
    N = y.shape[0]
    fig,axs = plt.subplots(2)
    fig.tight_layout()
    time_dense = np.linspace(time_[0],time_[-1],1000)
    x_avg = x.mean(0)

    
    for _ in range(n_bootstrap):
        indices = np.random.choice(N, N ,replace = True)
        x_sampled = x[indices,:].mean(0)
        y_sampled = y[indices,:].mean(0)
        z_sampled = np.concatenate([x_sampled, y_sampled])

        try:
            params, _ = curve_fit(complex_ramsey_fit, time_, z_sampled, p0=guess)
            #delta_f, delta_t, delta_phi1, delta_A, delta_offset
            error = np.abs(params-guess)
            
            max_error = [1/(time_[-1]-time_[0]),guess[1]*0.3,np.inf,guess[3]*0.5,guess[4]*0.5]
            max_error = [1/(time_[-1]-time_[0]),np.inf,np.inf,np.inf,np.inf]
            
            if (np.sum(error>max_error) == 0):
            
                f1_bootstrapped.append(params[0]) 
                axs[0].plot(time_dense,complex_ramsey_fit(time_dense,*params)[:len(time_dense)], alpha=0.1, color='black')
                axs[0].plot(time_,x_sampled,'x', alpha=0.5, color='black')
            
            else :
            
                axs[0].plot(time_dense,complex_ramsey_fit(time_dense,*params)[:len(time_dense)], alpha=0.1, color='blue')

        except:
            print('fit failed this is bad..')
            
            continue
        


    f1_bootstrapped = np.array(f1_bootstrapped)
    f1_means = np.mean(f1_bootstrapped)
    f1_std = np.std(f1_bootstrapped)
    axs[0].set_title(rf'Bootstrap plot of signal std is {f1_std*1e3:.4f} Hz')

    if plot:
        axs[0].set_xlabel('Time (ms)')
        axs[0].set_ylabel('population')
        axs[0].set_ylim(0,1)
        axs[0].plot(time_dense,complex_ramsey_fit(time_dense,*guess)[:len(time_dense)], color='red')
        axs[0].plot(time_,x_avg,'o' ,color='red')


        
        
        # h = 3.5 * f1_std / (n_bootstrap ** (1/3))
        # bins = int((f1_bootstrapped.max() - f1_bootstrapped.min()) / h)
        axs[1].hist(f1_bootstrapped, bins = 40)
        axs[1].set_title(rf'Bootstrap Analysis of Fitted Frequency std is {f1_std*1e3:.4f} Hz')
        axs[1].set_xlabel('Fitted Frequency (KHz)')
        axs[1].set_ylabel('Occurrences')
        axs[1].axvline(f1_means, color='red', linestyle='--')
        
        plt.show()
    return f1_std

def plot_chunked_averages(threshold, transition, n, data_click, time_, meas_time,
                          n_freq=1,decay_time = 3,
                          ylim=(-0.2, 1.2), figsize=(12,10)):
    """
    signal: your data dict, uses signal["data_click"][...,0]
    n:      chunk size
    n_freq: number of frequencies to pick and fit (default 1)
    """
    data_click = data_click[...,0]
    print(data_click.shape)
    Ntotal, Nt = data_click.shape[:2]
    n_chunks = Ntotal // n
    if n_chunks == 0:
        raise ValueError(f"Not enough data ({Ntotal}) for a single chunk of size {n}")
    time_per_chunk = round(meas_time / n_chunks, 3)
    
    # set up a grid: n_chunks rows, 2 cols (time-domain & FFT)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(n_chunks, 2, width_ratios=[3,1], hspace=0.4)
    axs = gs.subplots(sharex=False, sharey=False)
    
    freq_evol = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    
    data_I_before_averaging = ((data_click[:, :, 0] > threshold))
    data_Q_before_averaging = ((data_click[:, :, 1] > threshold))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    data_I = data_I_before_averaging.mean(0)
    data_Q = data_Q_before_averaging.mean(0)
    
    complex_Ramsey_signal = data_I + 1j * data_Q
    fft_data = abs(np.fft.fft(complex_Ramsey_signal-complex_Ramsey_signal.mean()))
    freqfft = np.fft.fftfreq(len(time_), time_[1] - time_[0])
    freqmax = freqfft[np.argmax((fft_data))]
    Z = np.concatenate([data_I, data_Q])
    # Initial parameter guesses for the curve fit
    # complex_ramsey_fit(t,f,T,phi,A,B)
    guess = [
        freqmax, # Frequency in Hz
        decay_time,  # Decay time constant T [ms]
        -1* np.pi,
        (np.max(data_I) - np.min(data_I)) / 2,  # Amplitude
        (np.average(data_I) + np.average(data_Q)) / 2,  # offset
    ]
    try:
        # Perform curve fitting with initial guesses
        global_params, params_covariance = curve_fit(complex_ramsey_fit, time_, Z, p0=guess)
    except Exception as e:
        print("overall Fit failed:", e)
        global_params = guess  # Use initial guess if fit fails
        params_covariance = []
    
    for i in range(n_chunks):
        # extract chunk
        block = data_click[i*n:(i+1)*n, :, ...]
        avg_real = (block > threshold).mean(axis=0)[:,0]
        avg_imag = (block > threshold).mean(axis=0)[:,1]
        Z = avg_real + 1j*avg_imag
        
        # FFT
        fft_data = np.fft.fft(Z - Z.mean())
        freqfft  = np.fft.fftfreq(len(time_), time_[1] - time_[0])
        
        # pick frequencies
        mag    = np.abs(fft_data)
        pos    = freqfft > 0
        fpos   = freqfft[pos]
        mpos   = mag[pos]
        
        if n_freq == 1:
            idx_peak = np.argmax(mpos)
            top_idx = [np.array(idx_peak)]
            freqs_peak = np.array([fpos[idx_peak]])
        else:
            peaks, _   = find_peaks(mpos)
            # take top n_freq
            top_idx    = peaks[np.argsort(mpos[peaks])[-n_freq:]]
            freqs_peak = np.sort(fpos[top_idx])
        
        # phase guesses from the FFT
        pos_inds = np.where(freqfft > 0)[0]
        phase_guesses = [
            np.angle(fft_data[pos_inds[idx]])
            for idx in top_idx
        ]

        # fit
        try:
            p_opt, _ = curve_fit(
                complex_ramsey_fit,
                time_,
                np.concatenate([avg_real, avg_imag]),
                p0=global_params
            )
        except Exception as e:
            print(f"Fit failed in chunk {i}: {e}")
            p_opt = np.array(guess)
        
        # store fitted frequencies (in Hz)
        freq_evol.append(p_opt[:n_freq]*1e3)
        
        # --- plot time-domain + fit ---
        ax_t = axs[i,0]
        t_fit = np.linspace(time_[0], time_[-1], 1000)
        fit_vals = complex_ramsey_fit(t_fit, *p_opt)
        fit_real = fit_vals[:len(t_fit)]
        fit_imag = fit_vals[len(t_fit):]
        
        ax_t.plot(t_fit, fit_real, lw = 3, color=colors[transition], label="fit Re")
        ax_t.plot(t_fit, fit_imag, '--', lw = 3, color=colors[transition], alpha=0.6, label="fit Im")
        ax_t.plot(time_, avg_real, 'o', color=colors[transition], markeredgecolor='k',
                  label=f"{i*time_per_chunk}h – {(i+1)*time_per_chunk}h")
        ax_t.plot(time_, avg_imag, 'o', color='k', alpha=0.3)
        ax_t.set_ylim(*ylim)
        # annotate T2* on the time-domain axis in the top-left corner
        T_fit = p_opt[n_freq]
        ax_t.text(
            0.55, 0.95,
            f"$T_2^* = {T_fit:.1f}\\,\\mathrm{{ms}}$",
            transform=ax_t.transAxes,
            ha="left",
            va="top"
        )
        # ax_t.legend(fontsize = 10, loc="upper right")
        if i == n_chunks-1:
            ax_t.set_xlabel("Ramsey time (ms)")
        ax_t.set_ylabel("Population")
        
        # --- plot FFT in dB or linear ---
        ax_f = axs[i,1]
        ax_f.plot(fpos, mpos, '-')
        for fp in freqs_peak:
            ax_f.axvline(fp, color='r', linestyle='--')
            ax_f.text(fp, ax_f.get_ylim()[1]*0.8,
                      f"{fp*1e3:.2f} Hz", rotation=90,
                      va='top', fontsize=8)
        if i == n_chunks-1:
            ax_f.set_xlabel("Frequency (Hz)")
        ax_f.set_ylabel("|FFT|")
        ax_f.set_yticks([])
    
    plt.tight_layout()
    plt.show()
    
    # plot evolution of all fitted frequencies
    freq_evol = np.array(freq_evol)  # shape (n_chunks, n_freq)
    time_axis = np.arange(n_chunks) * time_per_chunk
    fig2, ax2 = plt.subplots(figsize=(6,4))
    for j in range(n_freq):
        ax2.plot(
        time_axis,
        freq_evol[:, j] - freq_evol[:, j].mean(),
        '-o',
        markeredgecolor='k',
        label=f"{freq_evol[:, j].mean():.2f} Hz"
    )

    ax2.set_xlabel("Measurement Time (h)")
    ax2.set_ylabel(r"$\Delta f$ (Hz)")
    ax2.legend(loc="upper right")
    plt.tight_layout()     
    

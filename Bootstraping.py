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
from functions import complex_ramsey_gaussian_fit
from scipy.signal import find_peaks
from matplotlib.gridspec import GridSpec

def lorentz(delta, delta0, kappa, a, b):
    """
    Lorentzian distribution, the one which is practical for a fit.
    To be removed and abstracted in fitting classes.
    """
    return a / (1 + (delta - delta0) ** 2 / (kappa / 2) ** 2) + b


def chevron(x, cen, wid, t, A, B):
    return A * wid**2/((x-cen)**2+wid**2) * np.sin(2*np.pi*t*np.sqrt((x-cen)**2+wid**2)/2)**2 + B

def load_h5_to_dic(fullpath):
    with h5py.File(fullpath, 'r') as file:
        main_keys = list(file["/"].keys())
        data_vector = {}
        if isinstance(file[main_keys[0]], h5py.Dataset):
            #datasets_keys_list = [main_keys]
            for key in main_keys:
                data_vector[key]=file[key][()]
            return data_vector, main_keys
        else:
            datasets_keys_list = {}
            for j, key in enumerate(main_keys):
                datasets_keys = list(file[key].keys())
                datasets_keys_list[key]=list(file[key].keys())
                data_vector[key]={}
                for d_key in datasets_keys:
                    data_vector[key][d_key]=file[key][d_key][()]
            return data_vector, datasets_keys_list 



def lorentz(delta, delta0, kappa, a, b):
    """
    Lorentzian distribution, the one which is practical for a fit.
    To be removed and abstracted in fitting classes.
    """
    return a / (1 + (delta - delta0) ** 2 / (kappa / 2) ** 2) + b

def make_multi_lorentz(n):
    # params = [x01,g1,A1,  x02,g2,A2, ..., x0n,gn,An,  C]
    def f(x, *params):
        *p, C = params
        y = np.zeros_like(x, dtype=float)
        for i in range(n):
            x0, g, A = p[3*i:3*i+3]
            y += lorentz(x, x0, g, A, 0.0)
        return y + C
    return f


def plot_2d_sweep(data, x=[], y=[], xlabel='', ylabel='', clabel='', title='', xtick='auto', ytick='auto',
                  centre=None, vmin=None, vmax=None, cmap=sns.diverging_palette(240, 10, n=361),
                  horizontal_ticks=False, xticks_rotation=None, yticks_rotation=None, fontsize=None,
                  xlim=None, ylim=None, x_line=None, y_line=None, show = False):
    
    # xlim, ylim, x_line, y_line take indices instead of actual values, BUG???
    
    """
    Generic plotting function for 2D datasets with optional vertical and horizontal lines
    """
    
    if len(x) == 0: 
        x = np.linspace(0, data.shape[1]-1, data.shape[1], dtype=int)
    if len(y) == 0: 
        y = np.linspace(0, data.shape[0]-1, data.shape[0], dtype=int)
    
    fieldsweep_df = pd.DataFrame(data=np.flip(data, axis=0), index=np.flip(y, axis=0), columns=x)
    
    ax = sns.heatmap(fieldsweep_df, xticklabels=xtick, yticklabels=ytick, cmap=cmap, center=centre, vmin=vmin, vmax=vmax)
    ax.collections[0].colorbar.set_label(clabel, fontsize=fontsize)
    ax.collections[0].colorbar.ax.tick_params(labelsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    
    if horizontal_ticks:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    if xticks_rotation is not None:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=xticks_rotation)
    
    if yticks_rotation is not None:
        ax.set_yticklabels(ax.get_yticklabels(), rotation=yticks_rotation)
    
    ax.set_title(title, fontsize=fontsize)
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    if ylim is not None:
        ax.set_ylim(ylim)
    
    if x_line is not None:
        ax.axvline(x=x_line, color='black', linestyle='--', linewidth=1)
    
    if y_line is not None:
        ax.axhline(y=y_line, color='black', linestyle='--', linewidth=1)
    if show:
        plt.show()
        
        
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
    decay_time: float = 3000,
    fit_func = complex_ramsey_gaussian_fit
    
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
        -1 * np.pi,
        (np.max(data_I) - np.min(data_I)) / 2,  # Amplitude
        (np.average(data_I) + np.average(data_Q)) / 2,  # offset
    ]
    try:
        # Perform curve fitting with initial guesses
        params, params_covariance = curve_fit(fit_func, time_, Z, p0=guess)
    except Exception as e:
        print("Fit failed:", e)
        params = guess  # Use initial guess if fit fails
        params_covariance = []
    # Extract fit parameters for display
    f1_fit, T_fit, phi1_fit, A1_fit, offset_fit = params
    
    std = Bootstrap_analysis(time_, data_I_before_averaging,data_Q_before_averaging, params,plot=plot_bootstrap, fit_func = fit_func)
    print(f"Transition {transition}: Fitted Frequency = {1e3 * f1_fit:.2f} Hz,, Std = {std*1e3:.4f} Hz")
    fig = plt.figure(figsize=(15, 15))
    plt.subplot(321)

    # Generate a smooth line to overlay the fitted function
    x = np.linspace(time_[0], time_[-1], 1001)
    
    if plot:
        if plot_guess:
            plt.plot(
                x,
                fit_func(x, *guess)[:len(x)],
                label="Guess: Dual Cosine with Decay",
                linestyle="--",
                color=colors[transition],
                linewidth=2,
            )
        plt.plot(
            x,
            fit_func(x, *params)[:len(x)],  # Ensure `params` matches the expected parameter count
            color=colors[transition],
            label = r"$f_{ground}$ =" +
            rf"{1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz"
        )
        plt.plot(
            x,
            fit_func(x, *params)[len(x):],  # Ensure `params` matches the expected parameter count
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


def plotsb(
    signal: dict,
    amp,
    freqs,
):
    clicks = signal["clicks"]
    iteration = signal["iteration"]
    # delta_freq = signal["delta_freq"]
    freq_list = np.array(freqs) * 1e-3
    counts = clicks.sum(2)
    p_down = (counts[:, :, 1] - counts[:, :, 0] < 0).mean(0)
    p_up = (counts[:, :, 1] - counts[:, :, 0] >= 0).mean(0)
    if p_down.mean() < 0.5:
        p_fit = p_down
    else:
        p_fit = p_up
    guess = [
        freq_list[np.argmax(p_fit)],
        max(p_fit) - min(p_fit),
        (freq_list[-1] - freq_list[0]) / 5,
        min(p_fit),
    ]
    try:
        est, std = curve_fit(lorentz, freq_list,p_fit,guess)
    except Exception as e:
        print("fit failed, ", e)
        est, std, fine, data_fit = guess, guess, [], []


    fine = np.linspace(freq_list[0],freq_list[-1],1000)
    data_fit = lorentz(fine,*est)


    state = [r"$|\Downarrow>$", r"$|\Uparrow>$"]
    fig = plt.figure()
    plt.errorbar(
        freq_list,
        p_down,
        np.sqrt(p_down * (1 - p_down) / iteration),
        linewidth=2,
        label=state[0],
    )
    plt.errorbar(
        freq_list,
        p_up,
        np.sqrt(p_up * (1 - p_up) / iteration),
        linewidth=2,
        label=state[1],
    )
    plt.plot(fine, data_fit, "--", color="black", alpha=0.6)
    # plt.plot(fine, 1 - data_fit, "--", color="black", alpha=0.6)
    plt.title(
         "\n"
        + f"Peak frequency: {est[0]:.1f} kHz"
        + f"\n amp = {amp}"
        + "\n",
        # + f"averages: {iteration:.0f}",
        fontweight="bold",
    )
    plt.xlabel(r"Frequency (kHz)")
    plt.ylabel("Population")
    plt.ylim([-0.1, 1.1])
    plt.tight_layout()
    plt.legend()
    plt.show()

    # fig2 = plt.figure()
    # for i in range(2):
    #     plt.plot(freq_list, counts.mean(0)[:,i], label=state[i],)
    # plt.xlabel(r"Frequency (kHz)")
    # plt.ylabel("Counts")
    # plt.legend()
    # plt.show()

    # fig3 = plt.figure()
    # plt.plot(freq_list, counts.mean(0)[:,0] - counts.mean(0)[:,1], label=state[0]+'-'+state[1])
    # plt.xlabel(r"Frequency (kHz)")
    # plt.ylabel("Difference in Counts")
    # plt.legend()
    # plt.show()

    # fig1 = plt.figure()

    # for i in range(2):
    #         plt.hist(counts[:, :, i].flatten(), bins=np.arange(0, self.n_ro_nuclear / 2, 1),alpha = 0.5)



def plot_pretty(
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
    decay_time: float = 3000,
    fit_func = complex_ramsey_gaussian_fit
    
):
    
    plt.rcParams['font.weight'] = 'bold'
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titleweight'] = 'bold'

    plt.rcParams['lines.linewidth'] = 2.5      # line thickness
    plt.rcParams['axes.linewidth'] = 2     # box (spines) thickness
    plt.rcParams['xtick.major.width'] = 3
    plt.rcParams['ytick.major.width'] = 3



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
        -1 * np.pi,
        (np.max(data_I) - np.min(data_I)) / 2,  # Amplitude
        (np.average(data_I) + np.average(data_Q)) / 2,  # offset
    ]
    try:
        # Perform curve fitting with initial guesses
        params, params_covariance = curve_fit(fit_func, time_, Z, p0=guess)
    except Exception as e:
        print("Fit failed:", e)
        params = guess  # Use initial guess if fit fails
        params_covariance = []
    # Extract fit parameters for display
    f1_fit, T_fit, phi1_fit, A1_fit, offset_fit = params
    
    std = Bootstrap_analysis(time_, data_I_before_averaging,data_Q_before_averaging, params,plot=plot_bootstrap, fit_func = fit_func)
    print(f"Transition {transition}: Fitted Frequency = {1e3 * f1_fit:.2f} Hz,, Std = {std*1e3:.4f} Hz")
    fig = plt.figure()

    # Generate a smooth line to overlay the fitted function
    x = np.linspace(time_[0], time_[-1], 1001)
    
    if plot:
        if plot_guess:
            plt.plot(
                x,
                fit_func(x, *guess)[:len(x)],
                label="Guess: Dual Cosine with Decay",
                linestyle="--",
                color=colors[transition],
                linewidth=2,
            )
        plt.plot(
            x,
            fit_func(x, *params)[:len(x)],  # Ensure `params` matches the expected parameter count
            color=colors[transition],
            label = r"$f_{ground}$ =" +
            rf"{1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz"
        )
        plt.plot(
            x,
            fit_func(x, *params)[len(x):],  # Ensure `params` matches the expected parameter count
            '--',
            color='black',
            alpha = 0.2
        )
        plt.plot(time_, data_I, "o", label=transition, color=colors[transition], markeredgecolor = 'black',)
        plt.plot(time_, data_Q, "o", label='Ramsey quadrature', color='black', markeredgecolor = 'black', alpha = 0.2)
        plt.ylabel("Population",fontsize = 14)
        plt.ylim(0, 1)
        # plt.legend()
        plt.xlabel("Ramsey time (ms)",fontsize = 14)
        
        # Summary of fitted parameters for display
        plt.title(
            rf"$\boldsymbol{{\omega_{{\uparrow({transition}, {transition+1})}}}}$ = " +
            f"{1e9 * total_artificial_detuning + 1e3 * f1_fit + drive_freq:.1f} Hz",fontsize = 15
        )

      
        # Experiment metadata in the title
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)

        plt.show()

    return f1_fit, std

def Bootstrap_analysis(time_,x,y,guess,fit_func = complex_ramsey_gaussian_fit,plot=False):
    n_bootstrap = 1000
    f1_bootstrapped = []
    N = y.shape[0]
    if plot:
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
            params, _ = curve_fit(fit_func, time_, z_sampled, p0=guess)
            #delta_f, delta_t, delta_phi1, delta_A, delta_offset
            error = np.abs(params-guess)
            
            max_error = [1/(time_[-1]-time_[0]),guess[1]*0.3,np.inf,guess[3]*0.5,guess[4]*0.5]
            max_error = [1/(time_[-1]-time_[0]),np.inf,np.inf,np.inf,np.inf]
            
            if (np.sum(error>max_error) == 0):
            
                f1_bootstrapped.append(params[0]) 
                if plot:
                    axs[0].plot(time_dense,fit_func(time_dense,*params)[:len(time_dense)], alpha=0.1, color='black')
                    axs[0].plot(time_,x_sampled,'x', alpha=0.5, color='black')
                
            else :
                if plot:
                    axs[0].plot(time_dense,fit_func(time_dense,*params)[:len(time_dense)], alpha=0.1, color='blue')

        except:
            print('fit failed this is bad..')
            
            continue
        


    f1_bootstrapped = np.array(f1_bootstrapped)
    f1_means = np.mean(f1_bootstrapped)
    f1_std = np.std(f1_bootstrapped)

    if plot:
        
        axs[0].set_title(rf'Bootstrap plot of signal std is {f1_std*1e3:.4f} Hz')
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
                          n_freq=1,decay_time = 3,fit_func = complex_ramsey_gaussian_fit,
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
        global_params, params_covariance = curve_fit(fit_func, time_, Z, p0=guess)
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
                  label=f"{i*time_per_chunk}h - {(i+1)*time_per_chunk}h")
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
    
def plotSpectroscopy(
    signal: dict,
    threshold,
    n_peaks=1,
    experiment_name = "RamanSpectroscopy",
    avg_slice=slice(None),
    freq_slice=slice(None),
    width: float = 0.5    
):
    
    
    start_i, stop_i, step_i = avg_slice.indices(signal["iteration"])   # step_i will be 1
    nb_averages_sliced = max(0, stop_i - start_i)

    clicks = signal["clicks"][avg_slice,freq_slice,:,:] #(avg, n_point, NRO, read_outs)
    iteration = stop_i-start_i
    freq_list = (signal['freq_list'] * 1e-3)[freq_slice]
    delta_freq = signal['delta_freq'][freq_slice]

    # --- everything above stays the same up to your plt.subplots(...)

    fig, axs = plt.subplots(2, 2, figsize=(12, 9), tight_layout=True)

    print(clicks.shape)
    counts = clicks.sum(2)
    p_down = (counts[:, :, 1] > threshold).mean(0)
    p_up = (counts[:, :, 0]   > threshold).mean(0)
    
    if p_down.mean(0) > p_up.mean(0):
        p_fit = p_down
        p_not_fit = p_up
    else:
        p_fit = p_up
        p_not_fit = p_down
        
    prominence = 0.02
    peak_distance =  4
    p_idx, props = find_peaks(-p_fit, prominence=prominence, distance=peak_distance)
    # if p_idx.size == 0:
    #     raise RuntimeError("No peaks found – adjust 'prominence' or 'peak_distance'.")

    if p_idx.size != 0:
        # Keep only the N most prominent peaks if user requests it
        if n_peaks is not None and p_idx.size > n_peaks:
            order = np.argsort(props["prominences"])[::-1]  # descending
            p_idx = p_idx[order][:n_peaks]
        
        
        
        max_idxs = p_idx
        
        guess =   np.concatenate((np.array([[freq_list[max_idxs[i]] ,width,-abs(np.max(p_fit)- min(p_fit))] for i in range(n_peaks)] ).flatten(), [np.max(p_fit)]))
        fine = np.linspace(min(freq_list),max(freq_list),len(freq_list)*500)
        # bounds = [-np.inf,np.inf]*(3*n_peaks+1)
        # for k in range(n_peaks):
        #     bounds[2*k] = (min(p_not_fit),max(p_fit))
        
        try:
            
            fine = np.linspace(min(freq_list),max(freq_list),len(freq_list)*500)
            est, cov = curve_fit(make_multi_lorentz(n_peaks),freq_list,p_fit,guess)
            data_fit = make_multi_lorentz(n_peaks)(fine,*est)
            
        except Exception as e:
            print('could not fit')
            est, std, fine, data_fit = guess, guess, fine, make_multi_lorentz(n_peaks)(fine,*guess)

        n_peaks = (len(est) - 1) // 3
        peaks = []
        for k in range(n_peaks):
            x0, gamma, amp = est[3 * k : 3 + 3 * k]
            mean = est[-1]
            peaks.append({"amp": amp, "x0": x0, "gamma": gamma, "fwhm": 2 * gamma, 'mean':mean})
                
        # ---- (1) Population vs frequency with fit ----
        ax = axs[0, 0]
        
        for i, pk in enumerate(peaks, start=1):
            x0 = pk["x0"]
            ax.axvline(x0, ls="--", alpha=0.3,label = f"Peak {i + 1}: f = {pk['x0']:.3f} Khz")
            ax.text(
                x0,
                0.5,     # 2% above the max for visibility
                f"Peak {i}",
                rotation=90,
                va="bottom",
                ha="center",
                fontsize="small",
                color = "red",
            )
            ax.plot(fine, data_fit, "--", color="black", alpha=0.6)
            
            fine = np.asarray(fine)
            data_fit = np.asarray(data_fit)

            
    else :
        print("no peaks were found")
        est = [0]
        

    state = [r"$|\Downarrow>$", r"$|\Uparrow>$"]

    # ---- (1) Population vs frequency with fit ----
    ax = axs[0, 0]

    ax.errorbar(
        freq_list,
        p_down,
        np.sqrt(p_down * (1 - p_down) / iteration),
        label=state[0],
        color = 'b'
    )
    ax.errorbar(
        freq_list,
        p_up,
        np.sqrt(p_up * (1 - p_up) / iteration),
        label=state[1],
        color = 'g'
    )

    ax.set_title(
        "\n"
        + f"Peak frequency: {est[0]:.3f} kHz"
        + f"\nAverages: {iteration:.0f}",
        fontweight="bold",
    )
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel("Population")
    ax.set_ylim([-0.1, 1.1])
    # ax.set_xlim([950, 1000])
    ax.legend()

    # ---- (2) Mean counts vs frequency ----
    ax = axs[0, 1]
    for i in range(2):
        ax.plot(freq_list, counts.mean(0)[:, i], label=state[i])
    ax.set_title("Mean counts vs frequency", fontweight="bold")
    ax.set_xlabel(r"Frequency (kHz)")
    ax.set_ylabel("Counts")
    ax.legend()

    # # ---------------------------------------------------------------
    # # Compute FWHM region from the fitted curve (robust & param-free)
    # # ---------------------------------------------------------------

    # # Center at the maximum of the fitted curve
    # idx0 = int(np.argmax(data_fit))
    # f0_fit = float(fine[idx0])          # fitted peak frequency (kHz)
    # y0 = float(data_fit[idx0])

    # # Estimate "baseline" as the lower envelope of the fit (min of the fitted curve)
    # y_base = float(np.min(data_fit))
    # half_level = y_base + 0.5 * (y0 - y_base)

    # # Find left/right half-maximum crossing nearest to the peak
    # # (search to the left)
    # i_left = idx0
    # while i_left > 0 and data_fit[i_left] > half_level:
    #     i_left -= 1
    # # linear interpolate for better FWHM
    # if i_left < idx0:
    #     xL = np.interp(half_level, [data_fit[i_left], data_fit[i_left+1]], [fine[i_left], fine[i_left+1]])
    # else:
    #     xL = fine[0]

    # # (search to the right)
    # i_right = idx0
    # n_fine = len(fine)
    # while i_right < n_fine - 1 and data_fit[i_right] > half_level:
    #     i_right += 1
    # if i_right > idx0:
    #     xR = np.interp(half_level, [data_fit[i_right-1], data_fit[i_right]], [fine[i_right-1], fine[i_right]])
    # else:
    #     xR = fine[-1]

    # FWHM = float(xR - xL)
    # left_edge = f0_fit - 0.5 * FWHM
    # right_edge = f0_fit + 0.5 * FWHM

    # # Build masks for your discrete sweep points
    # mask_in = (freq_list >= left_edge) & (freq_list <= right_edge)
    # mask_out = ~mask_in

    # # Safety checks (avoid empty selections)
    # if not np.any(mask_in):
    #     # fall back to nearest point if window is too narrow
    #     nearest = np.argmin(np.abs(freq_list - f0_fit))
    #     mask_in = np.zeros_like(freq_list, dtype=bool)
    #     mask_in[nearest] = True
    #     mask_out = ~mask_in

    # # ------------------------------
    # # (3) Histograms: within FWHM
    # # ------------------------------
    # ax = axs[1, 0]
    # # counts shape: (n_iter, n_pts, 2), select freq points by mask_in then flatten
    # bins = np.arange(0, max(1, int(clicks.shape[1]/2)) + 1, 1)

    # ax.hist(
    #     counts[:, mask_in, 0].ravel(),
    #     bins=bins, alpha=0.55, density=True, label=r"$C_\Downarrow$ (in FWHM)"
    # )
    # ax.hist(
    #     counts[:, mask_in, 1].ravel(),
    #     bins=bins, alpha=0.55, density=True, label=r"$C_\Uparrow$ (in FWHM)"
    # )

    # ax.set_title(
    #     f"Histograms within FWHM\n"
    #     f"center={f0_fit:.3f} kHz, FWHM={FWHM:.3f} kHz, "
    #     f"N pts in={mask_in.sum()}",
    #     fontweight="bold",
    # )
    # ax.set_xlabel("Counts")
    # ax.set_ylabel("Density")
    # ax.legend()

    # # ------------------------------
    # # (4) Histograms: outside FWHM
    # # ------------------------------
    # ax = axs[1, 1]
    # ax.hist(
    #     counts[:, mask_out, 0].ravel(),
    #     bins=bins, alpha=0.55, density=True, label=r"$C_\Downarrow$ (background)"
    # )
    # ax.hist(
    #     counts[:, mask_out, 1].ravel(),
    #     bins=bins, alpha=0.55, density=True, label=r"$C_\Uparrow$ (background)"
    # )

    # ax.set_title(
    #     f"Histograms outside FWHM (background)\nN pts out={mask_out.sum()}",
    #     fontweight="bold",
    # )
    # ax.set_xlabel("Counts")
    # ax.set_ylabel("Density")
    # ax.legend()
    # figures = [fig]

    plt.show()





def plot_spectro(
        signal,
        fit: dict = None,
        transition = 0,
        N_RO = 300,
        nuclear_frequency = None,
        raman_pi_duration = None,
    ):
    
    
    
        spectro_pulse_duration = raman_pi_duration // 4
        fit = signal
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.rcParams['font.size'] = 16
        # Prepare data to be plotted


        data_ = signal["data"]
        data_click = signal["data_click"]
        threshold = signal['threshold']
        transition = signal['transition']

        freq_range = signal["raman_freq_sweep"]

        
        guess = [
            freq_range[np.argmin(data_)],
            (freq_range[-1] - freq_range[0]) / 5,
            min(data_) - max(data_),
            max(data_),
        ]
        
        guess = [
                freq_range[np.argmin(data_)],
                1/(spectro_pulse_duration*4*1e-9)/2,
                spectro_pulse_duration*4*1e-9,  # t
                -(max(data_)-min(data_)),
                max(data_),
            ]
                    
                    
        try:
            # Perform curve fitting with initial guesses
            est, _ = curve_fit(chevron, freq_range, data_, p0=guess)

        except Exception as e:
            print("Fit failed:", e)
            est = guess  # Use initial guess if fit fails
        
        fine = np.linspace(freq_range.min(), freq_range.max(), 1000)
        
        
        
        

        zero_index = 0
        pi_index = np.argmin(data_click.mean(0))
        
        freq_axis = freq_range * 1e-3 - nuclear_frequency * 1e-3
        
        
        plt.plot(
            fine*1e-3 - nuclear_frequency*1e-3,
            chevron(fine, *est),
            label=f"fit: {est[0]/1e6:.6f} MHz", color='black', alpha=0.5
        )


        fig = plt.figure(figsize=(14, 7), constrained_layout=True)
        # 2 rows, 3 columns; make the top row twice as tall as the bottom
        gs = GridSpec(2, 3, figure=fig, height_ratios=[2, 1], hspace=0.4, wspace=0.3)

        # ---- Main plot, spans all 3 columns on row 0 ----
        ax_main = fig.add_subplot(gs[0, :])
        p_err = np.sqrt(data_*(1-data_)/data_click.shape[0])
        ax_main.errorbar(freq_axis, data_, yerr=p_err, marker='o', linestyle='-',
                        label=f'{transition}', color=colors[transition])
        ax_main.plot(
            fine*1e-3 - nuclear_frequency*1e-3,
            chevron(fine, *est),
            label=f"fit: {est[0]/1e6:.6f} MHz", color='black', alpha=0.5
        )
        ax_main.set_xticks(
            np.linspace(freq_range.min(), freq_range.max(), 11)*1e-3
            - nuclear_frequency*1e-3
        )
        ax_main.set_ylim(0,1)
        ax_main.set_xlabel(
            f"$\\delta$ [kHz] from target {nuclear_frequency*1e-6:.3f} MHz"
        )
        ax_main.set_ylabel("Population")
        ax_main.legend(fontsize=12)
        # ax_main.set_title("Raman Spectroscopy – State Assignment")

        # ---- Bottom row: 3 equal‑width subplots ----

        # 1) average-click traces
        ax1 = fig.add_subplot(gs[1, 0])
        for i, data_click_ in enumerate(data_click.mean(0).T):
            ax1.plot(freq_axis, data_click_, label=nuclear_frequency + i,
                    color=colors[transition + i])
        ax1.axvline(freq_axis[zero_index])
        ax1.axvline(freq_axis[pi_index])
        ax1.set_xticks(
            np.linspace(freq_range.min(), freq_range.max(), 3)*1e-3
            - nuclear_frequency*1e-3
        )
        ax1.set_xlabel(
            f"$\\delta$ [kHz] from target {nuclear_frequency*1e-6:.3f} MHz"
        )
        ax1.set_ylabel("Counts")
        ax1.legend(fontsize=12)
        # ax1.set_title("Raman Spectroscopy – Clicks")

        # 2) overall histogram of all clicks
        ax2 = fig.add_subplot(gs[1, 1])
        for i, data_click_ in enumerate(np.concatenate(data_click).T):
            ax2.hist(data_click_, bins=np.arange(0, N_RO//2,1),
                    alpha=0.5, label=transition + i,
                    color=colors[transition + i])
        
        ax2.set_xlabel("Counts")
        # ax2.set_ylabel("Number of occurrences")
        ax2.legend(fontsize=12)
        # ax2.set_title("Histogram of Clicks (all)")

        # 3) zero‑vs‑pi histogram
        ax3 = fig.add_subplot(gs[1, 2])
        clicks_0 = data_click[:, zero_index, 0]
        clicks_pi = data_click[:, pi_index, 0]
        ax3.hist(clicks_0, bins=np.arange(0, N_RO//2,1),
                alpha=0.5, label=transition, color=colors[transition])
        ax3.hist(clicks_pi, bins=np.arange(0, N_RO//2,1),
                alpha=0.5, label=transition+1,
                color=colors[transition+1])
        ax3.set_xlabel("Counts")
        # ax3.set_ylabel("Number of occurrences")
        ax3.legend(fontsize=12)
        # ax3.set_title("Histogram of Clicks (0 vs π)")

        # super‑title and layout

        plt.tight_layout()
        plt.show()
        
        
def plotRabi(
    self,
    signal: dict,
    fit: dict,
    experiment_name="RamanRabi",
    plot_fig=True,
    threshold = 45,
):
    
    # ---------------- Rabi analysis + plotting ----------------
    clicks = signal["clicks"]
    iteration = signal["iteration"]

    # time axis (use your sweep vector directly)
    t_list = signal['rabi_time_list']   # e.g., in ns/us/s — whatever you used
    t_list = t_list - t_list.min()                         # start from 0 for fit/FFT robustness
    delta_freq = signal["delta_freq"]
    # populations from clicks
    print(clicks.shape)
    counts = clicks.sum(2) 
    ps = []
    for state in range(len(self.ro_freqs)):
        ps.append(((counts[:, :, state]> threshold).mean(0)))
    
    sorted_idx = np.argsort([np.std(p) for p in ps])
    p_fit  = ps[sorted_idx[-1]]
    print(sorted_idx)


    y = p_fit - np.mean(p_fit)
    N = len(y)
    win = np.hanning(N)
    y_w = y * win

    dt = np.median(np.diff(t_list))
    freqs_fft = np.fft.rfftfreq(N, d=dt)
    Y = np.fft.rfft(y_w)
    mag = np.abs(Y)

    if len(mag) > 1:
        peak_idx = 1 + np.argmax(mag[1:])
    else:
        peak_idx = 0
    f_guess_fft = float(freqs_fft[peak_idx])


    a0  = 0.5 * (np.max(p_fit) - np.min(p_fit))
    c0  = float(np.mean(p_fit))
    b0  = 0.0
    t00 = 0.0
    T0  = 100
    guess = [f_guess_fft, t00, a0, b0, c0, T0]

    try:
        est, std, fine, data_fit = fit_function(guess, rabi_decay_fit, t_list, p_fit)
        f_rabi = float(est[0])
        t0     = float(est[1])

        phi = -2.0 * np.pi * f_rabi * t0
        phi = (phi + np.pi) % (2.0 * np.pi) - np.pi

        t_pi  = 1.0 / (2.0 * f_rabi)
        t_pi2 = 1.0 / (4.0 * f_rabi)

        def principal_time(target_angle):
            k = np.ceil((2.0 * np.pi * f_rabi * (0.0 - t0) + target_angle) / (2.0 * np.pi)) - 1
            t = t0 + (target_angle + 2.0 * np.pi * k) / (2.0 * np.pi * f_rabi)
            if t < 0:
                t += 1.0 / f_rabi
            return float(t)
        t_pi2_phase = principal_time(np.pi/2)
        t_pi_phase  = principal_time(np.pi)
        
    except Exception as e:
        print("Rabi fit failed, falling back to FFT freq guess. Error:", e)
        est = guess
        std = [np.nan] * len(guess)
        fine = np.linspace(t_list.min(), t_list.max(), 4 * len(t_list))
        data_fit = rabi_decay_fit(fine, *est)


    fig, axs = plt.subplots(3, 2, figsize=(12, 9), tight_layout=True)

    ax = axs[0, 0]
    
    for i,p in enumerate(ps):
        ax.errorbar(
            t_list, p, np.sqrt(p * (1 - p) / iteration),
            linewidth=2, label=i,
        )

    ax.plot(fine, data_fit, "--", color="black", alpha=0.7, label="Rabi fit")

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Population")
    ax.set_ylim([-0.01, 1.01])
    ax.legend()

    # (2) Mean counts vs time
    ax = axs[0, 1]
    for i in range(len(self.ro_freqs)):
        ax.plot(t_list, counts.mean(0)[:, i], label=i)
    ax.set_title("Mean counts vs time", fontweight="bold")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Counts")
    ax.legend()

    # (3) FFT magnitude spectrum (new)
    ax = axs[1, 0]
    ax.plot(freqs_fft, mag, lw=2)
    if np.isfinite(f_guess_fft):
        ax.axvline(f_guess_fft, ls="--", color="k", alpha=0.7, label=f"FFT peak ~ {f_guess_fft:.6g}")
    ax.set_title("FFT of population (windowed)", fontweight="bold")
    ax.set_xlabel("Frequency (1/time)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.legend()

    # (4) Histograms of counts
    ax = axs[1, 1]
    for i in range(2):
        ax.hist(
            counts[:, :, i].flatten(),
            bins=np.arange(0, max(1, self.n_ro_nuclear // 2), 1),
            alpha=0.5,
            label=i,
            density=True,
        )
    ax.set_title("Histogram of counts", fontweight="bold")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Density")
    ax.legend()
    
    plt.suptitle(
        self.time_stamp + experiment_name
        + "\n"
        + rf"driving at {self.W_raman_freq*1e-3} KHz with amp1: {self.W_raman_relamp1} and amp2: {self.W_raman_relamp2}"
        + "\n"
        + (f"Rabi freq: {f_rabi:.6g} (kHz) | "
        f"pi/2:  {t_pi2_phase:.6g}, "
        f"pi: {t_pi_phase:.6g}")
        + f"\nAverages: {iteration:.0f}",
        fontweight="bold",
    )
    
    #(4) Histograms of counts at max contrast
    ax = axs[2, 0]
    index_max_contrast_quad = [np.argmax(np.abs(ps[sorted_idx[0]] - ps[sorted_idx[1]])[:]) for i in range(1)]
    
    for i in range(1):
        ax.hist(
            counts[:, index_max_contrast_quad, i].flatten(),
            bins=np.arange(0, max(1, self.n_ro_nuclear // 2), 1),
            alpha=0.5,
            label=i,
            density=True,
        )
    # ax.set_title(f"Histogram of counts at max contrast (t={t_list[index_max_contrast_quad]:.2f} ms and {t_list[index_max_contrast_quad]:.2f} ms)", fontweight="bold")
    ax.set_xlabel("Counts")
    ax.set_ylabel("Density")
    ax.legend()
    
    
    # (5) Tracking of frequency shifts
    ax = axs[2, 1]
    if self.track:
        ax.plot(delta_freq, label='delta_freq')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Delta frequency (Hz)')
    ax.set_title('Tracking of frequency shifts during the experiment')

    plt.show()

    self.save(dataset=signal,
        figure=fig,
    )

    return

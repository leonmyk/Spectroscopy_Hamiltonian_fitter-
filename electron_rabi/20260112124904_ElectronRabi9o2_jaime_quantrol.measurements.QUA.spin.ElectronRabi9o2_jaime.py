# built-in modules
# ### Imports and Dependencies
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd

# Import your custom modules
from quantrol.utilities.SaveLoad import save_data_fig_config
from quantrol.utilities.fit import fit_function, rabi_fit, rabi_decay_fit
from quantrol.utilities.Lib_QUA import *
from qm._results import JobResults
from qm.QuantumMachinesManager import QuantumMachinesManager
from quantrol.utilities.SaveLoad import capture_constructor_args
from quantrol.core.measures import QMSpinMeasurement

def placeholder_figure() -> plt.Figure:
    """
    Creates a placeholder figure indicating that no plot was generated.

    Returns
    -------
    matplotlib.pyplot.Figure
        A placeholder figure with a message indicating no plot was generated.
    """
    fig, ax = plt.subplots(figsize=(4, 3))
    message = "No plot was generated. Run .plot after .analysis to plot results."
    ax.text(0.5, 0.5, message, fontsize=12, ha="center", va="center")
    ax.axis("off")
    return fig


class ElectronRabi9o2_jaime(QMSpinMeasurement):

    def __init__(
        self,
        n_iterations: int,
        n_SMPD_cycles: int,
        
        duration_steps = 51,
        duration_init =  100,
        duration_final =  100e3,
        
        preparation: str = None,
        prepare_every_nth_iteration: int = 1,
        centre_freq: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Store parameters and setup instruments
        self.metadata = capture_constructor_args()
    
        if centre_freq is not None:
            self.config.circuit['spin']['frequency_offset'] = centre_freq

        self.centre_freq = self.config.circuit['spin']['frequency_offset']
        
        self.ro_angle = self.config.elements[self.ro_element].angle
        self.I_threshold = self.config.elements[self.ro_element].I_threshold
        self.I_threshold_reset = self.config.elements[self.ro_element].I_threshold_reset

        self.n_iterations = n_iterations
        
        self.n_SMPD_cycles = n_SMPD_cycles
        
        self.duration_steps = duration_steps
        self.duration_init = duration_init
        self.duration_final = duration_final

        self.duration_range = np.linspace(
            self.duration_init,
            self.duration_final,
            self.duration_steps
        ).astype(int).tolist()

        self.preparation = preparation       
        self.prepare_every_nth_iteration = prepare_every_nth_iteration

    def execute_measure(self) -> JobResults:
        with program() as prog:
            # iteration
            i = declare(int)
            i_reset = declare(int, value=0)
            duration_set = declare(int)

            # readout
            I = declare(fixed)
            p = declare(bool)
            pr = declare(bool)

            click = declare(int)

            # streams
            p_stream = declare_stream()
            index_stream = declare_stream()
            
            ramp_to_zero(self.flux_element, int(1e4/4))

            with for_(i, 0, i < self.n_iterations, i + 1):
                
                save(i, index_stream)
                assign(i_reset, i_reset + 1)

                
                ## Preparation
                with if_(i_reset == self.prepare_every_nth_iteration):
 
                    tungsten_start_freq = 100e6
                    tungsten_freq_range = 1000e6
                    tungsten_n_pumping = 10
                    
                    
                    if self.preparation == "ground + tungsten right":
                        
                        tungsten_chirp_sign = 1

                        # ground
                        chirp_rate = int((self.config.circuit['spin']['chirp_span'])*1e-3 / (self.config.elements[self.spin_element].pulses['sideband_pump_chirped'].length*1e-9))
                        nuclear_spin_chirped_pump(self.config.circuit["spin"]["frequency_offset"], self.config, -1, chirp_rate, n_pumping=10)
                        wait(int(50e6//4))
                        
                        # tungsten
                        chirp_rate = int(tungsten_freq_range*1e-3 / (self.config.elements[self.spin_element].pulses['sideband_pump_chirped'].length*1e-9))
                        freq = self.config.elements['spin'].IF + self.config.circuit["spin"]["frequency_offset"] + tungsten_chirp_sign*int(tungsten_start_freq)

                        index_pumping = declare(int)
                        
                        update_frequency('spin', freq)        

                        with for_(index_pumping, 0, index_pumping < tungsten_n_pumping, index_pumping + 1):
                            align()
                            update_frequency('spin', freq) # Recovers the frequency after chirping!!!
                            play('sideband_pump_chirped'*amp(1.0), 'spin', chirp = (tungsten_chirp_sign*chirp_rate,'KHz/sec'))

                    elif self.preparation == "ground + tungsten left":
                        
                        tungsten_chirp_sign = -1

                        # ground
                        chirp_rate = int((self.config.circuit['spin']['chirp_span'])*1e-3 / (self.config.elements[self.spin_element].pulses['sideband_pump_chirped'].length*1e-9))
                        nuclear_spin_chirped_pump(self.config.circuit["spin"]["frequency_offset"], self.config, -1, chirp_rate, n_pumping=10)
                        wait(int(50e6//4))
                        
                        # tungsten
                        chirp_rate = int(tungsten_freq_range*1e-3 / (self.config.elements[self.spin_element].pulses['sideband_pump_chirped'].length*1e-9))
                        freq = self.config.elements['spin'].IF + self.config.circuit["spin"]["frequency_offset"] + tungsten_chirp_sign*int(tungsten_start_freq)

                        index_pumping = declare(int)
                        
                        update_frequency('spin', freq)        

                        with for_(index_pumping, 0, index_pumping < tungsten_n_pumping, index_pumping + 1):
                            align()
                            update_frequency('spin', freq) # Recovers the frequency after chirping!!!
                            play('sideband_pump_chirped'*amp(1.0), 'spin', chirp = (tungsten_chirp_sign*chirp_rate,'KHz/sec'))

                    elif self.preparation == "ground":
    
                        # ground
                        chirp_rate = int((self.config.circuit['spin']['chirp_span'])*1e-3 / (self.config.elements[self.spin_element].pulses['sideband_pump_chirped'].length*1e-9))
                        nuclear_spin_chirped_pump(self.config.circuit["spin"]["frequency_offset"], self.config, -1, chirp_rate, n_pumping=10)
                        wait(int(50e6//4))

                    elif self.preparation == "tungsten_red":
                        
                        pump_W_bath(self.config, polarize_sign=-1)
                        wait(int(40e6//4))
                    
                    elif self.preparation == "calcium":
                        
                        pump_W_bath(self.config, polarize_sign=-1)
                        wait(int(1e6//4))
                        align()
                    
                    elif self.preparation == "new preparation":
                        # ground state preparation
                        prepare_Nb_GND_and_polarize(
                            config=self.config,
                            from_state=9,
                        )
                        align()
                        wait(int(40e6 // 4))
                        align()
                    
                    else:
                        pass
                    
                    assign(i_reset, 0)
                    
                    #wait(int(1e6//4))
                    
                with for_each_(duration_set, self.duration_range):

                    # perform RO
                    assign(click, 0)

                    # excite electron
                    update_frequency(self.pump_element, self.config.elements[self.pump_element].IF - self.centre_freq)
                    update_frequency(self.spin_element, self.config.elements[self.spin_element].IF + self.centre_freq)
                    play("ON", 'aux_trigger')
                    play(
                        # 'pi_gauss_short',
                        self.spin_pulse,
                        self.spin_element,
                        duration=duration_set/4 #self.config.elements[self.spin_element].pulses[self.spin_pulse].length,
                    )       
                    align()
                    
                    # Wait to allow the SA to capture all durations
                    # wait(int(10e6//4))

                    # let it ring down
                    wait(int(self.config.circuit['spin']['ringdown_time']/4))
                    align()

                    # collect clicks
                    I = readout_block_Idual(self.ro_element, self.ro_pulse, self.ro_angle, I)
                    assign(p, I > self.I_threshold)
                    assign(pr, I > self.I_threshold_reset)

                    measure_SMPD(
                        click_stream=p_stream,
                        n_SMPD_cycles=self.n_SMPD_cycles,
                        qe_ro=self.ro_element,
                        qe_qb=self.qubit_element,
                        ro_pulse=self.ro_pulse,
                        qb_pulse=self.qubit_pulse,
                        ro_angle=self.ro_angle,
                        pmp_pulse=self.pump_pulse,
                        qe_pmp=self.pump_element,
                        I_th=self.I_threshold,
                        I_th_reset=self.I_threshold_reset,
                        accumulate=True,
                        sound = False
                        )
                    align()

            with stream_processing():
                p_stream.buffer(len(self.duration_range)).save_all('clicks')
                p_stream.timestamps().buffer(len(self.duration_range)).save('timestamp')
                index_stream.save('iteration')

        # ### Execute the QUA Program
        qmm = QuantumMachinesManager(
            host=self.config.controllers[
                self.config.elements[self.ro_element].controller.name
            ].IP,
            port=self.config.controllers[
                self.config.elements[self.ro_element].controller.name
            ].port,
            octave=None,
        )
        qm = qmm.open_qm(self.QM_config)
        job = qm.execute(prog, flags=["auto-element-thread"])

        if self.simulate:
            simulate_sequence(qm=qm, program_qua=prog, duration=10_000)

        data = job.result_handles
        if self.wait_for_all_values:
            data.wait_for_all_values()

        return data

    def assemble_signal(self, data) -> dict:
        """
        Assembles the measurement data into a dictionary.

        Parameters
        ----------
        data : JobResults
            The raw data from measurement.

        Returns
        -------
        dict
            A dictionary with 'volt', 'freq_range', and 'flux_range' keys.
        """
        iteration = data.iteration.fetch_all()
        data_click = np.array([item[0] for item in data.clicks.fetch_all()])
        meas_time_hours = (data.timestamp.fetch_all()).mean() * 1e-9 / 3600

        return {
            "iteration": iteration,
            "data_click": data_click,
            "meas_time_hours": meas_time_hours,
            "duration": self.duration_range,
        }

    def perform_fit(self, signal) -> dict:
        """
        Performs curve fitting on the provided signal.

        Parameters
        ----------
        signal : dict
            Dictionary containing voltage data and ranges.

        Returns
        -------
        dict
            A dictionary with fitted parameters and flux dependence values.
        """
        data_click = signal["data_click"]
        duration = signal["duration"]
        excess = data_click.mean(0)

        # Initial guess for the fit parameters t,f,t0,a,b,c,T
        initial_guess = [3/(duration[-1]-duration[0]), 0, 0.06, (excess[-1]-excess[0])/(duration[-1]-duration[0]), min(excess), duration[-1]-duration[0]]
        params, params_covariance, _, _ = fit_function(initial_guess, rabi_decay_fit, duration, excess)

        return {
            "params": params,
            "params_covariance": params_covariance,
            "guess": initial_guess,
        }

    def assemble_dataset(self, signal, fit) -> dict:
        """
        Combines signal and fit data into a single dataset.

        Parameters
        ----------
        signal : dict
            The assembled signal data.
        fit : dict
            The fit data.

        Returns
        -------
        dict
            The combined dataset.
        """
        return {**signal, **fit}

    def save_results(self, signal, fit, fig=None):
        """
        Saves experiment results, including signal, fit, and figure data.

        Parameters
        ----------
        signal : dict
            The signal data to save.
        fit : dict
            The fit data to save.
        fig : matplotlib.pyplot.Figure, optional
            The figure to save
            Default to a placeholder figure indicating no figure was plotted.
        """
        if fig is None:
            fig = placeholder_figure()
        save_data_fig_config(
            fig=fig,
            class_instance=self,
            dataset=self.assemble_dataset(signal=signal, fit=fit),
            metadata=self.metadata,
        )

    def analysis(self, data: JobResults, save_results: bool = True):
        """
        Analyzes the measurement data by assembling and fitting the signal.

        Parameters
        ----------
        data : JobResults
            The measurement data.
        save_results : bool, optional
            Flag to save results (default is True).

        Returns
        -------
        tuple
            The signal and fit data.
        """
        signal = self.assemble_signal(data=data)
        fit = self.perform_fit(signal=signal)
        if save_results:
            self.save_results(signal=signal, fit=fit)

        return signal, fit

    def plot(
        self,
        signal,
        fit: dict = None,
        save_results: bool = True,
    ):
        
        duration = np.array(signal["duration"])/1e3
        duration_fit = np.linspace(duration[0], duration[-1], 1000)
        data_click = signal["data_click"]
        
        params = fit["params"]
        
        # Create a figure with two subplots (2 rows, 1 column)
        fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))

        # ---- First subplot: Original data and Lorentzian fit ----
        ax1.plot(duration, data_click.mean(0) , '-o', label='Data', color='blue')
        ax1.plot(duration_fit, rabi_decay_fit(duration_fit*1e3, *params))
        
        ax1.set_xlabel('Pulse duration (us)')
        ax1.set_ylabel(r'$\langle C \rangle$')
        ax1.set_title(
            f"{self.timestamp}_{self.experiment_name}"
            + f"\n# Iterations: {signal['iteration']}"
            + f'\nRabi freq: {params[0]*1e3:.1f} kHz\nPi pulse {params[1]:.0f} or {1/params[0]/2+params[1]:.0f} or {1/params[0]/2-abs(params[1]):.0f} ns' 
            + f"\nPreparation: '{self.preparation}' every {self.prepare_every_nth_iteration}"
        )
        # ax1.axvline(params[1]*1e-3)
        # Display the combined figure
        plt.show()

        if save_results:
            self.save_results(signal=signal, fit=fit, fig=fig)


if __name__ == "__main__":
    from quantrol.measurements.QUA import spin

    meas = spin.ElectronRabi9o2_jaime(
        n_iterations=1000000,
        n_SMPD_cycles=33,

        duration_steps = 301,
        duration_init =  100,
        duration_final =  100e3,
    )

    data = meas.execute_measure()

    # in the next cell
    signal, fit = meas.analysis(data=data, save_results=False)
    meas.plot(signal=signal, fit=fit, save_results=True)

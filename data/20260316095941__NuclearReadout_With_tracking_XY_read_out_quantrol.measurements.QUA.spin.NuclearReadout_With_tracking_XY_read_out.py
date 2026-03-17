import numpy as np
import matplotlib.pyplot as plt

from quantrol.utilities.fit import fit_function

from quantrol.utilities.math import calc_chirp_rate
from quantrol.utilities.fit import gaussian
from quantrol.utilities.SaveLoad import save_data_fig_config, capture_constructor_args
from quantrol.utilities.Lib_QUA import *
from quantrol.core.measures import QMSpinMeasurement

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.cluster import KMeans
import matplotlib.colors as mcolors






class NuclearReadout_With_tracking_XY_read_out(QMSpinMeasurement):
    """
    Maintainer: Jaime T
    --------

    Goal:
    --------
        Send a pulse into the spin sample and read out the SMPD

    Needed:
    --------
        qubit: good pi-pulse
        readout: readout pulse
        spin: spin pulse

    Parameters:
    --------
        qubit_element: qubit --> object
        qubit_pulse: qubit pulse --> str,
        ro_element: readout --> object,
        ro_pulse: readout pulse --> str,
        ph_element: spin --> object,
        ph_pulse: spin pulse --> str,
        
        n_SMPD_cycles : number of readout points --> int,
        n_iterations : number of iterations --> int,
        simulate : QM simulator, optional --> bool,
        wait_for_all_values: explicit --> bool

    Returns:
    --------
        Readout histograms --> fig
        angle (pi unit) --> print
        I threshold --> print
        I threshold reset --> print

    Template:
    --------
        meas = SpinDetection(qubit_element = qb,
                            qubit_pulse = qubit_pulse,
                            ro_element = ro,
                            ro_pulse = ro_pulse,
                            spin_element = spin,
                            spin_pulse = spin_pulse,
                            pump_element = pump,
                            pump_pulse = pump_pulse,
                            
                            n_iterations = 1_000_000,
                            n_SMPD_cycles = 200,
                            
                            config = config,
                            )
                            
        data = meas.execute_measure()

        signal = meas.analysis(data)
        meas.plot(signal, experiment_name='SpinDetection')
    """
    def __init__(self,
                 n_iterations: int,
                 n_SMPD_cycles: int,
                 resolution_enhancement_factor: int = 1,
                 resolution_enhancement_factor_large: int = 1,
                 nuclear_nro:int = 200,
                 spin_pulse: str = 'pi_gauss_short',
                 initial_track_freq: int = None,
                 polarisation_sign: int = -1,
                 states_to_prepare = [0,1,2,3],
                 track = True,
                 nro_delta_x: int  = 100,
                 nro_delta_y: int  = 100,
                 **kwargs,
                 ):

        self.metadata = capture_constructor_args()
        super().__init__(**kwargs)
        
        self.nuclear_nro = nuclear_nro
        self.nro_delta_x = nro_delta_x
        self.nro_delta_y = nro_delta_y
        self.resolution_enhancement_factor_large = resolution_enhancement_factor_large
        self.qe_qb =  self.config.circuit["smpd"]["qubit"]["element"]
        self.qb_pulse = self.config.circuit["smpd"]["qubit"]["pi_pulse"]
        self.qe_ro = self.config.circuit["smpd"]["readout"]["element"]
        self.ro_pulse = self.config.circuit["smpd"]["readout"]["pulse"]
        self.qe_pmp = self.config.circuit["smpd"]["pump"]["element"]
        self.pmp_pulse = self.config.circuit["smpd"]["pump"]["pulse"]
        self.polarisation_sign = polarisation_sign
        
        self.ro_angle = self.config.elements[self.qe_ro].angle
        self.I_threshold = self.config.elements[self.qe_ro].I_threshold
        self.I_threshold_reset = self.config.elements[self.qe_ro].I_threshold_reset
        self.qe_spin = 'spin'
        self.spin_pulse = spin_pulse

        self.centre_freq = self.config.circuit['spin']['frequency_offset']

        self.ro_angle = self.config.elements[self.ro_element].angle
        self.I_threshold = self.config.elements[self.ro_element].I_threshold
        self.I_threshold_reset = self.config.elements[self.ro_element].I_threshold_reset
        
        self.n_iterations = n_iterations
        self.n_SMPD_cycles = n_SMPD_cycles
        self.initial_track_freq = initial_track_freq    
        
        self.states_to_prepare = states_to_prepare
        self.resolution_enhancement_factor = resolution_enhancement_factor
        
        self.track = track
        raman_9o2 = self.config.circuit["spin"]["raman_4W"]
        self.raman_ramp_time = int(raman_9o2["ramp_time"])
        self.raman_freq = np.array(raman_9o2["nuclear_frequency"]).astype(int).tolist()
        self.raman_detuning = np.array(raman_9o2["detuning"]).astype(int).tolist()
        self.raman_pi_2_duration = (
            np.array(raman_9o2["pi_2_duration"]).astype(int).tolist()
        )
        self.raman_pi_duration = np.array(raman_9o2["pi_duration"]).astype(int).tolist()
        self.raman_relamp1 = np.array(raman_9o2["relamp_1"]).tolist()
        self.raman_relamp2 = np.array(raman_9o2["relamp_2"]).tolist()
        
        # Relevant spin frequencies built from the config file
        self.ringdown_time = self.config.circuit[self.spin_element]['ringdown_time']
        spin_A1 = self.config.circuit["spin"]["hyperfine_A1"]
        spin_A2 = self.config.circuit["spin"]["hyperfine_A2"]
        self.electron_freq = self.config.elements[self.spin_element].IF + self.initial_track_freq
        self.ro_freqs = np.array([
            int(self.electron_freq ), 
            int(self.electron_freq + spin_A1),      
            int(self.electron_freq + spin_A2),      
            int(self.electron_freq + spin_A1 + spin_A2)
        ])
        


    def nuclear_readout_block(self, click, spin_freq):
        """updates spin element frequency, pulses the spin at that frequency and reads it out once
        increments the variable click_acc if a click is detected"""
        
        update_frequency('spin', spin_freq)
        align()
        
        play(
            self.spin_pulse*amp(1./self.resolution_enhancement_factor),
            self.spin_element,
            duration=self.config.elements[self.spin_element].pulses[self.spin_pulse].length*self.resolution_enhancement_factor//4,
        )       
        align()

        wait(int(self.ringdown_time/4))
        align()

        n_click=measure_SMPD_nostream(
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
            return_click=True
        )
        wait(5_000//4)
        assign(click, n_click)
        
    def nuclear_readout_block_large(self, click, spin_freq):
        """updates spin element frequency, pulses the spin at that frequency and reads it out once
        increments the variable click_acc if a click is detected"""
        
        update_frequency('spin', spin_freq)
        align()
        
        play(
            self.spin_pulse*amp(1./self.resolution_enhancement_factor_large),
            self.spin_element,
            duration=self.config.elements[self.spin_element].pulses[self.spin_pulse].length*self.resolution_enhancement_factor_large//4,
        )       
        align()

        wait(int(self.ringdown_time/4))
        align()

        n_click=measure_SMPD_nostream(
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
            return_click=True
        )
        wait(5_000//4)
        assign(click, n_click)
     


            
      
    
    def nuclear_spin_RO_XY(
        self,
        xstream,
        readout_freqs,
        delta_freq,
        enable_fsv_trigger=False,
        accumulate=False,
    ):
        """reads nuclear spin at 4 readout frequencies given by readout_freqs
        readout pulses are interleaved and repeated N_ROcycle times each
        4 accumulator variables are incremented and returned one after another"""

        n_ro_set= declare(int)
        freq_set = declare(int)
        click_tmp = declare(int)
        readout_freqs_X = readout_freqs
        
        with for_(n_ro_set, 0, n_ro_set < self.nuclear_nro, n_ro_set + 1):
            
            # Runs nuclear_readout_block function once for each frequency
            assign(freq_set, readout_freqs[0])
            self.nuclear_readout_block(click_tmp, freq_set + delta_freq)
            save(click_tmp, xstream)
            # assign(delta_X1,click_tmp)
            
            # Runs nuclear_readout_block function once for each frequency
            assign(freq_set, readout_freqs[1])
            self.nuclear_readout_block(click_tmp, freq_set + delta_freq)
            save(click_tmp, xstream)
            # assign(delta_X,delta_X-click_tmp)
        
            # Runs nuclear_readout_block function once for each frequency
            assign(freq_set, readout_freqs[2])
            self.nuclear_readout_block(click_tmp, freq_set + delta_freq)
            save(click_tmp, xstream)
            # assign(delta_Y,delta_Y+click_tmp)
            
            # Runs nuclear_readout_block function once for each frequency
            assign(freq_set, readout_freqs[3])
            self.nuclear_readout_block(click_tmp, freq_set + delta_freq)
            save(click_tmp, xstream)
            # assign(delta_X,delta_X-click_tmp)
            
                        
                      
    def Prepare_State(self,State_to_prepare,delta_freq):
        

   
        with if_(State_to_prepare == 1):
            
            raman_cos_pulse_dont_keep_phase(
                nuclear_spin_freq=self.raman_freq[0],
                freq_electron=self.electron_freq + delta_freq,
                raman_detuning=self.raman_detuning[0],
                detuned_electron_amplitude=self.raman_relamp1[0],
                detuned_sideband_amplitude=self.raman_relamp2[0],
                pulse_duration=self.raman_pi_duration[0] // 4,
                wait_multiplier=1,
                ramp_time=self.raman_ramp_time // 4,
            )
            wait(int(30e6//4) // 4)
            wait(int(30e6//4))


        with if_(State_to_prepare == 2):

            raman_cos_pulse_dont_keep_phase(
                nuclear_spin_freq=self.raman_freq[1],
                freq_electron=self.electron_freq + delta_freq,
                raman_detuning=self.raman_detuning[1],
                detuned_electron_amplitude=self.raman_relamp1[1],
                detuned_sideband_amplitude=self.raman_relamp2[1],
                pulse_duration=self.raman_pi_duration[1] // 4,
                wait_multiplier=1,
                ramp_time=self.raman_ramp_time // 4,
            )
            wait(int(30e6) // 4)
            wait(int(30e6//4))
            wait(int(30e6//4))


        with if_(State_to_prepare == 3):

            raman_cos_pulse_dont_keep_phase(
                nuclear_spin_freq=self.raman_freq[0],
                freq_electron=self.electron_freq + delta_freq,
                raman_detuning=self.raman_detuning[0],
                detuned_electron_amplitude=self.raman_relamp1[0],
                detuned_sideband_amplitude=self.raman_relamp2[0],
                pulse_duration=self.raman_pi_duration[0] // 4,
                wait_multiplier=1,
                ramp_time=self.raman_ramp_time // 4,
            )
            wait(int(30e6)// 4)
            wait(int(30e6//4))
            wait(int(30e6//4))

            raman_cos_pulse_dont_keep_phase(
                nuclear_spin_freq=self.raman_freq[1],
                freq_electron=self.electron_freq + delta_freq,
                raman_detuning=self.raman_detuning[1],
                detuned_electron_amplitude=self.raman_relamp1[1],
                detuned_sideband_amplitude=self.raman_relamp2[1],
                pulse_duration=self.raman_pi_duration[1] // 4,
                wait_multiplier=1,
                ramp_time=self.raman_ramp_time // 4,
            )

            wait(int(30e6)// 4)
            wait(int(30e6//4))
            wait(int(30e6//4))
            wait(int(30e6//4))

    def execute_measure(self):
        #################################################
        #                                               #
        # Convention: always define /4 in the QUA code  #
        #                                               #
        #################################################
                
        with program() as prog:
            
            k = declare(int)
            j = declare(int)
            sign = declare(int)
            
            delta_freq = declare(int)
            assign(delta_freq, 0)
            
            state_to_prepare = declare(int)
            
            x_stream = declare_stream()
            y_stream = declare_stream()
            index_stream = declare_stream()
            tracking_stream = declare_stream()
            
            Y = declare(fixed)
            delta_freq = declare(int)
            delta_freq_acc = declare(int)
            
            
            assign(Y, 0)
            assign(delta_freq, 0)
            assign(delta_freq_acc, 0)


            with for_(k, 0, k < self.n_iterations, k + 1):
                save(k, index_stream)
                
                # Play chirped sideband pulse
                
                with for_(state_to_prepare, 0, state_to_prepare < len(self.states_to_prepare), state_to_prepare + 1):
                    play("ON", 'aux_trigger')
                    
                    pump_W_bath(self.config, polarize_sign=self.polarisation_sign)
                    align()
                    wait(int(30e6//4))
                    wait(int(30e6//4))
                    wait(int(30e6//4))
                    
                    with if_(self.track == True):
                        delta_freq = track_frequency_spins(
                            self.qe_spin,
                            self.qe_qb,
                            self.qe_pmp,
                            self.qe_ro,  # elements
                            "pi_gauss_short",
                            self.ro_pulse,
                            self.qb_pulse,
                            self.pmp_pulse,  # pulses
                            self.ro_angle,
                            self.electron_freq,
                            self.config.elements[self.qe_spin]
                            .pulses["pi_gauss_short"]
                            .length,  # pulse parameters
                            self.I_threshold,
                            self.I_threshold_reset,  # readout parameters
                            delta_freq,
                            delta_freq_acc,
                            Y,  # tracking variables
                        )
                        wait(int(2_000_000))
                        align()
                        save(delta_freq, tracking_stream)
                        
                        pump_W_bath(self.config, polarize_sign=self.polarisation_sign)
                    
                        align()
                        wait(int(30e6//4))
                        wait(int(30e6//4))
                        wait(int(30e6//4))
                        wait(int(30e6//4))
                    
                    
                    self.Prepare_State(state_to_prepare, delta_freq)
                    align()


                    # RO after Preparation onto the down state

                    self.nuclear_spin_RO_XY(x_stream, self.ro_freqs, delta_freq, accumulate=False)
                        
                    align()
                    wait(int(4e6//4))
                    align()
                    # self.Prepare_State(state_to_prepare, delta_freq)
                    
                    
                    # RO after Preparation onto the up state
                

            with stream_processing():

                x_stream.buffer(4).buffer(self.nuclear_nro ).buffer(len(self.states_to_prepare)).save_all('clicksX')
                    
                tracking_stream.save_all('delta_freq')
                index_stream.save('iteration')
        
        self.start_time = time.time()
        data = self.execute_qm_program(prog)

        return data


    def analysis(self, data):
        
        self.end_time = time.time()
        clicksX  = np.array([sublist[0] for sublist in data.clicksX.fetch_all()])
        iteration = data.iteration.fetch_all()
        delta_freq = data.delta_freq.fetch_all()['value']
        


        
        print(f'Averages {iteration}')
        
        data_to_fit = {
        }


        signal = {'clicksX': clicksX,
                  'iteration': iteration,
                  'start_time': self.start_time,
                  'end_time': self.end_time,
                  'iteration': iteration,
                  'delta_freq': delta_freq,
                  **data_to_fit
        }
        
        fit = {}
        
        return signal, fit


    def plot(self, signal: dict, fit: dict, nro_delta_x, nro_delta_y):

        figs = []
        
        
        clicksX = signal['clicksX']
        avg = clicksX.shape[0]
        nb_prep_states = clicksX.shape[1]
        xnro = clicksX.shape[2]
        self.nro_delta_x = nro_delta_x
        self.nro_delta_y = nro_delta_y
        print('clicks shape is ',clicksX.shape)  #clicks shape is  (43, 4, 50, 4)
        
        delta_X = np.zeros((nb_prep_states,avg))
        delta_X_nro = np.zeros((nb_prep_states,avg,self.nuclear_nro))
        delta_Y_nro = np.zeros((nb_prep_states,avg,self.nuclear_nro))
        delta_Y = np.zeros((nb_prep_states,avg))
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        fig1 = plt.figure()
        for prepped_state in self.states_to_prepare:
            
            
            dx = clicksX[:,prepped_state,:,0] + clicksX[:,prepped_state,:,1] - clicksX[:,prepped_state,:,2] - clicksX[:,prepped_state,:,3]
            dy = clicksX[:,prepped_state,:,0] + clicksX[:,prepped_state,:,2] - clicksX[:,prepped_state,:,3] - clicksX[:,prepped_state,:,1]
            
            delta_X_nro[prepped_state] = dx
            delta_Y_nro[prepped_state] = dy
            delta_X[prepped_state] = dx[:,:self.nro_delta_x].sum(1)
            delta_Y[prepped_state] = dy[:,:self.nro_delta_y].sum(1)
            plt.plot(dx[:,:self.nro_delta_x].sum(1),dy[:,:self.nro_delta_y].sum(1),'o',label = f'state {prepped_state}',alpha = 0.2)
        
        plt.xlabel('Delta X  = (0+1) - (2+3)')
        plt.ylabel('Delta Y  = (0+1) - (2+3)')
        plt.legend()
        figs.append(fig1)
        fig2,ax = plt.subplots(6,len(self.states_to_prepare),figsize = (20,20))

        for i,prepped_state in enumerate(self.states_to_prepare):
            
            
            ax[0,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,0].sum(1),alpha = 0.5,color = colors[0],label = 'state 0')
            ax[0,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,1].sum(1), alpha = 0.5,color = colors[1],label = 'state 1')
            ax[0,i].set_title(f'Preparing state {i} X positive counts distribution')
            ax[0,i].legend()
            
            ax[1,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,2].sum(1),alpha = 0.5,color = colors[2],label = 'state 2')
            ax[1,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,3].sum(1), alpha = 0.5,color = colors[3],label = 'state 3')
            ax[1,i].set_title('X negative counts distribution')
            ax[1,i].legend()
            
            ax[2,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,0].sum(1) + clicksX[:,prepped_state,:self.nro_delta_x,1].sum(1),color = 'brown',alpha = 0.3,label = 'state 1+0')
            ax[2,i].hist(clicksX[:,prepped_state,:self.nro_delta_x,2].sum(1) + clicksX[:,prepped_state,:self.nro_delta_x,3].sum(1),color = 'black', alpha = 0.3,label = 'state 2+3')
            ax[2,i].set_title('X positive and negative counts distribution')
            ax[2,i].legend()
            
            
            ax[3,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,0].sum(1),alpha = 0.5,color = colors[0],label = 'state 0')
            ax[3,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,2].sum(1), alpha = 0.5,color = colors[2],label = 'state 2')
            ax[3,i].set_title(f' Y positive counts distribution')
            ax[3,i].legend()
            
            ax[4,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,3].sum(1),alpha = 0.5,color = colors[3],label = 'state 3')
            ax[4,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,1].sum(1), alpha = 0.5,color = colors[1],label = 'state 1')
            ax[4,i].set_title('Y negative counts distribution')
            ax[4,i].legend()
            
            ax[5,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,0].sum(1) + clicksX[:,prepped_state,:self.nro_delta_y,2].sum(1),color = 'brown',alpha = 0.3,label = 'state 0+2')
            ax[5,i].hist(clicksX[:,prepped_state,:self.nro_delta_y,3].sum(1) + clicksX[:,prepped_state,:self.nro_delta_y,1].sum(1),color = 'black', alpha = 0.3,label = 'state 1+3')
            ax[5,i].set_title('Y positive and negative counts distribution')
            ax[5,i].legend()
        
        
        figs.append(fig2)
        states_points  = np.array([np.stack((delta_X[i], delta_Y[i]),axis= 1) for i in range(4)])
        states_points_nro  = np.array([np.stack((delta_X_nro[i,:], delta_Y_nro[i,:]),axis= 1) for i in range(4)])
        print("state points shape is ", states_points_nro.shape)
        labels = np.zeros(states_points.shape[:2], dtype=int)
        labels_nro = np.zeros(states_points.shape[:2], dtype=int)-1
        fidelity = np.zeros((4,self.nuclear_nro))
        fidelity_fig = plt.figure()
        for i, state  in enumerate(states_points):
            
            
            for nro in range(1,self.nuclear_nro):
                
                _dx = states_points_nro[i][:, 0,:nro].sum(1)
                _dy = states_points_nro[i][:, 1,:nro].sum(1)
                labels_nro[i, (_dx > 0) & (_dy > 0)] = 0
                labels_nro[i, (_dx < 0) & (_dy > 0)] = 2
                labels_nro[i, (_dx > 0) & (_dy< 0)] = 1
                labels_nro[i, (_dx < 0) & (_dy< 0)] = 3
                
                a = np.array([np.sum(labels_nro[i,:]==0),np.sum(labels_nro[i,:]==1),np.sum(labels_nro[i,:]==2),np.sum(labels_nro[i,:]==3)])
                if nro == 100:
                    print(a)
                fidelity[i,nro] = a[i]/np.sum(a)
                
            labels[i, (state[:, 0] > 0) & (state[:, 1] > 0)] = 0
            labels[i, (state[:, 0] < 0) & (state[:, 1] > 0)] = 2
            labels[i, (state[:, 0] > 0) & (state[:, 1] < 0)] = 1
            labels[i, (state[:, 0] < 0) & (state[:, 1] < 0)] = 3
            
            
            plt.plot(np.arange(1,self.nuclear_nro),fidelity[i,1:],label = f'state {i} max fidelity {np.max(fidelity[i,:]):.3f} for nro = {np.argmax(fidelity[i,:])}')
        
        
        figs.append(fidelity_fig) 
        plt.legend()
        plt.show()
        
        coords = np.concatenate(states_points_nro[i])
        x_fidelity = []
        y_fidelity = []
        
        
        
        # for nro in range(1,self.nuclear_nro):
            
        #    print('coords', coords.shape)
            
        #    _dx = coords[:, 0,:nro].sum(1)
        #    print('_dx', _dx.shape)
        #    x_fidelity.append(np.sum(_dx>0)/coords.shape(0))
        #    y_fidelity.append(np.sum(_dy>0)/coords.shape(0))
        
        # plt.figure()
        # plt.plot(np.arange(1,self.nuclear_nro),x_fidelity[1:])
        # plt.plot(np.arange(1,self.nuclear_nro),y_fidelity[1:])
        # plt.show()


        
        
        

        
        fig9 = Plot_State_prep_matrix2(labels,states_points)
        fig9.suptitle('Without Kmeans')
        
        fig5,labels = Get_Centers_and_labels(states_points)
        # figs.append(fig5)
        
        fig7 = Plot_State_prep_matrix2(labels,states_points)
        fig7.suptitle('With Kmeans')
        
        # fig5,fig7 = Kmeans_ro2(states_points)
        # figs.append(fig7)
        
        self.save(
            figure=figs,
            dataset={
                **signal,
            },
        )



def Get_Centers_and_labels(states_points):

    init_centroids = []
    stateA,stateB,stateC,stateD = states_points
    Ro_coordinates = np.concatenate([stateA, stateB, stateC, stateD])

    labels = []
    for state in states_points:
        center = np.array([np.mean(state[:,0],),np.mean(state[:,1])])
        init_centroids.append(center)
     
    init_centroids = np.array(init_centroids)   
    K_object = KMeans(n_clusters=4,init=init_centroids)
    K_object.fit(Ro_coordinates)
    
    
    # centroids = np.array(centroids)
    centroids = K_object.cluster_centers_


    for state in states_points:
        
        dist = np.zeros((state.shape[0],4))

        for i in range(state.shape[0]):
        
            for c in range(len(centroids)):
            
                dist[i,c] = np.sum((state[i]-centroids[c])**2)
            
        # labels.append(np.argmin(dist,axis = 1))
        labels.append(K_object.predict(state))
                
    
    print(centroids)
    colors=['blue','orange','green','red']

    mx = np.min(Ro_coordinates[:,0])
    Mx = np.max(Ro_coordinates[:,0])
    my = np.min(Ro_coordinates[:,1])
    My = np.max(Ro_coordinates[:,1])
    # ---- create a grid covering the plane ----
    x = np.linspace(mx, Mx, 400)
    y = np.linspace(my, My, 400)

    dx = x[1]-x[0]
    dy = y[1]-y[0]
    # ---- compute distance to each centroid ----
    distances = np.zeros((y.shape[0],x.shape[0],4))
    distances_map = np.zeros((y.shape[0],x.shape[0]))

    for i in range(len(x)):
        for j in range(len(y)):
            for c in range(len(centroids)):
                distances[j, i, c] = np.sum((np.array([x[i],y[j]]) - centroids[c])**2)  
            distances_map[j, i] = np.argmin(distances[j, i, :])

    cmap = mcolors.ListedColormap([
        "blue",      # value 0
        "orange",     # value 1
        "red",    # value 2
        "green"    # value 3
    ])

    # Bin edges around the integers 0,1,2,3
    norm = mcolors.BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5],
        ncolors=cmap.N
    )
    
    fig = plt.figure(figsize=(14,14))
    fig.suptitle('Kmeans assigned clustering')
    # ---- plot decision regions ----
    labels2 = K_object.labels_
    # ---- plot centroids ----
    plt.scatter(Ro_coordinates[np.where(labels2==0), 0], Ro_coordinates[np.where(labels2==0), 1], alpha = 0.3, label='State 0')
    plt.scatter(Ro_coordinates[np.where(labels2==1), 0], Ro_coordinates[np.where(labels2==1), 1], alpha = 0.3, label='State 1')
    plt.scatter(Ro_coordinates[np.where(labels2==2), 0], Ro_coordinates[np.where(labels2==2), 1], alpha = 0.3, label='State 2')
    plt.scatter(Ro_coordinates[np.where(labels2==3), 0], Ro_coordinates[np.where(labels2==3), 1], alpha = 0.3, label='State 3')


    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
    plt.scatter(init_centroids[:, 0], init_centroids[:, 1], color='yellow', marker='X', s=100, label='init Centroids')

    fig = plt.figure(figsize=(14,14))
    
    fig.suptitle('Kmeans fitted decision regions')

    # ---- plot decision regions ----
    plt.imshow(distances_map,extent=(mx-dx/2,Mx+dx/2,my-dy/2,My+dy/2),origin='lower',cmap=cmap,norm=norm,alpha=0.3)

    # ---- plot centroids ----
    plt.scatter(stateA[:, 0], stateA[:, 1], alpha = 0.3, label='State 0')
    plt.scatter(stateB[:, 0], stateB[:, 1], alpha = 0.3, label='State 1')
    plt.scatter(stateC[:, 0], stateC[:, 1], alpha = 0.3, label='State 2')
    plt.scatter(stateD[:, 0], stateD[:, 1], alpha = 0.3, label='State 3')

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
    
    return fig,labels






def Plot_State_prep_matrix2(labels,states_points):
    
    state_matrix = np.zeros((4, 4))

    for i in range(4):
        for j in range(4):
            state_matrix[i, j] = (labels[i]==j).sum(0)/len(states_points[i])
                
    labels = ['00', '01', '10', '11']  # adjust if different ordering

    fig4, ax = plt.subplots()
    im = ax.imshow(state_matrix)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Probability")

    # Axis ticks and labels
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    ax.set_xlabel("Measured state")
    ax.set_ylabel("Prepared state")

    # Annotate values inside squares
    for i in range(state_matrix.shape[0]):
        for j in range(state_matrix.shape[1]):
            value = state_matrix[i, j]
            ax.text(j, i,
            f"{value:.4f}",
                        ha="center",
                        va="center",
                        color="white" if value < 0.5 else "black")

    ax.set_aspect("equal")

    return(fig4)

# def Plot_State_prep_matrix(centroids,states_points):
    
#     state_matrix = np.zeros((4, 4))

#     kmeans = KMeans(
#         n_clusters=4,
#         init=centroids,
#         n_init=1
#     )
    
#     Ro_coordinates = np.vstack(states_points)

#     kmeans.fit(Ro_coordinates)

#     for i in range(4):
#         for j in range(4):
#             state_matrix[i, j] = (kmeans.predict(states_points[i])==j).sum(0)/len(states_points[i])


#     print("mean fidelity = %.1f%%"%(100*np.diag(state_matrix).mean()))           


#     labels = ['00', '01', '10', '11']  # adjust if different ordering

#     fig4, ax = plt.subplots()
#     im = ax.imshow(state_matrix)

#     # Colorbar
#     cbar = plt.colorbar(im, ax=ax)
#     cbar.set_label("Probability")

#     # Axis ticks and labels
#     ax.set_xticks(range(len(labels)))
#     ax.set_yticks(range(len(labels)))
#     ax.set_xticklabels(labels)
#     ax.set_yticklabels(labels)

#     ax.set_xlabel("Measured state")
#     ax.set_ylabel("Prepared state")

#     # Annotate values inside squares
#     for i in range(state_matrix.shape[0]):
#         for j in range(state_matrix.shape[1]):
#             value = state_matrix[i, j]
#             ax.text(j, i,
#             f"{value:.4f}",
#                         ha="center",
#                         va="center",
#                         color="white" if value < 0.5 else "black")

#     ax.set_aspect("equal")

#     plt.tight_layout()
#     plt.show()
#     return(fig4)

# def Kmeans_ro(states_points):
    
#     stateA,stateB,stateC,stateD = states_points
#     Ro_coordinates = np.vstack([stateA, stateB, stateC, stateD])

#     centroids = []

#     for coords in states_points:
#         K_object = KMeans(n_clusters=1).fit(coords)
#         labels = K_object.labels_
#         centroid = K_object.cluster_centers_[0]
#         centroids.append(centroid)

#     centroids = np.array(centroids)

#     colors=['blue','orange','green','red']


#     # ---- create a grid covering the plane ----
#     x = np.linspace(min(Ro_coordinates[:,0]), max(Ro_coordinates[:,0]), 400)
#     y = np.linspace(min(Ro_coordinates[:,1]), max(Ro_coordinates[:,1]), 400)
#     xx, yy = np.meshgrid(x, y)
#     grid_points = np.c_[xx.ravel(), yy.ravel()]

#     # ---- compute distance to each centroid ----
#     distances = np.zeros((grid_points.shape[0], 4))
#     for i in range(4):
#         distances[:, i] = np.sum((grid_points - centroids[i])**2, axis=1)

#     # ---- assign nearest centroid ----
#     labels = np.argmin(distances, axis=1)
#     labels = labels.reshape(xx.shape)

#     fig = plt.figure()
#     # ---- plot decision regions ----
#     plt.contourf(xx, yy, labels, alpha=0.2, colors=colors,levels=[-0.5,0.5,1.5,2.5,3.5])

#     # ---- plot decision boundary lines ----
#     # contour at boundaries between regions
#     plt.contour(xx, yy, labels, levels=[0,1,2,3], colors='dimgrey', alpha = 0.5,linewidths=1)

#     # ---- plot centroids ----
#     plt.scatter(stateA[:, 0], stateA[:, 1], alpha = 0.5, label='State 0')
#     plt.scatter(stateB[:, 0], stateB[:, 1], alpha = 0.5, label='State 1')
#     plt.scatter(stateC[:, 0], stateC[:, 1], alpha = 0.5, label='State 2')
#     plt.scatter(stateD[:, 0], stateD[:, 1], alpha = 0.5, label='State 3')

#     plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
#     plt.legend()

#     plt.xlabel("ΔX")
#     plt.ylabel("ΔY")
#     plt.title("Nearest-centroid classification with decision boundaries")

#     fig4 = Plot_State_prep_matrix(centroids,states_points)
    
#     return fig,fig4




def Kmeans_ro2(states_points):
    
    stateA,stateB,stateC,stateD = states_points
    Ro_coordinates = np.concatenate([stateA, stateB, stateC, stateD])
    from sklearn.metrics import pairwise_distances_argmin

    centroids = np.array([[-60,-20],[-60,20],[60,-20],[100,20]])

    labels = pairwise_distances_argmin(Ro_coordinates, centroids)
    print(labels)

    colors=['blue','orange','green','red']


    # ---- create a grid covering the plane ----
    x = np.linspace(min(Ro_coordinates[:,0]), max(Ro_coordinates[:,0]), 1000)
    y = np.linspace(min(Ro_coordinates[:,1]), max(Ro_coordinates[:,1]), 1000)
    xx, yy = np.meshgrid(x, y)
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # ---- compute distance to each centroid ----
    distances = np.zeros((grid_points.shape[0], 4))
    for i in range(4):
        distances[:, i] = np.sum((grid_points - centroids[i])**2, axis=1)

    # ---- assign nearest centroid ----
    labels = np.argmin(distances, axis=1)
    labels = labels.reshape(xx.shape)

    fig = plt.figure()
    # ---- plot decision regions ----
    plt.contourf(xx, yy, labels, alpha=0.2, colors=colors,levels=[-0.5,0.5,1.5,2.5,3.5])

    # ---- plot decision boundary lines ----
    # contour at boundaries between regions
    #plt.contour(xx, yy, labels, levels=[-0.5,0.5,1.5,2.5,3.5], colors='dimgrey', alpha = 0.5,linewidths=1)

    # ---- plot centroids ----
    plt.scatter(stateA[:, 0], stateA[:, 1], alpha = 0.5, label='State 0')
    plt.scatter(stateB[:, 0], stateB[:, 1], alpha = 0.5, label='State 1')
    plt.scatter(stateC[:, 0], stateC[:, 1], alpha = 0.5, label='State 2')
    plt.scatter(stateD[:, 0], stateD[:, 1], alpha = 0.5, label='State 3')

    plt.scatter(centroids[:, 0], centroids[:, 1], color='black', marker='X', s=100, label='Centroids')
    plt.legend()

    plt.xlabel("ΔX")
    plt.ylabel("ΔY")
    plt.title("Nearest-centroid classification with decision boundaries")

    fig4 = Plot_State_prep_matrix(centroids,states_points)
    
    return fig,fig4

      
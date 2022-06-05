import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_regular


def stim_protocol_rebound():

    """
    Test different negative amplitudes and durations of negative current 
    to observe if presence of a rebound spike in regular HH neuron
    """

    ### DIFFERENT CURRENT AMPLITUDES ###

    # Definition of the different amplitudes of negative pulse currents
    I = np.linspace(0, -20, 10)
    
    
    # Define the plots of Vm and I_ext for different amplitudes
    fig1,ax_amp = plt.subplots(2,1)
    ax_amp[0].set_xlabel("t [ms]")
    ax_amp[0].set_ylabel("v [mV]")
    ax_amp[0].grid()
    ax_amp[1].set_xlabel("t [ms]")
    ax_amp[1].set_ylabel("$I_{e}$ [$\mu$ A]")
    ax_amp[1].grid()


    
    # Run the simulation for different amplitudes of negative step currents
    for I_ in I:
        current = input_factory.get_step_current(10, 45, b2.ms, I_ * b2.uA)
        state_monitor = simulate_HH_neuron_regular(current, 110 * b2.ms)


        ax_amp[0].plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
        ax_amp[1].plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
        
    ax_amp[1].axis((
            0,
            np.max(state_monitor.t / b2.ms),
            min(state_monitor.I_e[0] / b2.uamp) * 1.1,
            max(state_monitor.I_e[0] / b2.uamp) * 1.1+0.1
        ))
    fig1.suptitle('Stimulation protocol for Rebound Spike: different amplitudes')


    ### DIFFERENT STEP CURRENT DURATIONS ###

    # Define the plots of Vm and I_ext for different durations
    fig2,ax_dur = plt.subplots(2,1)
    ax_dur[0].set_xlabel("t [ms]")
    ax_dur[0].set_ylabel("v [mV]")
    ax_dur[0].grid()
    ax_dur[1].set_xlabel("t [ms]")
    ax_dur[1].set_ylabel("$I_{e}$ [$\mu$ A]")
    ax_dur[1].grid()

    # Definition of the different durations of negative pulse currents
    durs = [10, 20, 45, 60]

     # Run the simulation for different durations of negative step currents
    for dur_ in durs:
        current = input_factory.get_step_current(10, dur_, b2.ms, -7.2 * b2.uA)
        state_monitor = simulate_HH_neuron_regular(current, 120 * b2.ms)


        ax_dur[0].plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
        ax_dur[1].plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
        
    ax_dur[1].axis((
            0,
            np.max(state_monitor.t / b2.ms),
            min(state_monitor.I_e[0] / b2.uamp) * 1.1,
            max(state_monitor.I_e[0] / b2.uamp) * 1.1+0.1
        ))
    fig2.suptitle('Stimulation protocol for Rebound Spike: different durations')
    plt.show()


def plot_x_inf_tau(vm, minf, ninf, hinf, tm, tn, th):
    """Function to plot the variables x_inf and tau."""

    fig,ax = plt.subplots(3,1)
    ax[0].plot(vm, minf, "black", lw=2)
    ax[0].set_xlabel("$V_m$ [mV]")
    ax[0].set_ylabel("$m_{\infty}$")
    ax[0].grid()

    ax[1].plot(vm, ninf, "blue", lw=2)
    ax[1].set_xlabel("$V_m$ [mV]")
    ax[1].set_ylabel("$n_{\infty}$")
    ax[1].grid()

    ax[2].plot(vm, hinf, "red", lw=2)
    ax[2].set_xlabel("$V_m$ [mV]")
    ax[2].set_ylabel(r"$h_{\infty}$")
    ax[2].grid()
    

    fig,ax = plt.subplots(3,1)
    ax[0].plot(vm, tm, "black", lw=2)
    ax[0].set_xlabel("$V_m$ [mV]")
    ax[0].set_ylabel(r"$\tau_m$ [ms]")
    ax[0].grid()

    ax[1].plot(vm, tn, "blue", lw=2)
    ax[1].set_xlabel("$V_m$ [mV]")
    ax[1].set_ylabel(r"$\tau_n$ [ms]")
    ax[1].grid()

    ax[2].plot(vm, th, "red", lw=2)
    ax[2].set_xlabel("$V_m$ [mV]")
    ax[2].set_ylabel(r"$\tau_h$ [ms]")
    ax[2].grid()

    plt.show()



    
def gate_var_simul():

    """ Function to plot the gating variables x_inf and tau 
    as a function of the membrane voltage for our simulation."""

    # Stimulation of the regular HH neuron.
    current = input_factory.get_step_current(10, 45, b2.ms, -7.2 * b2.uA)
    state_monitor = simulate_HH_neuron_regular(current, 120 * b2.ms)

    # Compute the x_inf and load tau :
    V_m =state_monitor.vm[0]/ b2.mV
    Th =state_monitor.th[0] / b2.ms
    Tn =state_monitor.tn[0] / b2.ms
    Tm =state_monitor.tm[0] / b2.ms
    N_inf =state_monitor.ninf[0] / b2.volt
    M_inf =state_monitor.minf[0] / b2.volt
    H_inf =state_monitor.hinf[0] / b2.volt

    N =state_monitor.n[0] / b2.volt
    M =state_monitor.m[0] / b2.volt
    H =state_monitor.h[0] / b2.volt
    T = state_monitor.t / b2.ms

    #Plots: 
    
    fig,ax = plt.subplots(3,1)
    ax[0].plot(T, N, "black", lw=2)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("n")
    ax[0].grid()

    ax[1].plot(T, H, "blue", lw=2)
    ax[1].set_xlabel("time")
    ax[1].set_ylabel(r"h")
    ax[1].grid()

    ax[2].plot(T, M, "red", lw=2)
    ax[2].set_xlabel("time")
    ax[2].set_ylabel("m")
    ax[2].grid()

    plot_x_inf_tau(V_m, M_inf, N_inf, H_inf,Tm,Tn, Th)


    

def gate_var_analytique():

    """ 
    Function to plot the analytical behavior of the gating variables
    x_inf and tau as a function of the membrane voltage.
    """

    # Range of membrane voltage for which plot x_inf and tau
    vm = np.linspace(-100, 50, 100)

    # Analytical expression of the gating variables
    alphah = 0.128*np.exp(-(vm+43)/18)
    alpham = -0.32*(47+vm)/(np.exp(-0.25*(vm+47))-1)
    alphan = -0.032*(45+vm)/(np.exp(-0.2*(vm+45))-1)
    betah = 4./(1+np.exp(-0.2*(vm + 20)))
    betam = 0.28*(vm + 20)/(np.exp(0.2*(vm + 20))-1)
    betan = 0.5*np.exp(-(vm + 50)/40)
    minf = alpham/(alpham+betam)
    ninf = alphan/(alphan+betan)
    hinf = alphah/(alphah+betah)
    tm = 1/(alpham + betam)
    tn = 1/(alphan + betan)
    th = 1/(alphah + betah)

    # Plot
    plot_x_inf_tau(vm,minf,ninf,hinf,tm,tn,th)

   


if __name__ == "__main__":
    
    stim_protocol_rebound() #Stimulation protocol
    gate_var_simul() #Plot of x_inf and tau for our simulation
    gate_var_analytique() #Plot of x_inf and tau for a big range of membrane pot.

import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_adaptative
from implementation_HH import plot_data
from scipy.signal import find_peaks


def spike_timings(state_monitor) : 
    """function extracting spike timing from voltage trace"""
    Volt= state_monitor.vm[0] / b2.mV
    time = state_monitor.t / b2.ms
    time_spike = []
    indice_spike = find_peaks(Volt)
    time_spike = time[indice_spike[0]]
    diff_spiking_time = []
    for i in range(len(time_spike)-1) :
        diff_spiking_time.append(time_spike[i+1]-time_spike[i])

    return time_spike,diff_spiking_time

def figure_adaptative(state_monitor,several_plot = False) : 
    
    time_spike,diff_spiking_time = spike_timings(state_monitor)
    if several_plot == False : # case where i just have one value of coefficient a,b,c respectively.
        plt.figure()
    plt.plot(time_spike[1:],diff_spiking_time)
    plt.xlabel("t [ms]")
    plt.ylabel("delta t [ms]")
    plt.grid()
    if several_plot == False : # case where i just have one value of coefficient a,b,c respectively.
        plt.show()

def volting_trace(state_monitor) :
    plt.figure()
    plt.plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
    plt.xlabel("t [ms]")
    plt.ylabel("v [mV]")
    plt.title("Voltage trace")
    plt.grid()
    plt.show()


def simulate_HH_neuron_adaptative_change_parameter(input_current, simulation_time,a=1,b=1,c=1):

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "I_na", "I_k", "I_m", "m", "n", "h", "p","a","b","c"]
    """

    # neuron parameters
    El = -69 * b2.mV
    EK = -90 * b2.mV
    ENa = 50 * b2.mV
    gl = 0.1 * b2.msiemens
    gK = 5 * b2.msiemens
    gNa = 50 * b2.msiemens
    C = 1 * b2.ufarad
    gM = 0.07 * b2.msiemens

    # forming HH model with differential equations
    eqs = """
    I_e = input_current(t,i) : amp
    membrane_Im = I_e - I_na + gl*(El-vm) - I_k - I_m : amp
    alphah = .128*exp(-(vm+43*mV)/(18*mV))/ms : Hz
    alpham = -.32*(47*mV+vm)/(exp(-0.25*(vm/mV+47))-1)/mV/ms : Hz
    alphan = -.032*(45*mV+vm)/(exp(-0.2*(vm/mV+45))-1)/mV/ms : Hz
    betah = 4./(1+exp(-0.2*(vm/mV + 20)))/ms : Hz
    betam = .28*(vm/mV + 20)/(exp(0.2*(vm/mV + 20))-1)/ms : Hz
    betan = .5*exp(-(vm/mV + 50)/40)/ms : Hz
    pinf = a*1./(exp(-0.1*(vm/mV+40))+1) : 1
    tau_p = b*2000/(3.3 * exp((vm/mV + 20)/20)+exp(-(vm/mV + 20)/20))*ms : second
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dp/dt = (pinf-p)/tau_p : 1
    dvm/dt = membrane_Im/C : volt
    I_na = -gNa*m**3*h*(ENa-vm) : amp
    I_k = -gK*n**4*(EK-vm) : amp
    I_m = -c*gM*p*(EK-vm) : amp

    """

    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization
    neuron.vm = -70* b2.mV
    neuron.m = 0.0
    neuron.h = 1.0
    neuron.n = 0.0
    neuron.p = 0.05

    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e","I_na", "I_k", "I_m", "m", "n", "h", "p"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon

def first_condition(current) : 
    t_coef = np.array([0.5,1,2])
    plt.figure()
    legend =[]
    for j in range(len(t_coef)) :
        state_monitor_adaptative_changes = simulate_HH_neuron_adaptative_change_parameter(current, 1600 * b2.ms,b=t_coef[j])
        #plot_data(state_monitor_adaptative_changes, type='adaptative', title="HH Neuron, step current, adaptative")
        figure_adaptative(state_monitor_adaptative_changes,several_plot = True)
        legend.append(t_coef[j]) 
    plt.legend(legend) 
    plt.title("Adaptative figure for different values of coefficient multiplying the original tau_p")
    plt.show()

def second_condition(current) :
    p_coef = np.array([0.75,1,1.25])
    plt.figure()
    legend =[]
    for j in range(len(p_coef)) :
        state_monitor_adaptative_changes = simulate_HH_neuron_adaptative_change_parameter(current, 1600 * b2.ms,a=p_coef[j])
        #plot_data(state_monitor_adaptative_changes, type='adaptative', title="HH Neuron, step current, adaptative")
        figure_adaptative(state_monitor_adaptative_changes,several_plot = True)
        legend.append(p_coef[j]) 
    plt.legend(legend) 
    plt.title("Adaptative figure for different coefficient a, with pinf = a/(exp(-0.1(V+40))+1)")
    plt.show() 

    fig,ax = plt.subplots(len(p_coef),1)
    for j in range(len(p_coef)) :
        state_monitor_adaptative_changes = simulate_HH_neuron_adaptative_change_parameter(current, 1600 * b2.ms,a=p_coef[j])
        ax[j].plot(state_monitor_adaptative_changes.t / b2.ms, state_monitor_adaptative_changes.vm[0] / b2.mV, lw=2)
        ax[j].set_xlabel("t [ms]")
        ax[j].set_ylabel("v [mV]")
        #ax[j].set_legend(p_coef[j])
        ax[j].grid()
    plt.show()
    

def third_condition(current) :
    p_coef = np.array([1,-1])
    plt.figure()
    legend =[]
    for j in range(len(p_coef)) :
        state_monitor_adaptative_changes = simulate_HH_neuron_adaptative_change_parameter(current, 1600 * b2.ms,a=p_coef[j])
        #plot_data(state_monitor_adaptative_changes, type='adaptative', title="HH Neuron, step current, adaptative")
        figure_adaptative(state_monitor_adaptative_changes,several_plot = True)
        legend.append(p_coef[j]) 
    plt.legend(legend) 
    plt.title("Adaptative figure for different coefficient a, with pinf = a/(exp(-0.1(V+40))+1)")
    plt.show() 




if __name__ == "__main__":

    current =  input_factory.get_step_current(10, 1510, b2.ms, 2.0 * b2.uA)

    #beginning 1.3
    #state_monitor_adaptative = simulate_HH_neuron_adaptative(current, 1600 * b2.ms)
    #plot_data(state_monitor_adaptative, type='adaptative', title="HH Neuron, step current, adaptative")
    #figure_adaptative(state_monitor_adaptative)

    ###change of parameters : 
    #1.first condition : slow down the adaptation rate : increase the tp, the relaxation time constant for p
    #first_condition(current)

    #2.second condition : Decrease the stable firing rate without changing the adaptation rate : change the value of pinf ? not working yet
    #second_condition(current)
    
    #3.third condition : Reverse the adaptation: stable firing rate lower than initial firing rate : change IM ?not working, still on it
    #third_condition(current)

    IM_coef = np.array([-0.75,1,1.25])
    plt.figure()
    legend =[]
    for j in range(len(IM_coef)) :
        state_monitor_adaptative_changes = simulate_HH_neuron_adaptative_change_parameter(current, 1600 * b2.ms,c=IM_coef[j])
        #plot_data(state_monitor_adaptative_changes, type='adaptative', title="HH Neuron, step current, adaptative")
        figure_adaptative(state_monitor_adaptative_changes,several_plot = True)
        legend.append(IM_coef[j]) 
    plt.legend(legend) 
    plt.title("Adaptative figure for different coefficient a, with pinf = a/(exp(-0.1(V+40))+1)")
    plt.show() 
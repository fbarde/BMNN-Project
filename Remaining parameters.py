from neurodynex3.adex_model import AdEx
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from implementation_HH import simulate_HH_neuron_adaptative

if __name__ == "__main__":

    test_current = input_factory.get_step_current(0, 1500, 1*b2.ms, 2*b2.uA)  #-0.9243 

    ## HH model to copy 
    step_current = input_factory.get_step_current(0, 1500, 1*b2.ms, 2 *b2.uA)
    HH_state_monitor_adaptive = simulate_HH_neuron_adaptative(step_current, simulation_time=1500 * b2.ms)
    #plot_data_adaptive(HH_state_monitor_adaptive, 1500, title="HH Neuron, adaptive")

    Adex_state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=test_current, simulation_time=1500 * b2.ms, tau_m=9.2* b2.ms, 
    R=9.49628545* b2.kohm, v_rest=-70.60737437 * b2.mV, v_rheobase=-55.75543484 * b2.mV, delta_T= 1.9633764705215981* b2.mV , a=10655.855814888468 * b2.nS, 
    tau_w=70* b2.ms, v_spike = 30 * b2.mV, b= 10* b2.pA, v_reset= -70.60737437 * b2.mV) #-68.0 * b2.mV 
    plt.figure(figsize=(15,10))
    plt.plot(HH_state_monitor_adaptive.t / b2.ms, HH_state_monitor_adaptive.vm[0] / b2.mV, '-b')
    plt.plot(Adex_state_monitor.t / b2.ms, Adex_state_monitor.v[0] / b2.mV, '-g')
    plt.legend(('HH adaptive neuron', 'AdEx neuron'))
    plt.grid()
    plt.xlabel('t [ms]')
    plt.ylabel('v [mV]')
    plt.show()
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory
from neurodynex3.adex_model import AdEx
from implementation_HH import simulate_HH_neuron_adaptative
from adaptation import spike_timings
from Remaining_parameters import spike_timings_AdEx

# Set the random seed for reproductability 
np.random.seed(1)

def plot_voltage_trace_and_spike_timing(state_monitor_1, state_monitor_2, time_spike1, time_spike2):

    """
    Function to plot the voltage traces and mark the spikes.

    Args:
        state_monitor_1: state monitor of HH adaptative neuron
        state_monitor_2: state monitor of AdEx neuron model
        time_spike_1 : time of the spikes of the first state monitor
        time_spike_2 : time of spikes of the second state monitor

    """

    # Define the marks for the spikes
    mark_1 = 55*np.ones((len(time_spike1)))
    mark_2 = 55*np.ones((len(time_spike2)))

    # Plot of the voltage traces:
    plt.figure()
    plt.plot(state_monitor_1.t / b2.ms, state_monitor_1.vm[0] / b2.mV, '-b', label='HH adaptive neuron')
    plt.plot(state_monitor_2.t / b2.ms, state_monitor_2.v[0] / b2.mV, '-r', label='AdEx neuron model')
    #Marking the spikes:
    plt.plot(time_spike2, mark_2, 'rx', label='Spike AdEx neuron model', lw=2 )
    plt.plot(time_spike1, mark_1, 'b+', label='Spike HH adaptive neuron',lw=2 )
    plt.legend(loc='lower right')
    plt.grid()
    plt.xlabel(r'$t$ [ms]')
    plt.ylabel(r'$V_m$ [mV]')
    plt.suptitle(r'HH adaptative neuron and AdEx model with gaussian random current.')
    plt.show()


def gaussian_random_current(t_start, t_end, unit_time, amp_mu, amp_sigma, append_zero=True):
    """Creates a gaussian random current. The amplitude of current has a 
    Gaussian distribution.

    Args:
        t_start (int): start of the step
        t_end (int): end of the step
        unit_time (Brian2 unit): unit of t_start and t_end. 
        amp_mu (Quantity): mean amplitude of the gaussian. 
        amp_sigma (Quantity): varriance amplitude of the gaussian.
        append_zero (bool, optional): if true, 0Amp is appended at t_end+1.
        Without that trailing 0, Brian reads out the last value in the array (=amplitude) for all indices > t_end.

    Returns:
        TimedArray: Brian2.TimedArray
    """

    assert isinstance(t_start, int), "t_start_ms must be of type int"
    assert isinstance(t_end, int), "t_end must be of type int"
    assert b2.units.fundamentalunits.have_same_dimensions(amp_mu, b2.amp), \
        "amplitude must have the dimension of current e.g. brian2.uamp"
    assert b2.units.fundamentalunits.have_same_dimensions(amp_sigma, b2.amp), \
        "amplitude must have the dimension of current e.g. brian2.uamp"
    tmp_size = 1 + t_end  # +1 for t=0
    if append_zero:
        tmp_size += 1
    tmp = np.zeros((tmp_size, 1)) * b2.amp
    if t_end > t_start:  # if deltaT is zero, we return a zero current
        dur=len(range(0, (t_end - t_start) + 1))
        tmp[t_start: t_end + 1, 0] = amp_sigma * np.random.randn(dur) + amp_mu

    curr = b2.TimedArray(tmp, dt=1. * unit_time)
    return curr


def stimulate_gaussian_random(dur = 500):

    """
    Function to stimulate HH adaptative neuron and AdEx neuron with a gaussian 
    random current input.

    Args: 
        dur: duration of the stimulation
    """

    # Set the parameters here of the AdEx neuron model :
    V_RESET= -77.2
    B= 45.35 
    TAU_W= 295 

    # Define the gaussian random current  
    current = gaussian_random_current(0, dur, b2.ms, 1.0*b2.uA, 15*b2.uA)

    # HH adaptative neuron model
    HH_state_monitor_adaptive = simulate_HH_neuron_adaptative(current, simulation_time= dur * b2.ms)
    
    # Adex model : need to find the right parameters:
    AdEx_state_monitor, spike_monitor = AdEx.simulate_AdEx_neuron(I_stim=current, simulation_time= dur * b2.ms, tau_m=9.0* b2.ms, 
    R=9.49628545* b2.kohm, v_rest=-70.60737437 * b2.mV, v_rheobase=-55.75543484 * b2.mV, delta_T= 1.9633764705215981* b2.mV , a=10655.855814888468 * b2.nS, 
    tau_w= TAU_W* b2.ms, v_spike = 30 * b2.mV, b= B* b2.nA, v_reset= V_RESET* b2.mV)  
    
    # Record the spike times:
    time_spike_adapt,diff_spiking_time_adapt = spike_timings(HH_state_monitor_adaptive)
    time_spike_AdEx, diff_spiking_time_AdEx = spike_timings_AdEx(AdEx_state_monitor)

    print('Number of spikes for HH adaptative neuron:',len(time_spike_adapt))
    print('Number of spikes for AdeX model:',len(time_spike_AdEx))

    # Plot the gaussian random current I_ext :
    plt.figure()
    plt.plot(HH_state_monitor_adaptive.t / b2.ms, HH_state_monitor_adaptive.I_e[0] / b2.uamp, "red", lw=2)
    plt.xlabel('t [ms]')
    plt.ylabel(r'$I_{ext}$ [$\mu$ A]')
    plt.suptitle(r'Gaussian random current $I_{ext}$')
    plt.grid()

    # Plot the overlap of the two voltage traces and marking of the spikes
    plot_voltage_trace_and_spike_timing(HH_state_monitor_adaptive, AdEx_state_monitor, time_spike_adapt, time_spike_AdEx)


if __name__ == "__main__":

    stimulate_gaussian_random(dur=500)
    stimulate_gaussian_random(dur=2500)


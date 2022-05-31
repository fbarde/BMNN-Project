from turtle import color
import brian2 as b2
import matplotlib.pyplot as plt
import numpy as np
import neurodynex3.tools.input_factory as input_factory


############# DEFINITION OF THE PLOTTING FUNCTIONS ##############


def plot_data2(state_monitor, type='regular', title=None):
    """Plots the state_monitor variables ["vm", "I_e" "I_na", "I_k", "I_m", "m", "n", "h", "p"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        type (string, optional): 'regular' or 'adaptative', indicate the type of HH model we are using: 
        doesn't produce the same plots
        title (string, optional): plot title to display
    """
    # Plot of membrane potential (vm), external current I_e, and other current I_a, I_k
    
    if type == 'adaptative' :
        fig,ax = plt.subplots(4,1)
    else: fig,ax = plt.subplots(3,1)

    ax[0].plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
    ax[0].set_xlabel("t [ms]")
    ax[0].set_ylabel("v [mV]")
    ax[0].grid()

    ax[1].plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, lw=2)
    ax[1].axis((
        0,
        np.max(state_monitor.t / b2.ms),
        min(state_monitor.I_e[0] / b2.uamp) * 1.1,
        max(state_monitor.I_e[0] / b2.uamp) * 1.1
    ))
    ax[1].set_xlabel("t [ms]")
    ax[1].set_ylabel("$I_{e}$ [$\mu$ A]")
    ax[1].grid()

    ax[2].plot(state_monitor.t / b2.ms, state_monitor.I_na[0] / b2.uamp, lw=2, label='$I_{Na}$')
    ax[2].plot(state_monitor.t / b2.ms, state_monitor.I_k[0] / b2.uamp, lw=2,label='$I_{K}$')
    #ax[2].axis((
        #0,
        #np.max(state_monitor.t / b2.ms),
        #min(state_monitor.I_na[0] / b2.uamp) * 1.1,
        #max(state_monitor.I_k[0] / b2.uamp) * 1.1
    #))
    ax[2].set_xlabel("t [ms]")
    ax[2].set_ylabel("$I_{Na}$, $I_k$ [mA]")
    ax[2].legend(loc='upper right')
    ax[2].grid()

    if type == 'adaptative' :
        ax[3].plot(state_monitor.t / b2.ms, state_monitor.I_m[0] / b2.uamp, lw=2)
        ax[3].set_ylabel("$I_m$ [mA]")
        ax[3].axis((
        0,
        np.max(state_monitor.t / b2.ms),
        min(state_monitor.I_m[0] / b2.uamp) * 1.1,
        max(state_monitor.I_m[0] / b2.uamp) * 1.1
        ))
        ax[3].grid()

    if title is not None:
        plt.suptitle(title)
    #plt.show()
    
    # Plot of the channel variables:

    if type == 'adaptative' :
        fig,ax = plt.subplots(4,1)
    else: fig,ax = plt.subplots(3,1)

    ax[0].plot(state_monitor.t / b2.ms, state_monitor.m[0] / b2.volt, "black", lw=2)
    ax[0].set_xlabel("t (ms)")
    ax[0].set_ylabel("m")
    ax[0].set_ylim((-0.2, 1.2))
    ax[0].grid()

    ax[1].plot(state_monitor.t / b2.ms, state_monitor.n[0] / b2.volt, "blue", lw=2)
    ax[1].set_xlabel("t (ms)")
    ax[1].set_ylabel("n")
    ax[1].set_ylim((-0.2, 1.2))
    ax[1].grid()    

    ax[2].plot(state_monitor.t / b2.ms, state_monitor.h[0] / b2.volt, "red", lw=2)
    ax[2].set_xlabel("t (ms)")
    ax[2].set_ylabel("h")
    ax[2].set_ylim((-0.2, 1.2))
    ax[2].grid()

    if type == 'adaptative' :
        ax[3].plot(state_monitor.t / b2.ms, state_monitor.p[0] / b2.volt, "green", lw=2)
        ax[3].set_xlabel("t (ms)")
        ax[3].set_ylabel("p")
        ax[3].set_ylim((-0.2, 1.2))
        ax[3].grid()

    if title is not None:
        plt.suptitle(title)
    plt.show()


def plot_Vm_I2(state_monitor, title=None):
    """Plots the state_monitor variables ["vm", "I_e"] vs. time.

    Args:
        state_monitor (StateMonitor): the data to plot
        title (string, optional): plot title to display
    """

    fig,ax = plt.subplots(2,1)

    ax[0].plot(state_monitor.t / b2.ms, state_monitor.vm[0] / b2.mV, lw=2)
    ax[0].set_xlabel("t [ms]")
    ax[0].set_ylabel(r"$V_m$ [mV]")
    ax[0].grid()

    ax[1].plot(state_monitor.t / b2.ms, state_monitor.I_e[0] / b2.uamp, "red", lw=2)
    ax[1].set_xlabel("t [ms]")
    ax[1].set_ylabel("$I_{e}$ [$\mu$ A]")
    ax[1].grid()

    if title is not None:
        plt.suptitle(title)
    plt.show()


########## DEFINITION OF THE HH NEURON MODELS ##############

def simulate_HH_neuron_regular2(input_current, simulation_time):

    """A Hodgkin-Huxley neuron implemented in Brian2.

    Args:
        input_current (TimedArray): Input current injected into the HH neuron
        simulation_time (float): Simulation time [seconds]

    Returns:
        StateMonitor: Brian2 StateMonitor with recorded fields
        ["vm", "I_e", "I_na", "I_k", "m", "n", "h","minf","ninf","hinf","tm","tn","th"]
    """

    # neuron parameters
    El = 10.6 * b2.mV
    EK = -12 * b2.mV
    ENa = 115 * b2.mV
    gl = 0.3 * b2.msiemens
    gK = 36 * b2.msiemens
    gNa = 120 * b2.msiemens
    C = 1 * b2.ufarad

    # forming HH model with differential equations
  
    eqs = """
    I_e = input_current(t,i) : amp
    membrane_Im = I_e + -gl*(vm-El) + I_na + I_k : amp
    alphah = .07*exp(-.05*vm/mV)/ms    : Hz
    alpham = .1*(25*mV-vm)/(exp(2.5-.1*vm/mV)-1)/mV/ms : Hz
    alphan = .01*(10*mV-vm)/(exp(1-.1*vm/mV)-1)/mV/ms : Hz
    betah = 1./(1+exp(3.-.1*vm/mV))/ms : Hz
    betam = 4*exp(-.0556*vm/mV)/ms : Hz
    betan = .125*exp(-.0125*vm/mV)/ms : Hz
    dh/dt = alphah*(1-h)-betah*h : 1
    dm/dt = alpham*(1-m)-betam*m : 1
    dn/dt = alphan*(1-n)-betan*n : 1
    dvm/dt = membrane_Im/C : volt
    minf = alpham/(alpham+betam) : 1
    ninf = alphan/(alphan+betan) : 1
    hinf = alphah/(alphah+betah) : 1
    tm = 1/(alpham + betam) : second
    tn = 1/(alphan + betan) : second
    th = 1/(alphah + betah) : second
    I_na = -gNa*(m**3)*h*(vm-ENa) : amp
    I_k = -gK*(n**4)*(vm-EK) : amp
    """
    neuron = b2.NeuronGroup(1, eqs, method="exponential_euler")

    # parameter initialization
    neuron.vm = 0
    neuron.m = 0.05
    neuron.h = 0.60
    neuron.n = 0.32

    #neuron.vm = -70* b2.mV
    #neuron.m = 0.0
    #neuron.h = 1.0
    #neuron.n = 0.0

    # tracking parameters
    st_mon = b2.StateMonitor(neuron, ["vm", "I_e", "I_na", "I_k", "m", "n", "h","minf","ninf","hinf","tm","tn","th"], record=True)

    # running the simulation
    hh_net = b2.Network(neuron)
    hh_net.add(st_mon)
    hh_net.run(simulation_time)

    return st_mon


def getting_started2():
    """
    An example to quickly get started with the Hodgkin-Huxley module.
    """
    current_r =  input_factory.get_step_current(10, 100, b2.ms, 2.0 * b2.uA)


    state_monitor_regular = simulate_HH_neuron_regular2(current_r, 100 * b2.ms)
    plot_data2(state_monitor_regular, type='regular', title="HH Neuron, step current, regular")

    current_a =  input_factory.get_step_current(10, 100, b2.ms, 2.0 * b2.uA)

    
def find_stable_pt2():

    current = input_factory.get_zero_current()
    state_monitor_regular = simulate_HH_neuron_regular2(current, 70 * b2.ms)
    plot_data2(state_monitor_regular, type='regular', title="HH Neuron, step current, regular")

    print("The variable stable points for REGULAR neuron are: \n vm = -70 mV \n m = 0.0 \n h = 1.0 \
    \n n = 0.0 ")


if __name__ == "__main__":
    getting_started2()
    #find_stable_pt2()
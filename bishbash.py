# Gerador dos gráficos do Bish Bash

import numpy as np
from scipy.io.wavfile import write

def write_signal():
    t = np.arange(0, 2, 1/44100)
    sinusoids = np.cos(2*np.pi*(1000*t)) + np.cos(2*np.pi + 1.414*1000*t)

    pulses = np.concatenate((np.zeros(2*14050), np.ones(2*2000), np.zeros(2*12000), np.ones(2*2000), np.zeros(2*14050)))

    signal = sinusoids + pulses

    write("audio/bishbash.wav", 44100, signal.astype(np.float32))




if __name__ == "__main__":
    write_signal()

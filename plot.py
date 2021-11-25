import librosa.display
import matplotlib.pyplot as plt

class AudioAnalysis:

    def plot_stfts(self):
        fig, ax = plt.subplots(2, 2) #TODO ajeitar essa opção para não necessariamente 4 resoluções.
        img = librosa.display.specshow(librosa.amplitude_to_db(STFT_sp2_4096, ref=np.max), y_axis='log', x_axis='time',
                                       ax=ax[0, 0])
        ax[0, 0].set_title('Spectrogram for Spanish2.wav with window length of 4096')
        fig.colorbar(img, ax=ax[0, 0], format="%+2.0f dB")

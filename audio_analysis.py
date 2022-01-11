import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle
from definitions import combination_methods
from timeit import default_timer

class AudioAnalysis():
    def __init__(self, audio_file_path, t_inicio, t_fim, resolutions, count_time):
        self.audio = Audio(audio_file_path, t_inicio, t_fim)
        self.resolutions = resolutions
        self.count_time = count_time
        self.tfrs_tensor = None
        self.combined_tfr = None
        self.method = None

        #self.__set_stft_params()
        self.hop_length = 256
        self.n_fft = 1024
        if self.count_time:
            time_i = default_timer()
        self.__calculate_tfrs()
        if self.count_time:
            time_f = default_timer()
            print(f"Spectrograms calculation execution time: {time_f - time_i:.3f}s")

    @classmethod
    def from_file(cls, file):
        """
        Instancia um objeto AudioAnalysis a partir da leitura de um arquivo binário, gravado previamente com o método save_to_file.
        Utiliza a biblioteca pickle.
        :param file: Nome do arquivo com o objeto gravado.
        :return: None.
        """
        with open(file, "rb") as input_file:
            audio_analysis = pickle.load(input_file)
            return audio_analysis

    def calculate_tfr_combination(self, method, **kwargs): #TODO Futuramente permitir guarda múltiplos tipos de combinações diferentes.
        """
        Calcula e salva a combinação de RTFs usando o método de combinação fornecida, no atributo "combined_tfr".
        Também salva o nome do método usado no atributo "method", para formatação do plot.

        :param method: String que contém entrada para algum método em "combination_methods", no arquivo "definitions.py".
        :param kwargs: Argumentos chave-valor a serem passados para a função de combinação. TODO Receber os argumentos.
        :return: None.
        """
        if self.count_time:
            time_i = default_timer()
        self.combined_tfr = combination_methods[method]["function"](self.tfrs_tensor, **kwargs)
        if self.count_time:
            time_f = default_timer()
            print(f"Combination execution time: {time_f - time_i:.3f}s")
        self.method = combination_methods[method]["name"]

    def save_to_file(self, file_path):
        """
        Salva o objeto AudioAnalysis em um arquivo, possibilitando que ele seja recuperado depois sem a necessidade de refazer cálculos.
        :param file_path: Caminho do arquivo no qual o objeto será salvo.
        :return: None.
        """
        with open(file_path, "wb") as output_file:
            pickle.dump(self, output_file)


    # Private methods.
    def __set_stft_params(self):
        """
        Calcula o número de pontos e o hop length das STFTs, a partir das resoluções fornecidas para as janelas temporais.
        Os valores calculados são armazenados nos parâmetros correspondentes "n_fft" e "hop_length".
        O número de pontos utilizado é o dobro da maior resolução, aproximado para a próxima potência de 2.
        O hop length utilizado é 1/8 da menor solução, aproximado para a próxima potência de 2.
        :param resolutions: Lista de resoluções fornecidas.
        :return: None.
        """
        # TODO Usar interpolação ou permitir maior flexibilidade na escolha dos parâmetros. Outros tipos de transformadas também.
        max_resolution = max(self.resolutions)
        min_resolution = min(self.resolutions)
        iter = 1

        while iter < min_resolution // 8:
            iter *= 2

        self.hop_length = iter
        
        while iter < max_resolution * 2:
            iter *= 2
        
        self.n_fft = iter

        
    def __calculate_tfrs(self):
        self.tfrs_tensor = np.array([np.square(np.abs(librosa.stft(self.audio.audio_data, n_fft=self.n_fft,
                                                    hop_length=self.hop_length, win_length=resolution
                                                    ))) for resolution in self.resolutions]).astype(np.double)

        #print(f"{self.resolutions=}")
        #print(f"{self.n_fft=}, {self.hop_length=}")
        print(f"Shape={self.tfrs_tensor.shape}")

    def plot(self): #TODO Permitir mais customização na chamada dessa função, unindo ao plot2.
        assert self.tfrs_tensor is not None
        assert self.combined_tfr is not None # TODO transformar asserts em erros.
        assert self.method is not None

        num_plots = len(self.resolutions)
        lines, cols, ax_i = AudioAnalysis.__get_iterable_axis_indices(num_plots)
        fig, ax = plt.subplots(lines, cols)
        for i in range(num_plots):
            img = librosa.display.specshow(librosa.amplitude_to_db(self.tfrs_tensor[i], ref=np.max),
                                           y_axis='log', x_axis='time',
                                           hop_length=self.hop_length, sr=self.audio.sample_rate,
                                           ax=ax[ax_i[i][0], ax_i[i][1]])
            ax[ax_i[i][0], ax_i[i][1]].set_title("Spectrogram for {} with window length of {}".format(self.audio.audio_file_path, self.resolutions[i]))
            fig.colorbar(img, ax=ax[ax_i[i][0], ax_i[i][1]], format="%+2.0f dB")

        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(self.combined_tfr, ref=np.max),
                                       y_axis='log', x_axis='time',
                                       hop_length=self.hop_length, sr=self.audio.sample_rate,
                                       ax=ax2)
        ax2.set_title("Combination of spectrograms using {}".format(self.method))
        fig.colorbar(img, ax=ax2, format="%+2.0f dB")

        plt.show()

    def plot2(self):
        assert self.tfrs_tensor is not None
        assert self.combined_tfr is not None
        assert self.method is not None

        num_figures = len(self.resolutions)
        handlers = []
        for i in range(num_figures):
            handlers.append(plt.subplots())
            img = librosa.display.specshow(librosa.amplitude_to_db(self.tfrs_tensor[i], ref=np.max),
                                           y_axis='log', x_axis='time',
                                           hop_length=self.hop_length, sr=self.audio.sample_rate,
                                           ax=handlers[i][1])
            handlers[i][1].set_title(
                "Spectrogram for {} with window length of {}".format(self.audio.audio_file_path, self.resolutions[i]))
            handlers[i][0].colorbar(img, ax=handlers[i][1], format="%+2.0f dB")
        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(librosa.amplitude_to_db(self.combined_tfr, ref=np.max),
                                       y_axis='log', x_axis='time',
                                       hop_length=self.hop_length, sr=self.audio.sample_rate,
                                       ax=ax2)
        ax2.set_title("Combination of spectrograms using {}".format(self.method))
        fig2.colorbar(img, ax=ax2, format="%+2.0f dB")

        plt.show()



    @classmethod
    def __get_iterable_axis_indices(cls, res_len):
        lines, cols = 2, 2 # TODO mudar isso depois usando res_len
        return lines, cols, list(map(lambda i: (i//cols, i%cols), range(lines*cols))) # TODO Mudar isso, esse list de map não tá legal.


class Audio:
    def __init__(self, audio_file_path, t_inicio, t_fim):
        self.audio_file_path = audio_file_path
        self.t_inicio = t_inicio
        self.t_fim = t_fim
        self.audio_data, self.sample_rate = self.__load_audio(audio_file_path, t_inicio, t_fim)

    def __load_audio(self, audio_file_path, t_inicio, t_fim):
        if t_inicio and t_fim:
            assert t_fim > t_inicio  #TODO transformar isso em um erro, provavelmente já no init.
            return librosa.load(audio_file_path, sr=None, offset=t_inicio, duration=t_fim-t_inicio)

        if t_inicio and not t_fim:
            return librosa.load(audio_file_path, sr=None, offset=t_inicio)

        if not t_inicio and t_fim:
            return librosa.load(audio_file_path, sr=None, duration=t_fim)

        return librosa.load(audio_file_path, sr=None)


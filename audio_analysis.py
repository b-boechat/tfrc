import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import pickle
from definitions import combination_methods
from timeit import default_timer

class AudioAnalysis():
    """ Classe que instancia uma análise de sinal de áudio por combinações de representações tempo-frequenciais, armazenando as informações pertinentes.

        Atributos:
            tfrs_tensor (ndarray): Array 3D contendo o tensor de RTFs computados para o áudio. Tem dimensões de Representações x Frequência x Tempo 
            combined_tfr (ndarray): Caso self.calculate_tfr_combination() tenha sido chamada, contém o Array 2D com a RTF combinada para o áudio, nas dimensões Frequência x Tempo. Caso contrário, contém None.
            method (String): Caso self.calculate_tfr_combination() tenha sido chamada, contém a String que representa o método usado na combinação, de acordo com o dicionário combination_methods. Caso contrário, contém None.
            hop_size (Number): 
            n_fft (Number):
            audio (AudioAnalysis.Audio): Objeto contendo as informações pertinentes ao sinal de áudio no tempo.
                audio.data (ndarray): Vetor numérico 1D representando o áudio em forma .wav
                audio.sample_rate (Number): 
                audio.energy (Number): Energia do áudio, calculada como np.linalg.norm(self.audio_data)
                audio.file_path (String): Caminho do áudio analisado, conforme fornecido na inicialização do objeto.
                audio.t_inicio (Number): Tempo inicial analisado do áudio. Se "None", equivale ao início do áudio.
                audio.t_fim (Number): Tempo final analisado do áudio. Se "None", equivale ao fim do áudio.
            resolutions (List): Lista contendo as resoluções usadas no cálculo das RTFs.
            count_time (Boolean): Se verdadeiro, o tempo de cálculo das combinações, caso self.calculate_tfr_combinations() seja chamada, é calculado e mostrado.

        Métodos:
            AudioAnalysis(audio_file_path, t_inicio, t_fim, resolutions, count_time): Instancia um objeto AudioAnalysis a partir de informações sobre o áudio e sobre as representações tempo-frequenciais (RTF).
            AudioAnalysis.from_file(file): Instancia um objeto AudioAnalysis a partir de uma análise previamente salva pela função self.save_to_file(file).
            self.calculate_tfr_combinations(method, **kwags): Calcula e salva a combinação de RTFs usando o método de combinação fornecida e as RTFs já calculadas.
            self.save_to_file(file_path): Salva o objeto atual AudioAnalysis no arquivo especificado. 
            self.plot(): Plota as RTFs e a RTF combinada, em uma única figure. Funciona melhor para len(self.resolutions) = 3. Não pode ser chamada antes de self.calculate_tfr_combinations().
            self.plot2(): Plota as RTFs e a RTF combinada, em figures separadas. Não pode ser chamada antes de self.calculate_tfr_combinations().
    """

    def __init__(self, audio_file_path, t_inicio=None, t_fim=None, resolutions=[1024, 2048, 4096], count_time=False):
        """
        Instancia um objeto AudioAnalysis a partir de informações sobre o áudio e sobre as representações tempo-frequenciais (RTF).

        :param audio_file_path (String): Caminho (absoluto ou relativo) para o áudio no formato .wav
        :param t_inicio (Number): Tempo inicial a ser analisado do áudio. Se especificado "None", equivale ao início do áudio. Default: "None" 
        :param t_fim (Number): Tempo final a ser analisado do áudio. Se especificado "None", equivale ao fim do áudio. Default: "None" 
        :param resolutions (List): Lista contendo as resoluções (largura da janela) a serem calculadas de DFT. São usadas também para calcular n_fft e hop_length. Futuramente, será adicionado suporte para outros tipos de transformadas. Default: [512, 1024, 2048]
        :param count_time (Boolean): Se verdadeiro, o tempo de cálculo das RTFs (e das combinações, caso self.calculate_tfr_combinations() seja chamada posteriormente) é calculado e mostrado. Default: False
        """
        self.audio = Audio(audio_file_path, t_inicio, t_fim)
        self.resolutions = resolutions
        self.resolutions.sort()
        self.count_time = count_time
        self.tfrs_tensor = None
        self.combined_tfr = None
        self.method = None

        self.__set_stft_params()
        #self.hop_length = 512
        self.n_fft = self.resolutions[-1] # Para a síntese não queremos n_fft > resolutions.
        if self.count_time:
            time_i = default_timer()
        self.__calculate_tfrs()
        if self.count_time:
            time_f = default_timer()
            print(f"Spectrograms calculation execution time: {time_f - time_i:.3f}s")

    @classmethod
    def from_file(cls, file):
        """
        Instancia um objeto AudioAnalysis a partir de uma análise previamente salva pela função self.save_to_file(file).
        Utiliza a biblioteca pickle.
        :param file: Nome do arquivo com o objeto gravado.
        :return: None.
        """
        with open(file, "rb") as input_file:
            audio_analysis = pickle.load(input_file)
            return audio_analysis

    def calculate_tfr_combination(self, method, **kwargs): #TODO Futuramente permitir guarda múltiplos tipos de combinações diferentes.
        """
        Calcula e salva a combinação de RTFs usando o método de combinação fornecida e as RTFs já calculadas.
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

        #print(f"Soma = {np.sum(self.combined_tfr, axis=None)}")
        #print(f"Fator = {self.audio.energy/np.sum(self.combined_tfr, axis=None)}")

        #self.combined_tfr *= self.audio.energy/np.sum(self.combined_tfr, axis=None)
        self.method = combination_methods[method]["name"]

    def save_to_file(self, file_path, confirmation=True):
        """
        Salva o objeto atual AudioAnalysis no arquivo especificado.
        :param file_path (String): Caminho do arquivo no qual o objeto será salvo.
        :param confirmation (Boolean): Se verdadeiro, exibe uma mensagem de confirmação após escrever o arquivo.
        :return: None.
        """
        with open(file_path, "wb") as output_file:
            pickle.dump(self, output_file)
            print(f"Output written in {file_path}")


    # Private methods.
    def __set_stft_params(self):
        """
        Calcula o número de pontos e o hop length das STFTs, a partir das resoluções fornecidas para as janelas temporais.
        Os valores calculados são armazenados nos parâmetros correspondentes "n_fft" e "hop_length".
        O número de pontos utilizado é o dobro da maior resolução, aproximado para a próxima potência de 2.
        O hop length utilizado é 1/2 da menor solução, aproximado para a próxima potência de 2.
        :param resolutions: Lista de resoluções fornecidas.
        :return: None.
        """
        # TODO Usar interpolação ou permitir maior flexibilidade na escolha dos parâmetros. Outros tipos de transformadas também.

        max_resolution = max(self.resolutions)
        min_resolution = min(self.resolutions)
        iter = 1

        while iter < min_resolution // 2: # TODO Na tese é min_resolution // 8
            iter *= 2

        self.hop_length = iter
        
        while iter < max_resolution * 2:
            iter *= 2
        
        self.n_fft = iter

        
    def __calculate_tfrs(self):

        self.tfrs_tensor = np.array([librosa.stft(self.audio.data, n_fft=self.n_fft,
                                                    hop_length=self.hop_length, win_length=resolution,
                                                    window='hamming', center=False
                                                    ) for resolution in self.resolutions])

        #self.tfrs_tensor *= self.audio.energy / np.linalg.norm(self.tfrs_tensor, axis=(1, 2), keepdims=True)

        self.tfrs_tensor = np.square(np.abs(self.tfrs_tensor)).astype(np.double)

        print(f"tfrs tensor shape={self.tfrs_tensor.shape}")

    def plot(self): #TODO Ajeitar essa função
        assert self.tfrs_tensor is not None
        assert self.combined_tfr is not None # TODO transformar asserts em erros.
        assert self.method is not None

        num_plots = len(self.resolutions) + 1
        lines, cols, ax_i = AudioAnalysis.__get_iterable_axis_indices(num_plots)
        fig, ax = plt.subplots(lines, cols)
        for i in range(num_plots - 1):
            img = librosa.display.specshow(librosa.amplitude_to_db(self.tfrs_tensor[i], ref=np.max),
                                           y_axis='log', x_axis='time',
                                           hop_length=self.hop_length, sr=self.audio.sample_rate,
                                           ax=ax[ax_i[i][0], ax_i[i][1]])
            ax[ax_i[i][0], ax_i[i][1]].set_title("Spectrogram for {} with window length of {}".format(self.audio.file_path, self.resolutions[i]))
            fig.colorbar(img, ax=ax[ax_i[i][0], ax_i[i][1]], format="%+2.0f dB")

        i = num_plots - 1

        img = librosa.display.specshow(librosa.amplitude_to_db(self.combined_tfr, ref=np.max),
                                       y_axis='log', x_axis='time',
                                       hop_length=self.hop_length, sr=self.audio.sample_rate,
                                       ax=ax[ax_i[i][0], ax_i[i][1]])
        ax[ax_i[i][0], ax_i[i][1]].set_title("Combination of spectrograms using {}".format(self.method))
        fig.colorbar(img, ax=ax[ax_i[i][0], ax_i[i][1]], format="%+2.0f dB")

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
        self.file_path = audio_file_path
        self.t_inicio = t_inicio
        self.t_fim = t_fim
        self.data, self.sample_rate = self.__load_audio(audio_file_path, t_inicio, t_fim)
        self.energy = np.linalg.norm(self.data)

    def __load_audio(self, audio_file_path, t_inicio, t_fim):
        if t_inicio and t_fim:
            assert t_fim > t_inicio  #TODO transformar isso em um erro, provavelmente já no init.
            return librosa.load(audio_file_path, sr=48000, offset=t_inicio, duration=t_fim-t_inicio)

        if t_inicio and not t_fim:
            return librosa.load(audio_file_path, sr=48000, offset=t_inicio)

        if not t_inicio and t_fim:
            return librosa.load(audio_file_path, sr=48000, duration=t_fim)

        return librosa.load(audio_file_path, sr=48000)


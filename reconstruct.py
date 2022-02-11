import sys
import numpy as np
import librosa
from phase import RTISI_LA
from audio_analysis import AudioAnalysis
from definitions import backup_folder, backup_files_extension

def reconstruct_RTISI_LA(input_file, output_file):
    analysis = AudioAnalysis.from_file(f"{backup_folder}/{input_file}{backup_files_extension}")
    
    Y_mag = np.sqrt(analysis.combined_tfr)
    #Y_mag = np.sqrt(analysis.tfrs_tensor[-1])
    sr = analysis.audio.sample_rate
    fft_size = analysis.n_fft
    #w_size = max(analysis.resolutions)
    w_size = analysis.resolutions[-1] # Espera-se que seja igual ao n_fft
    hop_size = analysis.hop_length
    LA = (fft_size // hop_size) - 1
    threshold = 0.01

    print(f"{sr=}, {fft_size=}, {w_size=}, {hop_size=}, {LA=}, {threshold=}")
    print(f"Y_mag.shape = {Y_mag.shape}")

    print("Y_mag")
    print(Y_mag)

    RTISI_LA(Y_mag, sr, fft_size, w_size, hop_size, f"reconstructed/RTISI_LA_{output_file}.wav", LA, threshold)

def main():
    if len(sys.argv) < 2:
        input_file = "backup"
    else:
        input_file = sys.argv[1]
    
    if len(sys.argv) < 3:
        output_file = input_file
    else:
        output_file = sys.argv[2]

    reconstruct_RTISI_LA(input_file, output_file)

def from_audio_file(file_name):
    input_audio_path = f"audio/{file_name}.wav"
    sr = 48000
    n_fft = 4096
    hop_length = 512

    y_orig, _ = librosa.load(input_audio_path, sr=sr)
    print(y_orig)
    Y_mag = np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hamming', center=False))

    RTISI_LA(Y_mag, sr, 4096, 4096, 512, f"reconstructed/RTISI_LA_{file_name}_stft{n_fft}.wav")



if __name__ == '__main__':
    main()
    #from_audio_file("maja")
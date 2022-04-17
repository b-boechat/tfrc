import sys
import numpy as np
import librosa
from phase import RTISI_LA
from audio_analysis import AudioAnalysis
from definitions import backup_folder, backup_files_extension
from pydub import AudioSegment



def normalize_to_target_dbfs(file_full_path, target_dBFS):
    sound = AudioSegment.from_file(file_full_path, "wav")
    change_in_dBFS = target_dBFS - sound.dBFS
    normalized = sound.apply_gain(change_in_dBFS)
    normalized.export(f"{file_full_path}", format="wav")

def get_dbfs(file_full_path):
    sound = AudioSegment.from_file(file_full_path, "wav")
    return sound.dBFS

def reconstruct_RTISI_LA(input_file, output_file):
    print(f"Reconstruindo do espectrograma combinado em {input_file}.bkp.")

    output_file_path = f"reconstructed/RTISI_LA_{output_file}.wav"

    analysis = AudioAnalysis.from_file(f"{backup_folder}/{input_file}{backup_files_extension}")
    
    Y_mag = np.sqrt(analysis.combined_tfr/(16*np.sum(analysis.combined_tfr, axis=None)))

    #print(Y_mag)

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

    #print("Y_mag")
    #print(Y_mag)

    RTISI_LA(Y_mag, sr, fft_size, w_size, hop_size, output_file_path, LA, threshold)
    
    normalize_to_target_dbfs(output_file_path, analysis.audio.dbfs)







def main():
    if len(sys.argv) < 2:
        print("No input file!")
        exit()
    
    input_file = sys.argv[1]
    
    if len(sys.argv) < 3:
        output_file = input_file
    else:
        if sys.argv[2] == "from_file":
            if len(sys.argv) == 4:
                n_fft = int(sys.argv[3])
            else:
                n_fft = 2048
            from_audio_file(input_file, n_fft)
            exit()
        output_file = sys.argv[2]

    reconstruct_RTISI_LA(input_file, output_file)

def from_audio_file(file_name, n_fft):
    print(f"Reconstruindo do espectrograma simples de {n_fft} pontos construído para arquivo {file_name}.wav.")
    input_audio_path = f"audio/{file_name}.wav"

    sr = 48000
    hop_length = n_fft//8

    y_orig, _ = librosa.load(input_audio_path, sr=sr)
    
    Y_mag = np.abs(librosa.stft(y_orig, n_fft=n_fft, hop_length=hop_length, win_length=n_fft, window='hamming', center=False))

    output_audio_path = f"reconstructed/RTISI_LA_{file_name}_stft{n_fft}.wav"

    RTISI_LA(Y_mag, sr, n_fft, n_fft, hop_length, output_audio_path)

    original_dbfs = get_dbfs(input_audio_path)
    normalize_to_target_dbfs(output_audio_path, original_dbfs)


def generate_command(file_names):
    methods = [#("mean", "mean"),
            #("median", "median"),
            ("lsm", "sls"),
            ("lsm -p zeta=-1", "nsls"),
            ("fls_matlab", "fls"),
            ("lt_matlab", "lt")]

    resolutions = [1024, 2048, 4096]

    commands = []

    for name in file_names:
        for r in resolutions:
            commands.append(f"python reconstruct.py {name} from_file {r}")
        for m in methods:
            commands.append(f"python tfrc.py g {name}.wav -t -o {name}_{m[1]} -m {m[0]}")
            commands.append(f"python reconstruct.py {name}_{m[1]}")

    command = " & ".join(commands)

    print(command)

if __name__ == '__main__':
    main()
    #from_audio_file("lagrima", 4096)
    #generate_command(["because", "iwantyou", "lagrima", "lotseven", "maja", "x1"])
    #generate_command(["goat"])





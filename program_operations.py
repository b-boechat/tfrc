from email.mime import audio
from audio_analysis import AudioAnalysis
from debug import print_arr

import numpy as np

def generate_tfrc(audio_file_path, 
                  sample_rate, t_inicio, t_fim, 
                  tfr_type, resolutions,
                  output_file_path,
                  combination_method,
                  count_time,
                  plot,
                  combination_params):

    audio_analysis = AudioAnalysis(audio_file_path=audio_file_path, 
                                   sample_rate=sample_rate, t_inicio=t_inicio, t_fim=t_fim, 
                                   tfr_type=tfr_type, resolutions=resolutions, count_time=count_time)

    audio_analysis.calculate_tfr_combination(combination_method, **combination_params)
    audio_analysis.save_to_file(output_file_path)

    print_arr(audio_analysis.combined_tfr[48:65, 0:17], round_digs=8)

    if plot:
        audio_analysis.plot()


def restore_tfrc(input_file_path):
    audio_analysis = AudioAnalysis.from_file(input_file_path)

    audio_analysis.plot()

    #print(np.max(audio_analysis.combined_tfr))    
    #print_arr(audio_analysis.combined_tfr[48:65, 0:17])
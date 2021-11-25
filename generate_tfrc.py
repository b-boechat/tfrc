from audio_analysis import AudioAnalysis

def generate_tfrc(audio_file, t_inicio, t_fim, resolutions,
                  input_file, output_file,
                  combination_method,
                  count_time,
                  combination_params):
    if input_file is not None:
        audio_analysis = AudioAnalysis.from_file(input_file)
    else:
        audio_analysis = AudioAnalysis(audio_file, t_inicio, t_fim, resolutions, count_time)
        audio_analysis.calculate_tfr_combination(combination_method, **combination_params)
        audio_analysis.save_to_file(output_file)

    audio_analysis.plot()
    #pyaudio_analysis.plot2()

import argparse
import re
#import locale
from program_operations import generate_tfrc, restore_tfrc
from definitions import combination_methods, backup_files_extension, backup_folder, audio_folder

def generate_tfrc_wrapper(args):
    """ Chama a função de combinação de rtfcs com os argumentos passados.

    :param args: Argumentos parseados pela linha de comando.
    """

    audio_file_path = f"{audio_folder}/{args.audio_file}"
    output_file_path = f"{backup_folder}/{args.output_file}{backup_files_extension}"

    if args.combination_params is not None:
        combination_params_str = " ".join(args.combination_params)
        combination_params = parse_params(combination_params_str)
    else:
        combination_params = dict()

    if args.resolutions is None:  
        if args.tfr_type == "stft":
            args.resolutions = [1024, 2048, 4096]
        else: #cqt
            args.resolutions = [12, 24, 36]


    print(args)

    generate_tfrc(audio_file_path=audio_file_path, 
                  sample_rate=args.sample_rate, t_inicio=args.crop_time[0], t_fim=args.crop_time[1], 
                  tfr_type=args.tfr_type, 
                  resolutions=args.resolutions,
                  output_file_path=output_file_path,
                  combination_method=args.combination_method,
                  count_time=args.count_time,
                  plot=args.plot,
                  combination_params=combination_params)

    


def restore_tfrc_wrapper(args):
    """ Chama a função de resturar rtfcs de backup com os argumentos passados.

    :param args: Argumentos parseados pela linha de comando.
    """

    input_file_path = f"{backup_folder}/{args.input_file}{backup_files_extension}"

    restore_tfrc(input_file_path=input_file_path)


def list_methods(_):
    for identifier, value in sorted(combination_methods.items()):
        print(f"{identifier:25}{value['name']}")

def parse_params(params_str):

    #print ("str =", params_str)

    def parse_with_type(value):
        try:
            num_value = int(value)
            return num_value
        except ValueError:
            try:
                num_value = float(value)
                return num_value
            except ValueError:
                return value

    params = dict()
    pattern = re.compile(r"(\w+)=([^\s]+)")

    while True:
        match = pattern.match(params_str)
        if match is None:
            break
        params_str = pattern.sub("", params_str, count=1).lstrip()
        params[match.group(1)] = parse_with_type(match.group(2))
        
    return params


def main():

    # Ajusta o locale para execução, para em usar vírgula em vez de ponto para decimal nos plots. TODO No momento, não funciona em conjunto com o usetex.
    #locale.setlocale(locale.LC_NUMERIC, 'de_DE.UTF-8')

    parser = argparse.ArgumentParser(description="Interface para o cálculo, armazenamento e visualização de combinações de representações tempo-frequenciais.")

    subparsers = parser.add_subparsers()

    # Cria o subparser para a função "generate" (chama generate_tfrc)
    parser_generate = subparsers.add_parser("generate", aliases=["g"], help="Gera uma combinação de RTFs a partir do arquivo de áudio fornecido.")

    parser_generate.add_argument("audio_file", metavar="AUDIO_FILE",
                              help=f"""Arquivo de áudio a ser analisado.
                                  Entende-se que está na pasta {audio_folder}, definida na variável \"audio_folder\" em \"definitions.py\"""")  # TODO adicionar opção para gerar arquivos sintéticos
    parser_generate.add_argument("-s", "--sr", dest="sample_rate", type=float, default=None, 
                                 help="Taxa de amostragem em que o sinal vai ser reamostrado.")
    parser_generate.add_argument("-c", "--crop", dest="crop_time", type=int, nargs=2, metavar=("inicio", "fim"),
                        default=[None, None],
                        help="Tempo de início e fim (em segundos) demarcando o intervalo analisado.")

    parser_generate.add_argument("-y", "--type", dest="tfr_type", metavar="TFR_TYPE", default="stft", choices=["stft", "cqt"],
                        help="Tipo de representação tempo-frequencial. Pode ser \"stft\" ou \"cqt\". Default: \"stft\"")
    parser_generate.add_argument("-r", "--resolutions", dest="resolutions", type=int, nargs="+",
                        help="""Resoluções utilizadas nas RTFS. Referem-se ao espalhamento da janela temporal.
                        No caso de uma STFT, corresponde ao tamanho da janela, desconsiderando zero-padding. Default: [1024, 2048, 4096].
                        No caso de uma CQT, corresponde à resolução equivalente em número de bins por oitava. Default: [12, 24, 36].""")  # TODO Especificar número de pontos e hop length.
    parser_generate.add_argument("-o", dest="output_file", metavar="OUTPUT_FILE", default="backup",
                        help=f"""Arquivo no qual as representações vão ser salvas como backup. (sem extensão, que é 
                                 entendida como \"{backup_files_extension}\", definida na variável \"backup_files_extension\" em \"definitions.py\". 
                                 Os backups são salvos na pasta \"{backup_folder}\", definida na variável \"backup_folder\" em \"definitions.py\"""")
    parser_generate.add_argument("-m", "--method", dest="combination_method", metavar="COMBINATION_METHOD", default="median",
                        choices=list(combination_methods.keys()),
                        help="Método de combinação a ser realizado.")
    parser_generate.add_argument("-t", "--time", dest="count_time", action="store_true",
                        help="Se especificado, são exibidos os tempos de cálculo dos espectrogramas e da combinação.")
    #parser_generate.add_argument("-i", "--install", dest="install", action="store_true", 
    #                    help="Provisório. Se especificado, os módulos em Cython são construídos a partir dos arquivos \".pyx\".")
    parser_generate.add_argument("-n", "--noplot", dest="plot", action="store_false",
                        help="Se especificado, espectrogramas não são apresentados em gráficos.")
    parser_generate.add_argument("-p", "--params", dest="combination_params", nargs="+", metavar="PARAMS",
                        help="Permite passar argumentos para a função de combinação, na forma <chave>=<valor>. Se especificado, precisa ser a última opção na linha de comando.")
    parser_generate.set_defaults(func=generate_tfrc_wrapper)


    # Cria o subparser para a função "restore" (chama restore_tfrc)
    parser_restore = subparsers.add_parser("restore", aliases=["r"],
                                            help="Restaura uma combinação de RTFs a partir do arquivo de backup fornecido.")

    parser_restore.add_argument("input_file", metavar="INPUT_FILE",
                              help=f"""Arquivo de backup do qual a análise vai ser recuperada (sem extensão, que é 
                              entendida como \"{backup_files_extension}\", definida na variável \"backup_files_extension\" em \"definitions.py\". 
                              Entende-se que está na pasta \"{backup_folder}\", definida na variável \"backup_folder\" em \"definitions.py\"""")
    parser_restore.set_defaults(func=restore_tfrc_wrapper)


    parser_list = subparsers.add_parser("list", aliases=["l"],
                                        help="Lista os métodos de combinação disponíveis.")
    parser_list.set_defaults(func=list_methods)



    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()

if __name__ == '__main__':
    main()
import argparse
import re
from generate_tfrc import generate_tfrc
from definitions import backup_files_extension, audio_folder, combination_methods

def generate_tfrc_wrapper(args):
    """ Chama a função de combinação de rtfcs com os argumentos passados.

    :param args: Argumentos parseados pela linha de comuando
    """
    if args.input_file is not None:
        args.input_file = f"{args.input_file}{backup_files_extension}"
    if args.output_file is not None:
        args.output_file = f"{args.input_file}{backup_files_extension}"
    if args.audio_file is not None:
        args.audio_file = f"{audio_folder}/{args.audio_file}"

    if args.combination_params is not None:
        combination_params_str = "".join(args.combination_params)
        combination_params = parse_params(combination_params_str)
    else:
        combination_params = dict()


    generate_tfrc(audio_file=args.audio_file, t_inicio=args.crop_time[0], t_fim=args.crop_time[1], resolutions=args.resolutions,
                  input_file=args.input_file, output_file=args.output_file,
                  combination_method=args.combination_method,
                  count_time=args.count_time,
                  combination_params=combination_params)


def parse_params(params_str):
    number_matches = re.findall(r"(\w+)\s*=(\w+)", params_str)
    def parse_with_type(value):
        try:
            num_value = float(value) # TODO Pode dar type error se for usado como inteiro.
            return num_value
        except ValueError:
            return value

    params = {match[0] : parse_with_type(match[1]) for match in number_matches}
    return params



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Gera combinações de RTFs.")

    input_parser = parser.add_mutually_exclusive_group(required=True)
    input_parser.add_argument("-a", dest="audio_file", metavar="AUDIO_FILE",
                              help=f"""Arquivo de áudio a ser analisado. Opção mutuamente exclusiva com -i. 
                              Entende-se que está na pasta {audio_folder}, definida na variável \"audio_folder\" em \"definitions.py\"""")  # TODO adicionar opção para gerar arquivos sintéticos
    input_parser.add_argument("-i", dest="input_file", metavar="INPUT_FILE",
                              help=f"""Arquivo de backup do qual a análise vai ser recuperada (sem extensão, que é 
                              entendida como \"{backup_files_extension}\", definida na variável \"backup_files_extension\" em \"definitions.py\". 
                              Opção mutuamente exclusiva com -a.""")

    parser.add_argument("-c", "--crop", dest="crop_time", type=int, nargs=2, metavar=("inicio", "fim"), default=[None, None],
                        help="Tempo de início e fim (em segundos) demarcando o intervalo analisado.")
    parser.add_argument("-r", "--resolutions", dest="resolutions", type=int, nargs="+", default=[512, 1024, 2048, 4096],
                        help="Resoluções utilizadas nas RTFS. Referem-se ao espalhamento da janela temporal.")  # TODO Especificar número de pontos e hop length.

    parser.add_argument("-o", dest="output_file", metavar="OUTPUT_FILE", default="backup.obj",
                        help=f"""Arquivo no qual as representações vão ser salvas como backup. (sem extensão, que é 
                              entendida como \"{backup_files_extension}\", definida na variável \"backup_files_extension\" em \"definitions.py\".""")
    parser.add_argument("-m", "--method", dest="combination_method", metavar="COMBINATION_METHOD", default="median",
                        choices=list(combination_methods.keys()),
                        help="Método de combinação a ser realizado.")
    parser.add_argument("-t", "--time", dest="count_time", action="store_true",
                        help="Se especificado, são exibidos os tempos de cálculo dos espectrogramas e da combinação.")
    parser.add_argument("-p", "--params", dest="combination_params", nargs="+",
                        help="#TODO")

    parser.set_defaults(func=generate_tfrc_wrapper)
    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()

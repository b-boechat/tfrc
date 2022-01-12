import argparse
from joblib.logger import PrintTime
import pyximport 
import re
from program_operations import generate_tfrc, restore_tfrc
from definitions import backup_files_extension, backup_folder, audio_folder

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

    #if args.pyximport_install:
    #    print("Here?")
    #    pyximport.install(language_level="3") # TODO Permitir passar argumentos para o pyximport.

    generate_tfrc(audio_file_path=audio_file_path, t_inicio=args.crop_time[0], t_fim=args.crop_time[1], resolutions=args.resolutions,
                  output_file_path=output_file_path,
                  combination_method=args.combination_method,
                  count_time=args.count_time,
                  combination_params=combination_params)


def restore_tfrc_wrapper(args):
    """ Chama a função de resturar rtfcs de backup com os argumentos passados.

    :param args: Argumentos parseados pela linha de comando.
    """

    input_file_path = f"{backup_folder}/{args.input_file}{backup_files_extension}"

    restore_tfrc(input_file_path=input_file_path)




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
        
    #for items in params.items():
    #    print (items)

    return params



if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Interface para o cálculo, armazenamento e visualização de combinações de representações tempo-frequenciais.")

    subparsers = parser.add_subparsers()

    # Cria o subparser para a função "generate" (chama generate_tfrc)
    parser_generate = subparsers.add_parser("generate", aliases=["g"], help="Gera uma combinação de RTFs a partir do arquivo de áudio fornecido.")

    parser_generate.add_argument("audio_file", metavar="AUDIO_FILE",
                              help=f"""Arquivo de áudio a ser analisado.
                                  Entende-se que está na pasta {audio_folder}, definida na variável \"audio_folder\" em \"definitions.py\"""")  # TODO adicionar opção para gerar arquivos sintéticos
    parser_generate.add_argument("-c", "--crop", dest="crop_time", type=int, nargs=2, metavar=("inicio", "fim"),
                        default=[None, None],
                        help="Tempo de início e fim (em segundos) demarcando o intervalo analisado.")
    parser_generate.add_argument("-r", "--resolutions", dest="resolutions", type=int, nargs="+", default=[512, 1024, 2048],#, 2048, 4096],
                        help="Resoluções utilizadas nas RTFS. Referem-se ao espalhamento da janela temporal.")  # TODO Especificar número de pontos e hop length.

    parser_generate.add_argument("-o", dest="output_file", metavar="OUTPUT_FILE", default="backup",
                        help=f"""Arquivo no qual as representações vão ser salvas como backup. (sem extensão, que é 
                                 entendida como \"{backup_files_extension}\", definida na variável \"backup_files_extension\" em \"definitions.py\". 
                                 Os backups são salvos na pasta \"{backup_folder}\", definida na variável \"backup_folder\" em \"definitions.py\"""")
    parser_generate.add_argument("-m", "--method", dest="combination_method", metavar="COMBINATION_METHOD", default="median",
                        choices=["median", "mean", "lsm", "lsmold"], # TODO Algum workaround pra poder usar list(combination_methods.keys()), que por enquanto não é possível por causa do pyximport.
                        help="Método de combinação a ser realizado.")
    parser_generate.add_argument("-t", "--time", dest="count_time", action="store_true",
                        help="Se especificado, são exibidos os tempos de cálculo dos espectrogramas e da combinação.")
    #parser_generate.add_argument("-i", "--install", dest="pyximport_install", action="store_true",   # Não funcionando por enquanto.
    #                    help="Se especificado, os módulos em Cython são construídos a partir dos arquivos \".pyx\".")
    parser_generate.add_argument("-p", "--params", dest="combination_params", nargs="+",
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



    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_usage()

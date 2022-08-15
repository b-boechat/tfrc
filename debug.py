import colorama

def print_arr(arr, color_range = None, color = colorama.Fore.RED, round_digs=5):
    colorama.init(autoreset=True)
    I, J = arr.shape
    for i in range(I):
        for j in range(J):
            if color_range is not None and i >= color_range[0] and i < color_range[1] and j >= color_range[2] and j < color_range[3]:
                print("{}".format(color + str(round(arr[i][j], round_digs))), end="  ")
            else:
                print("{}".format(str(round(arr[i][j], round_digs))), end="  ")
        print()

    print("\n")


def print_arr_interpol(arr, int_row = 2, int_col = 2, color = colorama.Fore.CYAN, round_digs = 3):
    colorama.init(autoreset=True)
    I, J = arr.shape
    for i in range(I):
        for j in range(J):
            if i % int_row != 0 or j % int_col != 0:
                print("{}".format(color + str(round(arr[i][j], round_digs))), end="  ")
            else:
                print("{}".format(str(round(arr[i][j], round_digs))), end="  ")
        print()

    print("\n")
 
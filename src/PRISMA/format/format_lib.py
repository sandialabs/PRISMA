def format_float(f):
    return "{:.2f}".format(f)


def format_float_l(f_l):
    return list(map(format_float, f_l))

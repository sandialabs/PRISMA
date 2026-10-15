from numbers import Real
from numpy import ndarray


def rescale_bounds(x: Real | ndarray[Real],
                   input_min: float, input_max: float,
                   output_min: float, output_max: float) \
                    -> Real | ndarray[Real]:
    """
    Convert measurements proprotionally to bounds.

    Parameters
    ----------
    x:
        Input value to be converted.
    input_min:
        Minimum of range of inputs.
    input_max:
        Maximum of range of inputs.
    output_min:
        Minimum of range of outputs.
    output_max:
        Maximum of range of outputs.
    """
    return output_min + (x - input_min) * (output_max - output_min) \
        / (input_max - input_min)


def rescale_bounds_inv(s: Real | ndarray[Real],
                       input_min: float, input_max: float,
                       output_min: float, output_max: float) \
                        -> Real | ndarray[Real]:
    return input_min + (s - output_min) * (input_max - input_min) \
        / (output_max - output_min)

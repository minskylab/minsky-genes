
def map_float(value: float,   i_start: float,   i_stop: float,   o_start: float,   o_stop: float) -> float:
    return o_start + (o_stop - o_start) * ((value - i_start) / (i_stop - i_start))

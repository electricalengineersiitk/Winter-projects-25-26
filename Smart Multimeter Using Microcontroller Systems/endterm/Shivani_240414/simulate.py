from measurement import get_resistance
from protocol import display_result

values = [100, 1000, 10000, 1000000]

for v in values:
    measured, err = get_resistance(v)
    print(display_result(v, measured, err))
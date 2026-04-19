def choose_resistor(value):
    ranges = [100, 1000, 10000, 100000]

    for r in ranges:
        if value <= 10 * r:
            return r

    return ranges[-1]
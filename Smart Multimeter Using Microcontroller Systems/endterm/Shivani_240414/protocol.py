def display_result(actual, measured, error):
    return (
        f"Input Resistance : {actual} Ω\n"
        f"Measured Value   : {round(measured, 2)} Ω\n"
        f"Percentage Error : {round(error, 4)} %\n"
        "-----------------------------"
    )
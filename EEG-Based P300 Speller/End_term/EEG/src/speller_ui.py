"""
speller_ui.py
=============
P300 Speller visual stimulus interface built with PsychoPy.

Presents a 6×6 character matrix that flashes rows and columns
in a randomised order. Records precise flash timestamps for
synchronisation with EEG acquisition hardware.

Usage
-----
    python speller_ui.py

Requirements
------------
    pip install psychopy
"""

import random
import time
import logging
from pathlib import Path

log = logging.getLogger(__name__)

# 6×6 character matrix (P300 standard layout)
CHAR_MATRIX = [
    ["A", "B", "C", "D", "E", "F"],
    ["G", "H", "I", "J", "K", "L"],
    ["M", "N", "O", "P", "Q", "R"],
    ["S", "T", "U", "V", "W", "X"],
    ["Y", "Z", "1", "2", "3", "4"],
    ["5", "6", "7", "8", "9", "_"],
]
N_ROWS = 6
N_COLS = 6
N_CHARS = N_ROWS * N_COLS   # 36

# Stimulus parameters
FLASH_DURATION_S  = 0.100   # 100 ms flash (on)
ISI_S             = 0.075   # 75 ms inter-stimulus interval (off)
N_REPETITIONS     = 10      # flash each row/col this many times per trial
HIGHLIGHT_COLOR   = (0.8, 0.8, 0.8)   # PsychoPy RGB [-1, 1] space
DEFAULT_COLOR     = (-0.5, -0.5, -0.5)
WINDOW_SIZE       = (1280, 720)


def run_speller(target_char: str = "A",
                n_repetitions: int = N_REPETITIONS,
                log_dir: str = "results") -> list[dict]:
    """
    Run one P300 speller trial for a given target character.

    Parameters
    ----------
    target_char   : the character the user intends to spell
    n_repetitions : number of row/col flash repetitions
    log_dir       : directory to write the event log CSV

    Returns
    -------
    event_log : list of dicts with keys
                {flash_type, index, timestamp_s, is_target}
    """
    try:
        from psychopy import visual, core, event
    except ImportError:
        raise ImportError(
            "PsychoPy is required to run the speller UI.\n"
            "Install it with:  pip install psychopy"
        )

    # Find target row / column
    target_char = target_char.upper()
    target_row, target_col = None, None
    for r, row in enumerate(CHAR_MATRIX):
        for c, ch in enumerate(row):
            if ch == target_char:
                target_row, target_col = r, c
                break

    if target_row is None:
        raise ValueError(f"Character '{target_char}' not found in the matrix.")

    log.info("Running P300 speller | target: '%s' (row %d, col %d)",
             target_char, target_row, target_col)

    # -----------------------------------------------------------------------
    # Build PsychoPy window and stimuli
    # -----------------------------------------------------------------------
    win = visual.Window(
        size=WINDOW_SIZE, fullscr=False,
        color=(-1, -1, -1), units="pix",
        screen=0, title="EEG Brain Speller"
    )
    clock = core.Clock()

    # Pre-compute cell positions in a grid layout
    cell_size = 80
    padding   = 15
    grid_w    = N_COLS * (cell_size + padding)
    grid_h    = N_ROWS * (cell_size + padding)
    start_x   = -grid_w / 2 + cell_size / 2
    start_y   =  grid_h / 2 - cell_size / 2

    cells = []    # (row, col) → TextStim
    for r in range(N_ROWS):
        row_cells = []
        for c in range(N_COLS):
            x = start_x + c * (cell_size + padding)
            y = start_y - r * (cell_size + padding)
            stim = visual.TextStim(
                win, text=CHAR_MATRIX[r][c],
                pos=(x, y), height=36,
                color=DEFAULT_COLOR, bold=True
            )
            row_cells.append(stim)
        cells.append(row_cells)

    # Instructions
    instr = visual.TextStim(
        win,
        text=f"Focus on the letter:  {target_char}\n\nPress SPACE to start",
        pos=(0, -grid_h / 2 - 60), height=22, color=(0.9, 0.9, 0.9)
    )

    # -----------------------------------------------------------------------
    # Instruction screen
    # -----------------------------------------------------------------------
    instr.draw()
    for r in range(N_ROWS):
        for c in range(N_COLS):
            cells[r][c].draw()
    win.flip()
    event.waitKeys(keyList=["space"])

    # -----------------------------------------------------------------------
    # Main stimulus loop
    # -----------------------------------------------------------------------
    event_log: list[dict] = []
    flash_sequence = list(range(N_ROWS + N_COLS))    # 0-5 = rows, 6-11 = cols

    for rep in range(n_repetitions):
        random.shuffle(flash_sequence)

        for flash_idx in flash_sequence:
            is_row   = flash_idx < N_ROWS
            line_num = flash_idx if is_row else (flash_idx - N_ROWS)

            # Determine if this flash contains the target
            if is_row:
                is_target = (line_num == target_row)
            else:
                is_target = (line_num == target_col)

            # ── ON period: highlight the flashed row/column ──────────────
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    if (is_row and r == line_num) or \
                       (not is_row and c == line_num):
                        cells[r][c].setColor(HIGHLIGHT_COLOR)
                    else:
                        cells[r][c].setColor(DEFAULT_COLOR)
                    cells[r][c].draw()

            win.flip()
            ts = time.time()

            event_log.append({
                "rep":         rep,
                "flash_type":  "row" if is_row else "col",
                "index":       line_num,
                "timestamp_s": ts,
                "is_target":   int(is_target),
            })

            core.wait(FLASH_DURATION_S)

            # ── OFF period: all cells back to default ────────────────────
            for r in range(N_ROWS):
                for c in range(N_COLS):
                    cells[r][c].setColor(DEFAULT_COLOR)
                    cells[r][c].draw()
            win.flip()
            core.wait(ISI_S)

            # Allow early exit
            if event.getKeys(keyList=["escape", "q"]):
                log.info("User aborted speller.")
                win.close()
                return event_log

    # -----------------------------------------------------------------------
    # End screen
    # -----------------------------------------------------------------------
    visual.TextStim(
        win, text="Trial complete. Press any key to exit.",
        pos=(0, 0), height=28, color=(0.8, 0.8, 0.8)
    ).draw()
    win.flip()
    event.waitKeys()
    win.close()

    # -----------------------------------------------------------------------
    # Save event log
    # -----------------------------------------------------------------------
    import csv
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(log_dir) / f"events_{target_char}_{int(time.time())}.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=event_log[0].keys())
        writer.writeheader()
        writer.writerows(event_log)
    log.info("Event log saved to %s  (%d events)", log_path, len(event_log))

    return event_log


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    parser = argparse.ArgumentParser(description="EEG P300 Brain Speller UI")
    parser.add_argument("--target", "-t", type=str, default="A",
                        help="Target character (default: A)")
    parser.add_argument("--reps",   "-r", type=int, default=N_REPETITIONS,
                        help="Flash repetitions per trial")
    args = parser.parse_args()

    run_speller(target_char=args.target, n_repetitions=args.reps)

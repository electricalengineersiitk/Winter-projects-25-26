"""
P300 Speller UI — PsychoPy Implementation
==========================================
Implements a proper 6×6 character matrix with randomized row/column
flash sequences, as per the P300 paradigm (Farwell & Donchin, 1988).

SOA  = 175 ms  (100 ms flash ON + 75 ms inter-stimulus interval)
Reps = 10 repetitions of all 12 rows/cols per character
"""

import random
import time

try:
    from psychopy import visual, core, event, logging
    logging.console.setLevel(logging.WARNING)
    PSYCHOPY_AVAILABLE = True
except ImportError:
    PSYCHOPY_AVAILABLE = False
    print("PsychoPy not installed. Install with: pip install psychopy")


# --- Configuration ---
CHARS        = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
GRID_ROWS    = 6
GRID_COLS    = 6
FLASH_ON_S   = 0.100   # seconds stimulus is highlighted
ISI_S        = 0.075   # inter-stimulus interval (dark period)
SOA_S        = FLASH_ON_S + ISI_S   # = 0.175s
N_REPS       = 10      # repetitions of each row/col per character
TARGET_CHAR  = 'A'     # Character the subject is told to focus on


def _build_grid(win, chars):
    """Create TextStim objects positioned in a 6×6 grid."""
    stimuli = []
    for i, ch in enumerate(chars):
        row = i // GRID_COLS
        col = i % GRID_COLS
        x = (col - (GRID_COLS - 1) / 2.0) * 2.5
        y = ((GRID_ROWS - 1) / 2.0 - row) * 2.5
        stim = visual.TextStim(
            win, text=ch, pos=(x, y),
            height=1.5, color='white', bold=True
        )
        stimuli.append(stim)
    return stimuli


def _draw_grid(win, stimuli, highlighted_indices=None, highlight_color='yellow'):
    """Draw all characters; optionally highlight a subset."""
    for i, stim in enumerate(stimuli):
        if highlighted_indices and i in highlighted_indices:
            stim.color = highlight_color
        else:
            stim.color = 'white'
        stim.draw()


def _get_flash_sequence():
    """
    Returns one full repetition order: 6 row indices + 6 column indices, shuffled.
    Rows are encoded as 0–5, columns as 6–11 (matching BNCI2014_009 convention).
    """
    order = list(range(12))   # 0–5: rows, 6–11: columns
    random.shuffle(order)
    return order


def run_speller_ui():
    if not PSYCHOPY_AVAILABLE:
        print("PsychoPy is required to run the speller UI.")
        return

    win = visual.Window(
        size=[900, 700], monitor="testMonitor",
        units="deg", fullscr=False, color='black'
    )

    stimuli = _build_grid(win, CHARS)
    target_idx = CHARS.index(TARGET_CHAR)

    # --- Build row/col membership lists ---
    # rows[r] = list of char indices in row r
    rows = [[r * GRID_COLS + c for c in range(GRID_COLS)] for r in range(GRID_ROWS)]
    # cols[c] = list of char indices in col c
    cols = [[r * GRID_COLS + c for r in range(GRID_ROWS)] for c in range(GRID_COLS)]

    print(f"[Speller UI] Target character: '{TARGET_CHAR}' | SOA={SOA_S*1000:.0f}ms | Reps={N_REPS}")
    print("Press 'q' to quit.")

    clock = core.Clock()
    scores = {i: 0.0 for i in range(12)}  # Accumulated "flash score" per row/col

    for rep in range(N_REPS):
        flash_order = _get_flash_sequence()

        for flash_id in flash_order:
            # Determine which characters to highlight
            if flash_id < 6:
                highlighted = set(rows[flash_id])
            else:
                highlighted = set(cols[flash_id - 6])

            # Flash ON
            _draw_grid(win, stimuli, highlighted_indices=highlighted)
            win.flip()
            clock.reset()
            while clock.getTime() < FLASH_ON_S:
                if 'q' in event.getKeys():
                    win.close()
                    core.quit()
                    return

            # Flash OFF (ISI)
            _draw_grid(win, stimuli)
            win.flip()
            clock.reset()
            while clock.getTime() < ISI_S:
                if 'q' in event.getKeys():
                    win.close()
                    core.quit()
                    return

            # Accumulate "score": +1 if target was in this flash group
            if target_idx in highlighted:
                scores[flash_id] += 1.0

    # --- Decode: find best row and best col ---
    best_row = max(range(6),  key=lambda r: scores[r])
    best_col = max(range(6),  key=lambda c: scores[c + 6])
    predicted_idx  = best_row * GRID_COLS + best_col
    predicted_char = CHARS[predicted_idx]

    print(f"\n[Speller UI] Predicted character: '{predicted_char}'")
    print(f"[Speller UI] Target character:    '{TARGET_CHAR}'")
    print(f"[Speller UI] Correct: {predicted_char == TARGET_CHAR}")

    win.close()
    core.quit()


if __name__ == "__main__":
    run_speller_ui()

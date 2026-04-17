from psychopy import visual, core, event
import numpy as np

GRID = np.array([
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("56789_")
])


class P300Speller:

    def __init__(self, subject_label="A"):
        self.subject_label = subject_label
        self.win = visual.Window(size=(900, 700), color="black", fullscr=False, units="norm")

        # subject label top-left
        self.subject_stim = visual.TextStim(
            self.win,
            text=f"Subject {subject_label}",
            pos=(-0.75, 0.88),
            height=0.06,
            color="grey"
        )

        # decoded output display
        self.text_stim = visual.TextStim(
            self.win,
            text="Output: ",
            pos=(0, 0.88),
            height=0.07,
            color="white"
        )

        # 6x6 character grid
        self.grid_stim = []
        for i in range(6):
            row = []
            for j in range(6):
                stim = visual.TextStim(
                    self.win,
                    text=GRID[i][j],
                    pos=(-0.5 + j * 0.2, 0.55 - i * 0.2),
                    height=0.09,
                    color="white"
                )
                row.append(stim)
            self.grid_stim.append(row)

        self.output = ""

    def draw_grid(self):
        for row in self.grid_stim:
            for stim in row:
                stim.draw()

    def highlight_row(self, r):
        for j in range(6):
            self.grid_stim[r][j].color = "yellow"

    def highlight_col(self, c):
        for i in range(6):
            self.grid_stim[i][c].color = "yellow"

    def reset_colors(self):
        for row in self.grid_stim:
            for stim in row:
                stim.color = "white"

    def update_text(self, char):
        self.output += char
        self.text_stim.text = f"Output: {self.output}"

    def draw_all(self):
        self.draw_grid()
        self.text_stim.draw()
        self.subject_stim.draw()

    def flash_sequence(self, n_reps=5, flash_on=0.1, flash_off=0.075):
        for _ in range(n_reps):
            row_order = list(range(6))
            np.random.shuffle(row_order)

            for r in row_order:
                self.highlight_row(r)
                self.draw_all()
                self.win.flip()
                core.wait(flash_on)

                self.reset_colors()
                self.draw_all()
                self.win.flip()
                core.wait(flash_off)

            col_order = list(range(6))
            np.random.shuffle(col_order)

            for c in col_order:
                self.highlight_col(c)
                self.draw_all()
                self.win.flip()
                core.wait(flash_on)

                self.reset_colors()
                self.draw_all()
                self.win.flip()
                core.wait(flash_off)

    def run(self, pred_string, n_reps=5):
        # intro screen for this subject
        intro = visual.TextStim(
            self.win,
            text=f"Subject {self.subject_label} — {len(pred_string)} characters\n\nPress SPACE to start",
            pos=(0, 0),
            height=0.08,
            color="white"
        )
        intro.draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])

        for ch in pred_string:
            keys = event.getKeys(keyList=["escape"])
            if keys:
                break

            self.flash_sequence(n_reps=n_reps)
            self.update_text(ch)

            self.reset_colors()
            self.draw_all()
            self.win.flip()
            core.wait(0.4)

        # end screen for this subject
        done = visual.TextStim(
            self.win,
            text=f"Subject {self.subject_label} done!\n\nDecoded:\n{self.output}\n\nPress SPACE to continue",
            pos=(0, 0),
            height=0.07,
            color="white"
        )
        done.draw()
        self.win.flip()
        event.waitKeys(keyList=["space"])
        self.win.close()


def run_speller(ui_results, n_reps=5):
    """
    ui_results: dict like {"A": "decoded string", "B": "decoded string"}
    Called from main.py after decoding is done.
    """
    for subject, pred_string in ui_results.items():
        speller = P300Speller(subject_label=subject)
        speller.run(pred_string, n_reps=n_reps)

    core.quit()
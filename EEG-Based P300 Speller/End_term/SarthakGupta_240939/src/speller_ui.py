from psychopy import visual, core, event
import numpy as np

CHAR_GRID = np.array([
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPQR"),
    list("STUVWX"),
    list("YZ1234"),
    list("56789_")
])

class SpellerUI:

    def __init__(self, label="A"):
        self.label = label
        self.window = visual.Window(
            size=(900, 700),
            color="black",
            fullscr=False,
            units="norm"
        )

        # header (subject info)
        self.header = visual.TextStim(
            self.window,
            text=f"Subject {label}",
            pos=(-0.75, 0.88),
            height=0.06,
            color="grey"
        )

        # output text
        self.output_text = ""
        self.output_display = visual.TextStim(
            self.window,
            text="Output: ",
            pos=(0, 0.88),
            height=0.07,
            color="white"
        )

        # grid creation
        self.cells = []
        for r in range(6):
            row = []
            for c in range(6):
                cell = visual.TextStim(
                    self.window,
                    text=CHAR_GRID[r][c],
                    pos=(-0.5 + 0.2 * c, 0.55 - 0.2 * r),
                    height=0.09,
                    color="white"
                )
                row.append(cell)
            self.cells.append(row)

    def _draw_grid(self):
        for row in self.cells:
            for item in row:
                item.draw()

    def _draw_frame(self):
        self._draw_grid()
        self.output_display.draw()
        self.header.draw()

    def _set_row_highlight(self, row_idx):
        for j in range(6):
            self.cells[row_idx][j].color = "yellow"

    def _set_col_highlight(self, col_idx):
        for i in range(6):
            self.cells[i][col_idx].color = "yellow"

    def _clear_highlight(self):
        for row in self.cells:
            for item in row:
                item.color = "white"

    def _append_char(self, ch):
        self.output_text += ch
        self.output_display.text = f"Output: {self.output_text}"

    def _flash_cycle(self, repetitions, on_time, off_time):
        for _ in range(repetitions):

            rows = list(range(6))
            np.random.shuffle(rows)

            for r in rows:
                self._set_row_highlight(r)
                self._draw_frame()
                self.window.flip()
                core.wait(on_time)

                self._clear_highlight()
                self._draw_frame()
                self.window.flip()
                core.wait(off_time)

            cols = list(range(6))
            np.random.shuffle(cols)

            for c in cols:
                self._set_col_highlight(c)
                self._draw_frame()
                self.window.flip()
                core.wait(on_time)

                self._clear_highlight()
                self._draw_frame()
                self.window.flip()
                core.wait(off_time)

    def start(self, predicted, repetitions=5):

        intro_screen = visual.TextStim(
            self.window,
            text=f"Subject {self.label} — {len(predicted)} characters\n\nPress SPACE to start",
            pos=(0, 0),
            height=0.08,
            color="white"
        )

        intro_screen.draw()
        self.window.flip()
        event.waitKeys(keyList=["space"])

        for ch in predicted:
            if event.getKeys(keyList=["escape"]):
                break

            self._flash_cycle(repetitions, 0.1, 0.075)

            self._append_char(ch)

            self._clear_highlight()
            self._draw_frame()
            self.window.flip()
            core.wait(0.4)

        end_screen = visual.TextStim(
            self.window,
            text=f"Subject {self.label} done!\n\nDecoded:\n{self.output_text}\n\nPress SPACE to continue",
            pos=(0, 0),
            height=0.07,
            color="white"
        )

        end_screen.draw()
        self.window.flip()
        event.waitKeys(keyList=["space"])

        self.window.close()


def launch_speller(results_map, repetitions=5):
    """
    results_map example:
    {"A": "HELLO", "B": "WORLD"}
    """

    for subj, text in results_map.items():
        ui = SpellerUI(label=subj)
        ui.start(text, repetitions=repetitions)

    core.quit()
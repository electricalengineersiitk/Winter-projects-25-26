"""
app.py
------
Desktop GUI app for the Smart Digital Multimeter simulation.
Displays live replayed readings in a classic DMM style using Tkinter.

Run:
    python app.py

The app replays the 50-sample simulation for all three modes (R, C, L),
animating the display as if readings are arriving over an OTG serial link.
"""

import tkinter as tk
from tkinter import font as tkfont
import threading
import time
import os
import sys
import numpy as np

# ── Local imports ──────────────────────────────────────────────────────────────
from measurement import measure_resistance, measure_capacitance, measure_inductance
from autorange import AutoRanger, RANGES
from protocol import format_packet

# ── Simulation helpers (mirror simulate.py logic) ──────────────────────────────

MODES_CFG = {
    "R": {"label": "Resistance", "unit": "Ω",  "measure_fn": measure_resistance,  "ranges": RANGES["R"]},
    "C": {"label": "Capacitance","unit": "F",  "measure_fn": measure_capacitance, "ranges": RANGES["C"]},
    "L": {"label": "Inductance", "unit": "H",  "measure_fn": measure_inductance,  "ranges": RANGES["L"]},
}

N_SAMPLES = 50


def _generate_test_values(ranges, n=N_SAMPLES):
    values = []
    lower = 0.0
    per_range = n // len(ranges)
    for upper in ranges:
        lo = max(lower, upper * 0.01)
        vals = np.logspace(np.log10(lo), np.log10(upper * 0.85), per_range)
        values.extend(vals.tolist())
        lower = upper
    values = values[:n]
    while len(values) < n:
        values.append(values[-1])
    return values


def _build_samples(mode: str):
    cfg = MODES_CFG[mode]
    true_vals = _generate_test_values(cfg["ranges"])
    ranger = AutoRanger(mode)
    ranger.reset()
    samples = []
    ts = 0
    for tv in true_vals:
        meas, err = cfg["measure_fn"](tv)
        result = ranger.process(meas)
        for _ in range(20):
            if result.settled or result.overload:
                break
            result = ranger.process(meas)
        pkt = format_packet(mode, result.active_range, meas, err, ts)
        ts += 100
        samples.append({
            "true":   tv,
            "meas":   meas,
            "err":    err,
            "range":  result.active_range,
            "settled":result.settled,
            "pkt":    pkt,
        })
    return samples


# ── colour palette ─────────────────────────────────────────────────────────────
BG       = "#1a1a2e"   # deep navy
PANEL    = "#16213e"
DISPLAY  = "#0a0a0a"
GREEN    = "#00ff88"   # main readout
DIM      = "#2a6644"
YELLOW   = "#ffd700"
ORANGE   = "#ff8c00"
RED      = "#ff4444"
WHITE    = "#e0e0e0"
GREY     = "#555577"

MODE_COLOURS = {"R": "#00ccff", "C": "#00ff88", "L": "#ff88ff"}


class MultimeterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        np.random.seed(42)

        self.title("Smart Digital Multimeter — OTG Desktop App")
        self.resizable(False, False)
        self.configure(bg=BG)

        # ── State ──────────────────────────────────────────────────────────────
        self._mode      = tk.StringVar(value="R")
        self._running   = False
        self._thread    = None
        self._samples   = {}          # pre-built for each mode
        self._idx       = 0

        # Live display variables
        self._disp_val    = tk.StringVar(value="0.000000")
        self._disp_unit   = tk.StringVar(value="Ω")
        self._disp_range  = tk.StringVar(value="Range  1")
        self._disp_status = tk.StringVar(value="READY")
        self._disp_err    = tk.StringVar(value="Err  ---")
        self._disp_true   = tk.StringVar(value="True  ---")
        self._disp_pkt    = tk.StringVar(value="Awaiting packet …")
        self._disp_sample = tk.StringVar(value="Sample  0 / 50")
        self._disp_mode_label = tk.StringVar(value="RESISTANCE")

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ── UI construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── Title bar ───────────────────────────────────────────────────────────
        title_f = tk.Frame(self, bg=BG)
        title_f.pack(fill="x", padx=20, pady=(16, 4))
        tk.Label(title_f, text="⚡  Smart Digital Multimeter",
                 bg=BG, fg=WHITE,
                 font=("Courier New", 14, "bold")).pack(side="left")
        tk.Label(title_f, text="OTG Serial Link",
                 bg=BG, fg=GREY,
                 font=("Courier New", 10)).pack(side="right")

        # ── Main DMM display ────────────────────────────────────────────────────
        disp_outer = tk.Frame(self, bg=GREY, padx=2, pady=2)
        disp_outer.pack(fill="x", padx=20, pady=6)
        disp_f = tk.Frame(disp_outer, bg=DISPLAY, padx=18, pady=14)
        disp_f.pack(fill="x")

        # Mode label (top-left of display)
        top_row = tk.Frame(disp_f, bg=DISPLAY)
        top_row.pack(fill="x")
        self._mode_lbl = tk.Label(top_row, textvariable=self._disp_mode_label,
                                  bg=DISPLAY, fg=MODE_COLOURS["R"],
                                  font=("Courier New", 11, "bold"))
        self._mode_lbl.pack(side="left")
        tk.Label(top_row, textvariable=self._disp_range,
                 bg=DISPLAY, fg=GREY,
                 font=("Courier New", 11)).pack(side="right")

        # Big value readout
        val_row = tk.Frame(disp_f, bg=DISPLAY)
        val_row.pack(fill="x", pady=(6, 2))
        self._val_label = tk.Label(val_row, textvariable=self._disp_val,
                                   bg=DISPLAY, fg=GREEN,
                                   font=("Courier New", 36, "bold"),
                                   anchor="e", width=14)
        self._val_label.pack(side="left")
        self._unit_label = tk.Label(val_row, textvariable=self._disp_unit,
                                    bg=DISPLAY, fg=GREEN,
                                    font=("Courier New", 22, "bold"), anchor="w")
        self._unit_label.pack(side="left", padx=(6, 0))

        # Status + error row
        bot_row = tk.Frame(disp_f, bg=DISPLAY)
        bot_row.pack(fill="x")
        self._status_lbl = tk.Label(bot_row, textvariable=self._disp_status,
                                    bg=DISPLAY, fg=YELLOW,
                                    font=("Courier New", 10, "bold"))
        self._status_lbl.pack(side="left")
        tk.Label(bot_row, textvariable=self._disp_err,
                 bg=DISPLAY, fg=GREY,
                 font=("Courier New", 10)).pack(side="right")

        # ── Sub-info panel ──────────────────────────────────────────────────────
        info_f = tk.Frame(self, bg=PANEL, padx=14, pady=10)
        info_f.pack(fill="x", padx=20, pady=4)

        tk.Label(info_f, textvariable=self._disp_true,
                 bg=PANEL, fg=GREY, font=("Courier New", 9)).pack(anchor="w")
        tk.Label(info_f, textvariable=self._disp_sample,
                 bg=PANEL, fg=GREY, font=("Courier New", 9)).pack(anchor="w")

        # OTG packet display
        pkt_f = tk.Frame(self, bg="#0d0d1a", padx=14, pady=8)
        pkt_f.pack(fill="x", padx=20, pady=4)
        tk.Label(pkt_f, text="OTG Serial Packet:",
                 bg="#0d0d1a", fg=GREY, font=("Courier New", 8)).pack(anchor="w")
        tk.Label(pkt_f, textvariable=self._disp_pkt,
                 bg="#0d0d1a", fg="#00aa55", font=("Courier New", 8),
                 wraplength=440, justify="left").pack(anchor="w")

        # ── Range indicator bar ─────────────────────────────────────────────────
        range_f = tk.Frame(self, bg=BG)
        range_f.pack(fill="x", padx=20, pady=(4, 2))
        tk.Label(range_f, text="RANGE", bg=BG, fg=GREY,
                 font=("Courier New", 8)).pack(side="left", padx=(0, 8))
        self._range_boxes = []
        for i in range(1, 6):
            b = tk.Label(range_f, text=f" {i} ",
                         bg=GREY, fg=BG,
                         font=("Courier New", 9, "bold"),
                         relief="flat", padx=4, pady=2)
            b.pack(side="left", padx=2)
            self._range_boxes.append(b)

        # ── Mode selector buttons ───────────────────────────────────────────────
        mode_f = tk.Frame(self, bg=BG)
        mode_f.pack(pady=(10, 4))
        self._mode_btns = {}
        for mode, label in [("R", "Ω  Resistance"), ("C", "F  Capacitance"), ("L", "H  Inductance")]:
            btn = tk.Button(
                mode_f, text=label,
                bg=PANEL, fg=MODE_COLOURS[mode],
                activebackground=MODE_COLOURS[mode], activeforeground=BG,
                font=("Courier New", 10, "bold"),
                relief="flat", padx=14, pady=6,
                command=lambda m=mode: self._select_mode(m)
            )
            btn.pack(side="left", padx=6)
            self._mode_btns[mode] = btn

        # ── Control buttons ─────────────────────────────────────────────────────
        ctrl_f = tk.Frame(self, bg=BG)
        ctrl_f.pack(pady=(4, 16))
        self._start_btn = tk.Button(
            ctrl_f, text="▶  Start Replay",
            bg="#004422", fg=GREEN,
            activebackground=GREEN, activeforeground=BG,
            font=("Courier New", 10, "bold"),
            relief="flat", padx=16, pady=6,
            command=self._start
        )
        self._start_btn.pack(side="left", padx=8)
        self._stop_btn = tk.Button(
            ctrl_f, text="■  Stop",
            bg="#330000", fg=RED,
            activebackground=RED, activeforeground=BG,
            font=("Courier New", 10, "bold"),
            relief="flat", padx=16, pady=6,
            command=self._stop, state="disabled"
        )
        self._stop_btn.pack(side="left", padx=8)

        # Select R by default
        self._select_mode("R")

    # ── Mode selection ─────────────────────────────────────────────────────────

    def _select_mode(self, mode: str):
        self._stop()
        self._mode.set(mode)
        c = MODE_COLOURS[mode]
        self._mode_lbl.config(fg=c)
        self._val_label.config(fg=c)
        self._unit_label.config(fg=c)
        self._disp_unit.set(MODES_CFG[mode]["unit"])
        self._disp_mode_label.set(MODES_CFG[mode]["label"].upper())
        self._disp_val.set("0.000000")
        self._disp_status.set("READY")
        self._disp_sample.set("Sample  0 / 50")
        self._disp_pkt.set("Awaiting packet …")
        for b in self._range_boxes:
            b.config(bg=GREY, fg=BG)
        # Pre-build samples on first use
        if mode not in self._samples:
            self._samples[mode] = _build_samples(mode)

    # ── Playback thread ────────────────────────────────────────────────────────

    def _start(self):
        if self._running:
            return
        self._running = True
        self._start_btn.config(state="disabled")
        self._stop_btn.config(state="normal")
        self._idx = 0
        mode = self._mode.get()
        if mode not in self._samples:
            self._samples[mode] = _build_samples(mode)
        self._thread = threading.Thread(target=self._replay_loop, daemon=True)
        self._thread.start()

    def _stop(self):
        self._running = False
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")
        self._disp_status.set("STOPPED")

    def _replay_loop(self):
        mode = self._mode.get()
        samples = self._samples[mode]
        for i, s in enumerate(samples):
            if not self._running:
                break
            self.after(0, self._update_display, mode, i, s)
            time.sleep(0.18)   # ~180 ms per sample → ~9 s total replay
        if self._running:
            self.after(0, self._finish)

    def _update_display(self, mode: str, i: int, s: dict):
        meas = s["meas"]
        err  = s["err"]
        rng  = s["range"]
        pkt  = s["pkt"]

        # Format value
        if abs(meas) >= 1e6 or (abs(meas) < 1e-6 and meas != 0):
            val_str = f"{meas:.4e}"
        elif abs(meas) >= 1:
            val_str = f"{meas:.6g}"
        else:
            val_str = f"{meas:.4e}"
        self._disp_val.set(val_str)

        # Colour: green=settled, yellow=ranging
        colour = MODE_COLOURS[mode] if s["settled"] else YELLOW
        self._val_label.config(fg=colour)
        self._unit_label.config(fg=colour)

        self._disp_range.set(f"Range  {rng}")
        self._disp_status.set("SETTLED" if s["settled"] else "RANGING …")
        self._disp_err.set(f"Err  {err:.3f}%")
        self._disp_true.set(f"True value:  {s['true']:.6g}  {MODES_CFG[mode]['unit']}")
        self._disp_sample.set(f"Sample  {i+1} / 50")
        self._disp_pkt.set(str(pkt))

        # Range indicator bar
        for j, b in enumerate(self._range_boxes):
            if j + 1 == rng:
                b.config(bg=MODE_COLOURS[mode], fg=BG)
            else:
                b.config(bg=GREY, fg=BG)

        # Status label colour
        status_colour = GREEN if s["settled"] else ORANGE
        self._status_lbl.config(fg=status_colour)

    def _finish(self):
        self._running = False
        self._disp_status.set("DONE")
        self._start_btn.config(state="normal")
        self._stop_btn.config(state="disabled")

    def _on_close(self):
        self._running = False
        self.destroy()


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = MultimeterApp()
    app.mainloop()

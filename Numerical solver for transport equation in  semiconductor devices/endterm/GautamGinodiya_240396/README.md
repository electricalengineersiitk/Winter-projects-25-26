## Part 1 — What did you build?
I built a 1D p-n junction semiconductor device simulator using Python and DEVSIM. By running numerical simulations, I extracted internal physical quantities such as electrostatic potential and carrier densities, and measured the resulting I-V curve under forward and reverse bias.

## Part 2 — How to set it up
 
git clone [https://github.com/your-username/your-repo.git](https://github.com/your-username/your-repo.git)
cd your-repo/end_term/semiconductor_sim
pip install -r requirements.txt

## Part 3- How to Run
python device_setup.py   # creates mesh and doping profile
python simulate.py       # runs voltage sweep, saves data
python visualize.py      # generates all plots in results/
python verify.py         # prints theory vs simulation comparison 

## Part 4 — Your results

| Quantity | Theory (analytical) | Simulation (DEVSIM) |
| :--- | :--- | :--- |
| Depletion width (V=0) | 0.431 μm | 0.445 μm |
| Built-in potential | 0.716 V | 0.705 V |
| Forward current (V=0.6V) | 13.81 μA | 13.12 μA |

## Part 5 — Known limitations

This 1D simulation currently ignores advanced recombination models (like Auger or SRH recombination), temperature fluctuation effects, and quantum confinement. Given more time, I would expand this solver to a 2D geometry to analyze vector current fields and integrate more complex mobility models that account for scattering mechanisms.

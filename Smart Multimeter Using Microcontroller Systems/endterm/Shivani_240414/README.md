Digital Multimeter Simulation (Resistance Measurement)

This project is based on simulating how a digital multimeter measures resistance using a voltage divider concept.

In this program, different resistance values are taken and the corresponding voltage is calculated. From that voltage, the resistance is estimated again, similar to how a real multimeter works.

Some additional features are included to make the simulation more realistic:

* Auto-ranging is used to select different reference resistors depending on the input value
* Small random noise is added to represent real measurement errors
* Percentage error between actual and measured values is calculated

Graphs are plotted to understand the behavior:

1. Actual vs Measured resistance graph shows that the measured values closely follow the actual values
2. Error vs Resistance graph shows that error is higher for smaller resistance values and reduces for larger values

From the results, it can be observed that the system works accurately and behaves similar to a real measuring instrument.
## How to Run the Project

1. Navigate to the project folder:

cd end_term/smart_multimeter

2. Install required libraries:

pip install -r requirements.txt

3. Run the simulation:

python simulate.py

This will print the measured resistance values and error percentages in the terminal and generate graphs.

Tools used:
Python, NumPy, Matplotlib

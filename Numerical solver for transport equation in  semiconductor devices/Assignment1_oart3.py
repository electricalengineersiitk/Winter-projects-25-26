import devsim as ds
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Mesh Creation ---
ds.create_1d_mesh(mesh="resistor_mesh")
length = 50e-6
spacing = length / 99

ds.add_1d_mesh_line(mesh="resistor_mesh", pos=0.0, ps=spacing, tag="left")
ds.add_1d_mesh_line(mesh="resistor_mesh", pos=length, ps=spacing, tag="right")

ds.add_1d_contact(mesh="resistor_mesh", name="anode", tag="left", material="metal")
ds.add_1d_contact(mesh="resistor_mesh", name="cathode", tag="right", material="metal")

ds.add_1d_region(mesh="resistor_mesh", material="Silicon", region="Silicon_wire", tag1="left", tag2="right")
ds.finalize_mesh(mesh="resistor_mesh")
ds.create_device(device="resistor", mesh="resistor_mesh")

# --- 2. Parameters ---
ds.set_parameter(device="resistor", region="Silicon_wire", name="q", value=1.6e-19)
ds.set_parameter(device="resistor", region="Silicon_wire", name="kT", value=0.02585)
ds.set_parameter(device="resistor", region="Silicon_wire", name="eps_si", value=11.7*8.854e-14)
ds.set_parameter(device="resistor", region="Silicon_wire", name="ni", value=1e10) # Corrected ni

# --- 3. Doping Profile ---
ds.node_model(device="resistor", region="Silicon_wire", name="Ndn", equation="ifelse(x > 25e-6, 1, 0)")
ds.set_parameter(device="resistor", region="Silicon_wire", name="c1", value=1e14)
ds.node_model(device="resistor", region="Silicon_wire", name="Net_Doping", equation="c1 * Ndn")

# --- 4. Biasing & Equation Setup ---
ds.node_solution(device="resistor", region="Silicon_wire", name="potential")
xvals = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="x")

# Initial Guess (Built-in Potential step)
initial_phi = [0.238 if x > 25e-6 else 0.0 for x in xvals]
ds.set_node_values(device="resistor", region="Silicon_wire", name="potential", values=initial_phi)

ds.edge_from_node_model(device="resistor", region="Silicon_wire", node_model="potential")

ds.node_model(device="resistor", region="Silicon_wire", name="PoissonNode",
              equation="q * (ni * exp(potential/kT) - ni * exp(-potential/kT) - Net_Doping)")

ds.edge_model(device="resistor", region="Silicon_wire", name="PoissonFlux",
              equation="eps_si * EdgeInverseLength * (potential@n1 - potential@n0)")

ds.edge_model(device="resistor", region="Silicon_wire", name="PoissonFlux:potential@n0", equation="-eps_si * EdgeInverseLength")
ds.edge_model(device="resistor", region="Silicon_wire", name="PoissonFlux:potential@n1", equation="eps_si * EdgeInverseLength")

# Derivative of the charge term with respect to potential
ds.node_model(device="resistor", region="Silicon_wire", name="PoissonNode:potential",
              equation="q * (ni/kT * exp(potential/kT) + ni/kT * exp(-potential/kT))")

ds.equation(device="resistor", region="Silicon_wire", name="Poisson", variable_name="potential",
            node_model="PoissonNode", edge_model="PoissonFlux", variable_update="default")

# --- 5. Part 2: Equilibrium Solve (First) ---
ds.contact_node_model(device="resistor", contact="anode", name="anode_bias", equation="potential - 0.0")
ds.contact_node_model(device="resistor", contact="cathode", name="cathode_bias", equation="potential - 0.238")
ds.contact_equation(device="resistor", contact="anode", name="Poisson", node_model="anode_bias")
ds.contact_equation(device="resistor", contact="cathode", name="Poisson", node_model="cathode_bias")

ds.solve(type="dc", absolute_error=1e-5, relative_error=1e-5, maximum_iterations=100)

# Save equilibrium carrier values
phi_eq = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="potential")
n_eq = 1e10 * np.exp(np.array(phi_eq) / 0.02585)
p_eq = 1e10 * np.exp(-np.array(phi_eq) / 0.02585)
# Force the first node to the new boundary value as a starting point

# current_phi = list(ds.get_node_model_values(device="resistor", region="Silicon_wire", name="potential"))
# current_phi[0] = 0.3 
# ds.set_node_values(device="resistor", region="Silicon_wire", name="potential", values=current_phi)
# Create a linear ramp from 0.3 to 0.238
initial_phi = np.linspace(0.3, 0.238, len(xvals))
ds.set_node_values(device="resistor", region="Silicon_wire", name="potential", values=initial_phi)


# --- 6. Part 3: Applied 0.3V Bias ---
ds.contact_node_model(device="resistor", contact="anode", name="anode_bias", equation="potential - 0.3")
ds.contact_equation(device="resistor", contact="anode", name="Poisson", node_model="anode_bias")
ds.solve(type="dc", absolute_error=1e-12, relative_error=1e-8, maximum_iterations=100)

# --- 7. Final Models and Data Extraction ---
ds.node_model(device="resistor", region="Silicon_wire", name="Electrons", equation="ni * exp(potential/kT)")
ds.node_model(device="resistor", region="Silicon_wire", name="Holes", equation="ni * exp(-potential/kT)")

x = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="x")
phi = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="potential")
n = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="Electrons")
p = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="Holes")

# Calculate Excess Carriers (Current State - Equilibrium State)
excess_n = np.array(n) - n_eq
excess_p = np.array(p) - p_eq

# --- 8. Plotting ---
plt.figure(figsize=(15,5))

plt.subplot(1,3,1)
plt.plot(x, phi)
plt.xlabel("x (m)")
plt.ylabel("Potential (V)")
plt.title("Electrostatic Potential (0.3V Bias)")

plt.subplot(1,3,2)
plt.semilogy(x, n, label="Electrons")
plt.semilogy(x, p, label="Holes")
plt.xlabel("x (m)")
plt.ylabel("Concentration (cm⁻³)")
plt.legend()
plt.title("Carrier Concentrations")

plt.subplot(1,3,3)
plt.plot(x, excess_n, label="$\Delta n$")
plt.plot(x, excess_p, label="$\Delta p$")
plt.xlabel("x (m)")
plt.ylabel("Excess Carriers (cm⁻³)")
plt.title("Excess Carrier Concentrations (Non-Equilibrium)")
plt.legend()

plt.tight_layout()
plt.show()
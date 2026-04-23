import devsim as ds
import numpy as np
import matplotlib.pyplot as plt

# Create a 1D mesh for the resistor
ds.create_1d_mesh(mesh="resistor_mesh")

length = 50e-6
spacing = length / 99

ds.add_1d_mesh_line(mesh="resistor_mesh", pos=0.0, ps=spacing, tag="left")
ds.add_1d_mesh_line(mesh="resistor_mesh", pos=length, ps=spacing, tag="right")

ds.add_1d_contact(mesh="resistor_mesh", name="anode", tag="left", material="metal")
ds.add_1d_contact(mesh="resistor_mesh", name="cathode", tag="right", material="metal")

ds.add_1d_region(
    mesh="resistor_mesh",
    material="Silicon",
    region="Silicon_wire",
    tag1="left",
    tag2="right"
)

ds.finalize_mesh(mesh="resistor_mesh")
ds.create_device(device="resistor", mesh="resistor_mesh")

# Some of the required constants
ds.set_parameter( device="resistor",region="Silicon_wire",name="q",value=1.6e-19)
ds.set_parameter( device="resistor",region="Silicon_wire",name="kT",value=0.02585)
ds.set_parameter( device="resistor",region="Silicon_wire",name="eps_0",value=8.854e-14)
ds.set_parameter( device="resistor",region="Silicon_wire",name="eps_si",value=11.7*8.854e-14)

# Doping profile
# intrinsic upto 25um, n-type after that
ds.node_model(device = "resistor", region = "Silicon_wire", name = "Ndn",
        equation = "ifelse(x > 25e-6, 1, 0)")
ds.set_parameter(device = "resistor", region = "Silicon_wire", name = "c1",
                 value = 1e14)
ds.node_model(device = "resistor", region = "Silicon_wire", name = "NDn",
        equation = "c1 * Ndn")
ds.node_model(device = "resistor", region = "Silicon_wire", name = "NAn",
        equation = "0")
ds.node_model(device = "resistor", region = "Silicon_wire", name = "Net_Doping",
        equation = "NDn - NAn")




# Biasing 
# Declare potential
ds.node_solution(
    device="resistor",
    region="Silicon_wire",
    name="potential"
)

# Initialize potential
xvals = ds.get_node_model_values(
    device="resistor",
    region="Silicon_wire",
    name="x"
)
# Create an initial guess array based on x-coordinates
initial_phi = [0.238 if x > 25e-6 else 0.0 for x in xvals]
ds.set_node_values(
    device="resistor",
    region="Silicon_wire",
    name="potential",
    values=initial_phi
)

# Map potential to edges
ds.edge_from_node_model(
    device="resistor",
    region="Silicon_wire",
    node_model="potential"
)


ds.set_parameter(device="resistor", region="Silicon_wire", name="ni", value=1e10)
# Charge term
ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="PoissonNode",
    #equation="-q * Net_Doping"
    equation="q * (ni * exp(potential/kT) - ni * exp(-potential/kT) - Net_Doping)"
)
eps_si = 11.7 * 8.854e-14  # permittivity of Silicon
# Flux term
ds.edge_model(
    device="resistor",
    region="Silicon_wire",
    name="PoissonFlux",
    equation=f"{eps_si} * EdgeInverseLength * (potential@n1 - potential@n0)"
)
# Derivative of Flux with respect to potential at node n0 and n1
ds.edge_model(device="resistor", region="Silicon_wire", name="PoissonFlux:potential@n0", 
              equation=f"-{eps_si} * EdgeInverseLength")
ds.edge_model(device="resistor", region="Silicon_wire", name="PoissonFlux:potential@n1", 
              equation=f"{eps_si} * EdgeInverseLength")
# Assemble Poisson equation
ds.equation(
    device="resistor",
    region="Silicon_wire",
    name="Poisson",
    variable_name="potential",
    node_model="PoissonNode",
    edge_model="PoissonFlux",
    variable_update="default"
)
# Apply boundary conditions
ds.contact_node_model(
    device="resistor",
    contact="anode",
    name="anode_bias",
    equation="potential - 0.3"
)

ds.contact_node_model(
    device="resistor",
    contact="cathode",
    name="cathode_bias",
    equation="potential - 0.238"
)

ds.contact_equation(
    device="resistor",
    contact="anode",
    name="Poisson",
    node_model="anode_bias"
)

ds.contact_equation(
    device="resistor",
    contact="cathode",
    name="Poisson",
    node_model="cathode_bias"
)



ds.solve(
    type="dc",
    absolute_error=1e-3,
    relative_error=1e-3,
    maximum_iterations=200
)


ni = 1e10 # intrinsic carrier concentration in cm^-3
kT = 0.02585  # thermal voltage at room temperature in V
ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="Electrons",
    equation=f"{ni} * exp(potential/{kT})"
)

ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="Holes",
    equation=f"{ni} * exp(-potential/{kT})"
)


ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="ExcessElectrons",
    equation="0"
)

ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="ExcessHoles",
    equation="0"
)



# x = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="x")
# phi = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="potential")
# n   = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="Electrons")
# p   = ds.get_node_model_values(device="resistor", region="Silicon_wire", name="Holes")

# plt.figure(figsize=(12,4))

# plt.subplot(1,3,1)
# plt.plot(x, phi)
# plt.xlabel("x (m)")
# plt.ylabel("Potential (V)")
# plt.title("Electrostatic Potential")

# plt.subplot(1,3,2)
# plt.semilogy(x, n, label="Electrons")
# plt.semilogy(x, p, label="Holes")
# plt.xlabel("x (m)")
# plt.ylabel("Concentration (cm⁻³)")
# plt.legend()
# plt.title("Carrier Concentrations")

# plt.subplot(1,3,3)
# plt.plot(x, np.zeros(len(x)))
# plt.xlabel("x (m)")
# plt.ylabel("Excess carriers")
# plt.title("Excess n, p = 0 (Equilibrium)")

# plt.tight_layout()
# plt.show()

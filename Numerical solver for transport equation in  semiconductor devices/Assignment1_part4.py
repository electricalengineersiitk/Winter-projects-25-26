import devsim as ds
import numpy as np

# Building a 1D Mesh for PN Diode
ds.create_1d_mesh(mesh = "resistor_mesh")
length = 50
division = length/99
ds.add_1d_mesh_line(mesh = "resistor_mesh", pos = 0, ps = division, tag = "left")
ds.add_1d_mesh_line(mesh = "resistor_mesh", pos = length/2, ps = division, tag = "junction")
ds.add_1d_mesh_line(mesh = "resistor_mesh", pos = length, ps = division, tag = "right")

ds.add_1d_contact(mesh = "resistor_mesh", tag = "left", name = "anode", material = "metal")
ds.add_1d_contact(mesh = "resistor_mesh", tag = "right", name = "cathode", material = "metal")

ds.add_1d_region(mesh = "resistor_mesh", material = "Silicon", region= "p-type", tag1="left", tag2 = "junction")
ds.add_1d_region(mesh = "resistor_mesh", material = "Silicon", region = "n-type", tag1="junction", tag2 = "right")

ds.add_1d_interface(mesh = "resistor_mesh", name = "pn_interface", tag = "junction")
ds.finalize_mesh(mesh = "resistor_mesh")
ds.create_device(mesh ="resistor_mesh", device = "pn_diode" )

# Settting up the parameters
# " for p-type region"
ds.set_parameter(device = "pn_diode", region = "p-type", name = "q", value = 1.6e-19)
ds.set_parameter(device = "pn_diode", region = "p-type", name = "kT", value = 0.02585)
ds.set_parameter(device = "pn_diode", region = "p-type", name = "eps_0", value = 8.854e-14)
ds.set_parameter(device = "pn_diode", region = "p-type", name = "eps_si", value = 11.7*8.854e-14)
ds.set_parameter(device = "pn_diode", region = "p-type", name = "ni", value = 1e10)

# " for n-type region"
ds.set_parameter(device = "pn_diode", region = "n-type", name = "q", value = 1.6e-19)
ds.set_parameter(device = "pn_diode", region = "n-type", name = "kT", value = 0.02585)
ds.set_parameter(device = "pn_diode", region = "n-type", name = "eps_0", value = 8.854e-14)
ds.set_parameter(device = "pn_diode", region = "n-type", name = "eps_si", value = 11.7*8.854e-14)
ds.set_parameter(device = "pn_diode", region = "n-type", name = "ni", value = 1e10)

# doping parameters : "c1" and "c2"

ds.set_parameter(device="pn_diode", region="p-type", name="c2", value=1.0) # Acceptor multiplier
ds.set_parameter(device="pn_diode", region="n-type", name="c1", value=1.0) # Donor multiplier
# Doping Profile
ds.node_model(device = "pn_diode", region = "p-type", name = "NAp", equation = "1e17")
ds.node_model(device = "pn_diode", region = "n-type", name = "NDn", equation = "1e17")

## Netdoping
# In the p-type region: NetDoping = (c1*0) - (c2*NAp)
ds.node_model(device="pn_diode", region="p-type", name="NetDoping", 
              equation=" - c2 * NAp")

# In the n-type region: NetDoping = (c1*NDn) - (c2*0)
ds.node_model(device="pn_diode", region="n-type", name="NetDoping", 
              equation="c1 * NDn")

# Get x-coordinates and NetDoping values
x_p = ds.get_node_model_values(device="pn_diode", region="p-type", name="x")
doping_p = ds.get_node_model_values(device="pn_diode", region="p-type", name="NetDoping")

x_n = ds.get_node_model_values(device="pn_diode", region="n-type", name="x")
doping_n = ds.get_node_model_values(device="pn_diode", region="n-type", name="NetDoping")

# Declare potential
ds.node_solution(device = "pn_diode", region = "p-type", name = "Potential")
ds.node_solution(device = "pn_diode", region = "n-type", name = "Potential")
# Initialize potential
xvals_p = ds.get_node_model_values(device="pn_diode", region="p-type", name="x")
xvals_n = ds.get_node_model_values(device="pn_diode", region="n-type", name="x")
# Create an initial guess array based on x-coordinates
V_bi = 0.833 

initial_phi_p = [0.0 for x in xvals_p]
initial_phi_n = [V_bi for x in xvals_n]

ds.set_node_values(device="pn_diode", region="p-type", name="Potential", values=initial_phi_p)
ds.set_node_values(device="pn_diode", region="n-type", name="Potential", values=initial_phi_n)

# Map potential to edges
ds.edge_from_node_model(device = "pn_diode", region = "p-type", node_model = "Potential")
ds.edge_from_node_model(device = "pn_diode", region = "n-type", node_model = "Potential")

ds.node_model(device = "pn_diode", region = "p-type", name = "PoissonNode",
              equation="q * (ni * exp(Potential/kT) - ni * exp(-Potential/kT) - NetDoping)")
ds.node_model(device = "pn_diode", region = "n-type", name = "PoissonNode",
              equation="q * (ni * exp(Potential/kT) - ni * exp(-Potential/kT) - NetDoping)")

ds.edge_model(device = "pn_diode", region = "p-type", name = "PoissonFlux",
              equation ="eps_si * EdgeInverseLength * (Potential@n1 - Potential@n0)")
ds.edge_model(device = "pn_diode", region = "n-type", name = "PoissonFlux",
              equation = "eps_si * EdgeInverseLength * (Potential@n1 - Potential@n0)")

ds.edge_model(device = "pn_diode", region = "p-type", name = "PoissonFlux:Potential@n0",
              equation = "-eps_si * EdgeInverseLength")
ds.edge_model(device = "pn_diode", region = "p-type", name = "PoissonFlux:Potential@n1",
              equation = "eps_si * EdgeInverseLength")
ds.edge_model(device = "pn_diode", region = "n-type", name = "PoissonFlux:Potential@n0",
              equation = "-eps_si * EdgeInverseLength")
ds.edge_model(device = "pn_diode", region = "n-type", name = "PoissonFlux:Potential@n1",
              equation = "eps_si * EdgeInverseLength")

# Derivative of the charge term with respect to potential
ds.node_model(device = "pn_diode", region = "p-type", name = "PoissonNode:Potential",
              equation="q * (ni/kT * exp(Potential/kT) + ni/kT * exp(-Potential/kT))")
ds.node_model(device = "pn_diode", region = "n-type", name = "PoissonNode:Potential",
              equation="q * (ni/kT * exp(Potential/kT) + ni/kT * exp(-Potential/kT))")

# These must be defined for BOTH regions
for r in ["p-type", "n-type"]:
    # The derivative of (Potential@n1 - Potential@n0) with respect to Potential@n0
    ds.edge_model(device="pn_diode", region=r, name="PoissonFlux:Potential@n0",
                  equation="-eps_si * EdgeInverseLength")
    
    # The derivative of (Potential@n1 - Potential@n0) with respect to Potential@n1
    ds.edge_model(device="pn_diode", region=r, name="PoissonFlux:Potential@n1",
                  equation="eps_si * EdgeInverseLength")
    
ds.equation(device = "pn_diode", region = "p-type", name = "Poisson", variable_name = "Potential",
            node_model = "PoissonNode", edge_model = "PoissonFlux", variable_update = "default")
ds.equation(device = "pn_diode", region = "n-type", name = "Poisson", variable_name = "Potential",
            node_model = "PoissonNode", edge_model = "PoissonFlux", variable_update = "default")

\

ds.contact_node_model(device = "pn_diode", contact = "anode", name = "anode_bias", equation = "Potential - 0.0")
ds.contact_node_model(device = "pn_diode", contact = "cathode", name = "cathode_bias", equation = "Potential - 0.833")

ds.contact_equation(device = "pn_diode", contact = "anode", name = "Poisson", node_model = "anode_bias")
ds.contact_equation(device = "pn_diode", contact = "cathode", name = "Poisson", node_model = "cathode_bias")


# we forget to add equation for junction earlier
# Continuous Potential and Displacement across the junction
ds.interface_model(device="pn_diode", interface="pn_interface", name="continuousPotential", 
                   equation="Potential@r0 - Potential@r1")
ds.interface_model(device="pn_diode", interface="pn_interface", name="continuousPotential:Potential@r0", 
                   equation="1")
ds.interface_model(device="pn_diode", interface="pn_interface", name="continuousPotential:Potential@r1", 
                   equation="-1")

ds.interface_equation(device="pn_diode", interface="pn_interface", name="Poisson", 
                      interface_model="continuousPotential", type="continuous")
ds.solve(type = "dc", absolute_error = 1e-12, relative_error = 1e-8, maximum_iterations = 100)




import devsim as ds
from numpy import absolute

ds.create_1d_mesh(mesh = "resistor_mesh")
# domain between 0 to 50um to create 100 node points
length = 50e-6
spacing = length/99 
ds.add_1d_mesh_line(mesh = "resistor_mesh", pos = 0, ps = spacing, tag = "left")
ds.add_1d_mesh_line(mesh = "resistor_mesh", pos = length, ps =spacing, tag = "right")


# adding contacts at the left and right
ds.add_1d_contact(mesh = "resistor_mesh", name = "anode", tag = "left",material = "metal")
ds.add_1d_contact(mesh = "resistor_mesh", name = "cathode", tag = "right", material = "metal")

# adding the region between the two plates
ds.add_1d_region(mesh = "resistor_mesh", material = "Silicon", region = "Silicon_wire", tag1 = "left", tag2 = "right")


#Finalizing the mesh
ds.finalize_mesh(mesh = "resistor_mesh")
ds.create_device(device = "resistor", mesh = "resistor_mesh")

#Goal:- To find the net doping conc in the semiconductor at different points
#Several Constant
q = 1.6e-19  #charge of an electron
eps_0 = 8.854e-14 # permittivity of free space in
eps_si = 11.7*eps_0 # permitivity in Silicon
#Creating a step-function 
ds.node_model(device = "resistor", region = "Silicon_wire", name = "Ndn",
		#equation = "step(x - 25e-6)"
  equation="ifelse(x > 25e-6, 1, 0)"
  )

ds.set_parameter(device="resistor", region="Silicon_wire", name="c1", value=1e14)


ds.node_model(device = "resistor", region = "Silicon_wire", name = "NDn",
		equation = "c1*Ndn")

ds.node_model(device = "resistor", region = "Silicon_wire", name = "NAn",
		equation = "0")

# Net Doping
ds.node_model(device = "resistor", region = "Silicon_wire", name = "Net_Doping",
		equation = "NDn - NAn")
# Net doping
ds.node_model(
    device="resistor",
    region="Silicon_wire",
    name="Net_Doping",
    equation="NDn - NAn"
)

# Inspect values
net = ds.get_node_model_values(
    device="resistor",
    region="Silicon_wire",
    name="Net_Doping"
)

print(net)

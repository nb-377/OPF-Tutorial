# This script was written by Nicholas Barry on 2/16/2022. It is part of a tutorial on optimum power flow.
# This script demonstrates the feasibility of using optimal power flow to simulate a grid-connected microgrid.
# It connects to OpenDSS for some inputs and validation of the resulting voltage and powers.
# This is intended for educational purposes as a spring board for further research in electrical power systems.

# Import libraries
import pandas as pd
#import pyomo.environ as pyo
from pyomo.environ import *
from pyomo.opt import SolverStatus, TerminationCondition
import numpy as np
from dss import DSS
import os
from pyomo.core import *
from pyomo.opt import SolverFactory


class Object(object):                                                       # setup object class
    pass                                                                    # do nothing

dssFileName = r'C:\Users\nicho\OneDrive - The University of Texas at Austin\Desktop\Research\OPF Nanogrid\OPF Nanogrid Tutorial\OpenDSS Files\grid-connected_git.dss'

#------------------------------ Load system and extract admittance matrix---------------------------------------------#
DSS.Start(0)                                                                # initiate DSS COM interface
Text = DSS.Text                                                             # name DSS Text interface to control
Circuit = DSS.Circuits                                                      # setup circuit instance
Solution = DSS.Circuits.Solution                                            # setup solution instance

CircuitPath, CircuitName = os.path.split(dssFileName)                       # break file name off from path
CircuitName_No_Ext= os.path.splitext(CircuitName)[0]                        # isolate the circuit name without extension

Text.Command = r"Clear"                                                     # clear circuit for fresh run
Text.Command = f"Compile ({dssFileName})"                                   # compile DSS file
Text.Command = 'vsource.source.enabled=no'                                  # disable source for Y matrix build
Text.Command = "batchedit load..* enabled=false"                            # disable all loads to build Y matrix
Text.Command = "batchedit PVsystem..* enabled=false"                        # disable all loads to build Y matrix
Text.Command = "batchedit storage..* enabled=false"                         # disable all loads to build Y matrix
Text.Command = "disable vsource.source"                                     # disable voltage source
Text.Command = "CalcV"                                                      # calculate voltage bases
Text.Command = "Solve"                                                      # solve circuit

Y = Circuit.SystemY                                                         # collect Y matrix of file
Y = np.array(Y, dtype = float)                                              # convert Y to a numpy array to do math
Y = np.reshape(Y, (int(np.sqrt(np.size(Y)/2)), -1))                         # reshape Y into a nodexnode matrix
# print(Y)                                                                  # remove comment to review Y
G = Y[: , ::2]                                                              # seperate G from Y (real) for rectangular calcs
B = Y[: , 1::2]                                                             # seperate B from Y (imag)

# These next few lines of code form the B and G matrices used in the optimization below
# It is recommend to conduct this operation algorithmically for larger systems
# OpenDSS node names end in a .1 or .2 or .3 specifying the phase of that node

# Form G matrix for phase A by removing phase B and C entries
G_A =G
G_A = np.delete(G_A,2,axis=1)
G_A = np.delete(G_A,2,axis=1)
G_A = np.delete(G_A,3,axis=1)
G_A = np.delete(G_A,3,axis=1)
G_A = np.delete(G_A,4,axis=1)
G_A = np.delete(G_A,4,axis=1)

G_A = np.delete(G_A,2,axis=0)
G_A = np.delete(G_A,2,axis=0)
G_A = np.delete(G_A,3,axis=0)
G_A = np.delete(G_A,3,axis=0)
G_A = np.delete(G_A,4,axis=0)
G_A = np.delete(G_A,4,axis=0)

# Form G matrix for phase B by remove phase A and B entries
G_B  = G
G_B = np.delete(G_B, 0, axis=1)
G_B = np.delete(G_B, 0, axis=1)
G_B = np.delete(G_B, 1, axis=1)
G_B = np.delete(G_B, 1, axis=1)
G_B = np.delete(G_B, 2, axis=1)
G_B = np.delete(G_B, 2, axis=1)
G_B = np.delete(G_B, 3, axis=1)

G_B = np.delete(G_B, 0, axis=0)
G_B = np.delete(G_B, 0, axis=0)
G_B = np.delete(G_B, 1, axis=0)
G_B = np.delete(G_B, 1, axis=0)
G_B = np.delete(G_B, 2, axis=0)
G_B = np.delete(G_B, 2, axis=0)
G_B = np.delete(G_B, 3, axis=0)

# From G matrix for phase C by removing phase A and B entries
G_C = G
G_C = np.delete(G_C, 0, axis=1)
G_C = np.delete(G_C, 0, axis=1)
G_C = np.delete(G_C, 0, axis=1)
G_C = np.delete(G_C, 1, axis=1)
G_C = np.delete(G_C, 1, axis=1)
G_C = np.delete(G_C, 2, axis=1)
G_C = np.delete(G_C, 2, axis=1)

G_C = np.delete(G_C, 0, axis=0)
G_C = np.delete(G_C, 0, axis=0)
G_C = np.delete(G_C, 0, axis=0)
G_C = np.delete(G_C, 1, axis=0)
G_C = np.delete(G_C, 1, axis=0)
G_C = np.delete(G_C, 2, axis=0)
G_C = np.delete(G_C, 2, axis=0)

# Form B matrix for phase A by remove phase B and C entries
B_A = B
B_A = np.delete(B_A,2,axis=1)
B_A = np.delete(B_A,2,axis=1)
B_A = np.delete(B_A,3,axis=1)
B_A = np.delete(B_A,3,axis=1)
B_A = np.delete(B_A,4,axis=1)
B_A = np.delete(B_A,4,axis=1)

B_A = np.delete(B_A,2,axis=0)
B_A = np.delete(B_A,2,axis=0)
B_A = np.delete(B_A,3,axis=0)
B_A = np.delete(B_A,3,axis=0)
B_A = np.delete(B_A,4,axis=0)
B_A = np.delete(B_A,4,axis=0)

# Form B matrix for phase B by removing phase A and C entries
B_B = B
B_B = np.delete(B_B, 0, axis=1)
B_B = np.delete(B_B, 0, axis=1)
B_B = np.delete(B_B, 1, axis=1)
B_B = np.delete(B_B, 1, axis=1)
B_B = np.delete(B_B, 2, axis=1)
B_B = np.delete(B_B, 2, axis=1)
B_B = np.delete(B_B, 3, axis=1)

B_B = np.delete(B_B, 0, axis=0)
B_B = np.delete(B_B, 0, axis=0)
B_B = np.delete(B_B, 1, axis=0)
B_B = np.delete(B_B, 1, axis=0)
B_B = np.delete(B_B, 2, axis=0)
B_B = np.delete(B_B, 2, axis=0)
B_B = np.delete(B_B, 3, axis=0)

# From B matrix for phase C by removing phase A and B
B_C = B
B_C = np.delete(B_C, 0, axis=1)
B_C = np.delete(B_C, 0, axis=1)
B_C = np.delete(B_C, 0, axis=1)
B_C = np.delete(B_C, 1, axis=1)
B_C = np.delete(B_C, 1, axis=1)
B_C = np.delete(B_C, 2, axis=1)
B_C = np.delete(B_C, 2, axis=1)

B_C = np.delete(B_C, 0, axis=0)
B_C = np.delete(B_C, 0, axis=0)
B_C = np.delete(B_C, 0, axis=0)
B_C = np.delete(B_C, 1, axis=0)
B_C = np.delete(B_C, 1, axis=0)
B_C = np.delete(B_C, 2, axis=0)
B_C = np.delete(B_C, 2, axis=0)

# check to see order of node names for match to Y matrix order, depending on the configuration it may not
node_names = Circuit.AllNodeNames                                           # collect node names in order
print("Admittance calculation node order:")                                 # print output label
print(node_names)                                                           # print results

dss_var = Object()                                                          # setup DSS variable object
Text.Command = r"Clear"                                                     # clear circuit for fresh run
Text.Command = f"Compile ({dssFileName})"                                   # compile DSS file
Text.Command = "CalcV"                                                      # calculate voltage bases
Text.Command = 'Solve mode=yearly stepsize=1m number=1'                     # configure the solve mode for one time step on the load shape
Text.Command = 'Solve'                                                      # solve the circuit in OpenDSS
Text.Command = 'show voltages LN nodes'                                     # use for validation
Text.Command = 'show powers elements'                                       # show the powers by element (notepad window)

# Get voltages
V = np.array(Circuit.AllBusVolts)                                           # grab voltage array from OpenDSS
V = np.reshape(V, (-1,2))                                                   # re-shape into the proper format
VC = np.ravel(V[:,::2]) + 1.0j*np.ravel(V[:,1::2])                          # format array to fit parameters
dss_var.Vang = np.angle(VC)                                                 # in rad
dss_var.Vmag = np.abs(VC)                                                   # in V.
dss_var.Vpu = dss_var.Vmag/120                                              # in pu

dss_var.allNodeNames = np.asarray(Circuit.AllNodeNames, dtype=str)          # pull all node names from OpenDSS
node_names = np.asarray(DSS.Circuits.AllNodeNames, dtype=str)               # format node names as strings
print("Un-altered node order:")                                             # print output label
print(node_names)                                                           # print node names

dss_var.allNodeNames = node_names                                           # add node names to dss_var object
dss_var.Vang = dss_var.Vang                                                 # in rad
dss_var.Vmag = dss_var.Vmag                                                 # in V.
dss_var.Vpu = dss_var.Vpu                                                   # in V.
dss_var.timestep = 60                                                       # time step in seconds
it = 0                                                                      # initiate iteration counter

# ------------------------------------- Simulation configuration options  ---------------------------------------- #
dss_var.sim_length = 1                                                      # simulate 1 time step

output = "robust"                                                           # provides voltages by bus in output

start_hour = 0                                                              # initialize start time, fractional hour
it = int(start_hour * dss_var.timestep)                                     # calculate iteration number based on starting time
hour = start_hour                                                           # set hour to starting hour
it = 1
obj = 'feasible'                                                            # set objective to feasible

############### -------------------- End of Simualation configuration --------------------------------------############

# pull in load shapes
load = pd.read_csv(r"C:\Users\nicho\OneDrive - The University of Texas at Austin\Desktop\Research\OPF Nanogrid\OPF Nanogrid Tutorial\OpenDSS Files\loadshape.csv")
load_2_P_A = load.iloc[:, 0].values
load_2_Q_A = load.iloc[:, 1].values
load_2_P_B = load.iloc[:, 2].values
load_2_Q_B = load.iloc[:, 3].values
load_2_P_C = load.iloc[:, 4].values
load_2_Q_C = load.iloc[:, 5].values
load_1_P_A = load.iloc[:, 6].values
load_1_Q_A = load.iloc[:, 7].values
load_1_P_B = load.iloc[:, 8].values
load_1_Q_B = load.iloc[:, 9].values
load_1_P_C = load.iloc[:, 10].values
load_1_Q_C = load.iloc[:, 11].values
load_3_P_A = load.iloc[:, 12].values
load_3_Q_A = load.iloc[:, 13].values

# --------------------- Setup Output Variables------------------------------------------------------#
time = [None]
load_P = [0]
load_Q = [0]
V_load_1_A = [0]
V_load_1_B = [0]
V_load_1_C = [0]
V_load_2_A = [0]
V_load_2_B = [0]
V_load_2_C = [0]
V_load_3 =[0]

# setup the load objects
load_3_A = Object()
load_3_A.P_rated = 10000
load_3_A.Q_rated = 5000
load_3_A.P = load_3_A.P_rated * load_3_P_A[it]
load_3_A.Q = load_3_A.Q_rated * load_3_Q_A[it]

load_2_A = Object()
load_2_A.P_rated = 10000
load_2_A.Q_rated = 10000
load_2_A.P = load_2_A.P_rated * load_2_P_A[it]
load_2_A.Q = load_2_A.Q_rated * load_2_Q_A[it]

load_2_B = Object()
load_2_B.P_rated = 10000
load_2_B.Q_rated = 10000
load_2_B.P = load_2_B.P_rated * load_2_P_B[it]
load_2_B.Q = load_2_B.Q_rated * load_2_Q_B[it]

load_2_C = Object()
load_2_C.P_rated = 10000
load_2_C.Q_rated = 10000
load_2_C.P = load_2_C.P_rated * load_2_P_C[it]
load_2_C.Q = load_2_C.Q_rated * load_2_Q_C[it]

load_1_A = Object()
load_1_A.P_rated = 10000
load_1_A.Q_rated = 10000
load_1_A.P = load_1_A.P_rated * load_1_P_A[it]
load_1_A.Q = load_1_A.Q_rated * load_1_Q_A[it]

load_1_B = Object()
load_1_B.P_rated = 10000
load_1_B.Q_rated = 10000
load_1_B.P = load_1_B.P_rated * load_1_P_B[it]
load_1_B.Q = load_1_B.Q_rated * load_1_Q_B[it]

load_1_C = Object()
load_1_C.P_rated = 10000
load_1_C.Q_rated = 10000
load_1_C.P = load_1_C.P_rated * load_1_P_C[it]
load_1_C.Q = load_1_C.Q_rated * load_1_Q_C[it]

bus_voltage = 120                                                           # set magnitude of bus voltage in line to neutral
# -------------------------------------------setup OPF model----------------------------------------------------------#
opf = ConcreteModel()                                                       # setup model instance in Pyomo
opf.n = range(len(node_names))                                              # set up number of nodes in Pyomo

node_names_A = np.array([0,3,4,7])                                          # create array of node names for phase A
opf.n_A = range(len(node_names_A))

node_names_B = np.array([1,5,8])                                            # create array of node names for phase A
opf.n_B = range(len(node_names_B))

node_names_C = np.array([2,6,9])                                            # create array of node names for phase A
opf.n_C = range(len(node_names_C))

# define variables for optimization
opf.V_A = Var(opf.n_A, initialize=bus_voltage)                              # initialize voltage magnitude at 1.0 pu for phase A nodes
opf.V_B = Var(opf.n_B, initialize=bus_voltage)                              # initialize voltage magnitude at 1.0 pu for phase B nodes
opf.V_C = Var(opf.n_C, initialize=bus_voltage)                              # initialize voltage magnitude at 1.0 pu for phase C nodes

opf.Vang_A = Var(opf.n_A, initialize=0)                                     # initialize voltage angle for phase A nodes
opf.Vang_B = Var(opf.n_B, initialize=-2.0944)                               # initialize voltage angle for phase B nodes (rads)
opf.Vang_C = Var(opf.n_C, initialize=2.0944)                                # initialize voltage angle for phase C nodes (rads)

opf.Slack_P_A = Var(initialize=3600)                                        # initialize generator real power
opf.Slack_P_B = Var(initialize=2700)                                        # initialize generator real power
opf.Slack_P_C = Var(initialize=2700)                                        # initialize generator real power
opf.Slack_Q_A = Var(initialize=1500)                                        # initialize generator reactive power
opf.Slack_Q_B = Var(initialize=1500)                                        # initialize generator reactive power
opf.Slack_Q_C = Var(initialize=1300)                                        # initialize generator reactive power

opf.P_A = Var(opf.n_A, initialize=0)                                        # initialize node real power
opf.P_B = Var(opf.n_B, initialize=0)                                        # initialize node real power
opf.P_C = Var(opf.n_C, initialize=0)                                        # initialize node real power
opf.Q_A = Var(opf.n_A, initialize=0)                                        # initialize node real power
opf.Q_B = Var(opf.n_B, initialize=0)                                        # initialize node real power
opf.Q_C = Var(opf.n_C, initialize=0)                                        # initialize node real power

#----------------------------- define constraints for optimization----------------------------------------------------#
# Constraints for P and Q calculations to tied with voltage and phase angles
def constraint_P_A(opf, i):                                                   # real power balance equation
    return opf.P_A[i] == opf.V_A[i]*sum(opf.V_A[j]*(G_A[i,j]*cos(opf.Vang_A[i] - opf.Vang_A[j]) + B_A[i,j]*sin(opf.Vang_A[i] - opf.Vang_A[j])) for j in opf.n_A)
opf.constraint_P_A = Constraint(opf.n_A, rule=constraint_P_A)                     # set as rule in Pyomo

def constraint_P_B(opf, i):                                                   # real power balance equation
    return opf.P_B[i] == opf.V_B[i]*sum( opf.V_B[j]*(G_B[i,j]*cos(opf.Vang_B[i] - opf.Vang_B[j]) + B_B[i,j]*sin(opf.Vang_B[i] - opf.Vang_B[j])) for j in opf.n_B)
opf.constraint_P_B = Constraint(opf.n_B, rule=constraint_P_B)                     # set as rule in Pyomo

def constraint_P_C(opf, i):                                                   # real power balance equation
    return opf.P_C[i] == opf.V_C[i]*sum( opf.V_C[j]*(G_C[i,j]*cos(opf.Vang_C[i] - opf.Vang_C[j]) + B_C[i,j]*sin(opf.Vang_C[i] - opf.Vang_C[j])) for j in opf.n_C)
opf.constraint_P_C = Constraint(opf.n_C, rule=constraint_P_C)                     # set as rule in Pyomo

def constraint_Q_A(opf, i):                                                   # reactive power balance equation
    return opf.Q_A[i] == opf.V_A[i]*sum( opf.V_A[j]*(G_A[i,j]*sin(opf.Vang_A[i] - opf.Vang_A[j]) - B_A[i,j]*cos(opf.Vang_A[i] - opf.Vang_A[j])) for j in opf.n_A)
opf.constraint_Q_A = Constraint(opf.n_A, rule=constraint_Q_A)                     # set as rule in Pyomo

def constraint_Q_B(opf, i):                                                   # reactive power balance equation
    return opf.Q_B[i] == opf.V_B[i]*sum( opf.V_B[j]*(G_B[i,j]*sin(opf.Vang_B[i] - opf.Vang_B[j]) - B_B[i,j]*cos(opf.Vang_B[i] - opf.Vang_B[j])) for j in opf.n_B)
opf.constraint_Q_B = Constraint(opf.n_B, rule=constraint_Q_B)                     # set as rule in Pyomo

def constraint_Q_C(opf, i):                                                   # reactive power balance equation
    return opf.Q_C[i] == opf.V_C[i]*sum( opf.V_C[j]*(G_C[i,j]*sin(opf.Vang_C[i] - opf.Vang_C[j]) - B_C[i,j]*cos(opf.Vang_C[i] - opf.Vang_C[j])) for j in opf.n_C)
opf.constraint_Q_C = Constraint(opf.n_C, rule=constraint_Q_C)                     # set as rule in Pyomo

# constraints for real and reactive power injections
def constraint_inject_P_A(opf, n):
    if n == 0:
        return opf.P_A[0] == opf.Slack_P_A
    elif n == 1:
        return opf.P_A[1] == -1*load_3_A.P
    elif n == 2:
        return opf.P_A[2] == -1*load_1_A.P
    elif n == 3:
        return opf.P_A[3] == -1*load_2_A.P
opf.constraint_inject_P_A = Constraint(opf.n_A, rule=constraint_inject_P_A)

def constraint_inject_P_B(opf, n):
    if n == 0:
        return opf.P_B[0] == opf.Slack_P_B
    elif n == 1:
        return opf.P_B[1] == -1*load_1_B.P
    elif n == 2:
        return opf.P_B[2] == -1*load_2_B.P
opf.constraint_inject_P_B = Constraint(opf.n_B, rule=constraint_inject_P_B)

def constraint_inject_P_C(opf, n):
    if n == 0:
        return opf.P_C[0] == opf.Slack_P_C
    elif n == 1:
        return opf.P_C[1] == -1*load_1_C.P
    elif n == 2:
        return opf.P_C[2] == -1*load_2_C.P
opf.constraint_inject_P_C = Constraint(opf.n_C, rule=constraint_inject_P_C)

def constraint_inject_Q_A(opf, n):
    if n == 0:
        return opf.Q_A[0] == opf.Slack_Q_A
    elif n == 1:
        return opf.Q_A[1] == -1*load_3_A.Q
    elif n == 2:
        return opf.Q_A[2] == -1*load_1_A.Q
    elif n == 3:
        return opf.Q_A[3] == -1*load_2_A.Q
opf.constraint_inject_Q_A = Constraint(opf.n_A, rule=constraint_inject_Q_A)

def constraint_inject_Q_B(opf, n):
    if n == 0:
        return opf.Q_B[0] == opf.Slack_Q_B
    elif n == 1:
        return opf.Q_B[1] == -1*load_1_B.Q
    elif n == 2:
        return opf.Q_B[2] == -1*load_2_B.Q
opf.constraint_inject_Q_B = Constraint(opf.n_B, rule=constraint_inject_Q_B)

def constraint_inject_Q_C(opf, n):
    if n == 0:
        return opf.Q_C[0] == opf.Slack_Q_C
    elif n == 1:
        return opf.Q_C[1] == -1*load_1_C.Q
    elif n == 2:
        return opf.Q_C[2] == -1*load_2_C.Q
opf.constraint_inject_Q_C = Constraint(opf.n_C, rule=constraint_inject_Q_C)

#------------------------constraints bus voltages and angles ---------------------------------------------------#
def constraint_slack_A_vmag(opf):
    return opf.V_A[0] == dss_var.Vmag[0]                                 # enforce 1.0 pu at Gen bus
opf.constraint_slack_A_vmag = Constraint(rule=constraint_slack_A_vmag)

def constraint_slack_B_vmag(opf):
    return opf.V_B[0] == dss_var.Vmag[1]                                 # enforce 1.0 pu at Gen bus
opf.constraint_slack_B_vmag = Constraint(rule=constraint_slack_B_vmag)

def constraint_slack_C_vmag(opf):
    return opf.V_C[0] == dss_var.Vmag[2]                                 # enforce 1.0 pu at Gen bus
opf.constraint_slack_C_vmag = Constraint(rule=constraint_slack_C_vmag)

def constraint_bus_V_mag_A_max(opf,n):
    if n == 1 or 2 or 3:
        return opf.V_A[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_A_max = Constraint(opf.n_A, rule=constraint_bus_V_mag_A_max)

def constraint_bus_V_mag_B_max(opf,n):
    if n == 1 or 2:
        return opf.V_B[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_B_max = Constraint(opf.n_B, rule=constraint_bus_V_mag_B_max)

def constraint_bus_V_mag_C_max(opf,n):
    if n == 1 or 2:
        return opf.V_C[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_C_max = Constraint(opf.n_C, rule=constraint_bus_V_mag_C_max)

def constraint_bus_V_mag_A_min(opf,n):
    if n == 1 or 2 or 3:
        return opf.V_A[n] >= 0.95 * bus_voltage                             # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_A_min = Constraint(opf.n_A, rule=constraint_bus_V_mag_A_min)

def constraint_bus_V_mag_B_min(opf,n):
    if n == 1 or 2:
        return opf.V_B[n] >= 0.95 * bus_voltage                             # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_B_min = Constraint(opf.n_B, rule=constraint_bus_V_mag_B_min)

def constraint_bus_V_mag_C_min(opf,n):
    if n == 1 or 2:
        return opf.V_C[n] >= 0.95 * bus_voltage                            # limit bus voltage to 1.05 pu
    else:
        return Constraint.Skip
opf.constraint_bus_V_mag_C_min = Constraint(opf.n_C, rule=constraint_bus_V_mag_C_min)

def constraint_GEN_A_vangle(opf):
    return opf.Vang_A[0] == dss_var.Vang[0]                                # enforce slack bus angle = 0
opf.constraint_GEN_A_vangle = Constraint(rule=constraint_GEN_A_vangle)

def constraint_GEN_B_vangle(opf):
    return opf.Vang_B[0] == dss_var.Vang[1]                                # enforce slack bus angle = -120 deg
opf.constraint_GEN_B_vangle = Constraint(rule=constraint_GEN_B_vangle)
#
def constraint_GEN_C_vangle(opf):
    return opf.Vang_C[0] == dss_var.Vang[2]                                # enforce slack bus angle = 120 deg
opf.constraint_GEN_C_vangle = Constraint(rule=constraint_GEN_C_vangle)

def constraint_bus_V_ang_A(opf,n):
    if n == 1 or 2 or 3:
        return opf.Vang_A[n] <= 15 / 180 * np.pi                           # limit angle to 15 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_A = Constraint(opf.n_A, rule=constraint_bus_V_ang_A)

def constraint_bus_V_ang_B(opf,n):
    if n == 1 or 2:
        return opf.Vang_B[n] <= 15 / 180 * np.pi - 120 / 180 * np.pi      # limit angle to 15 -120 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_B = Constraint(opf.n_B, rule=constraint_bus_V_ang_B)

def constraint_bus_V_ang_C(opf,n):
    if n == 1 or 2:
        return opf.Vang_C[n] <= 15 / 180 * np.pi + 120 / 180 * np.pi      # limit angle to 15 + 120 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_C = Constraint(opf.n_C, rule=constraint_bus_V_ang_C)

def constraint_bus_V_ang_A_min(opf,n):
    if n == 1 or 2 or 3:
        return opf.Vang_A[n] >= -15 / 180 * np.pi                         # limit angle to 15 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_A_min = Constraint(opf.n_A, rule=constraint_bus_V_ang_A_min)

def constraint_bus_V_ang_B_min(opf,n):
    if n == 1 or 2:
        return opf.Vang_B[n] >= -15 / 180 * np.pi - 120 / 180 * np.pi     # limit angle to 15 -120 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_B_min = Constraint(opf.n_B, rule=constraint_bus_V_ang_B_min)

def constraint_bus_V_ang_C_min(opf,n):
    if n == 1 or 2:
        return opf.Vang_C[n] >= -15 / 180 * np.pi + 120 / 180 * np.pi     # limit angle to 15 + 120 deg
    else:
        return Constraint.Skip
opf.constraint_bus_ang_C_min = Constraint(opf.n_C, rule=constraint_bus_V_ang_C_min)

# ------------------------------Run optimization for power flow ------------------------------------------------#

# ------------------- Define Objectives ------------------------------------------------------------------------#
def rule_objective_feasible(opf):                                               # set objectives for optimization
    return 0                                                                    # returns 0

if obj == "feasible":
    opf.objective = Objective(rule=rule_objective_feasible, sense=minimize)     # objective is to minimize 0 (feasibility test)

##Create the ipopt solver plugin using the ASL interface
solver = 'ipopt'                                                                # select IPOPT as solver
solver_io = 'nl'                                                                # setups proper solver interface
stream_solver = False                                                           # True prints solver output to screen
keepfiles = False                                                               # True prints intermediate file names (.nl,.sol,...)
opt = SolverFactory(solver, solver_io=solver_io)                                # setup solver
opt.options['max_iter'] = '5000'                                                # limit iterations
results = opt.solve(opf, keepfiles=keepfiles, tee=False)                        # show results file is tee=true

# print(results.solver)                                                         # print solver results (confirms convergence)
# print('\n')                                                                   # skip line for readability

# re-sort voltages by node to compare with OpenDSS
V_out = [None] * len(node_names)
V_out[0] = opf.V_A.get_values()[0]                                              # gen phase A
V_out[1] = opf.V_B.get_values()[0]                                              # gen phase B
V_out[2] = opf.V_C.get_values()[0]                                              # gen phase C
V_out[3] = opf.V_A.get_values()[1]                                              # load_3
V_out[4] = opf.V_A.get_values()[2]                                              # load_1 phase A
V_out[5] = opf.V_B.get_values()[1]                                              # load_1 phase B
V_out[6] = opf.V_C.get_values()[1]                                              # load_1 phase C
V_out[7] = opf.V_A.get_values()[3]                                              # load_2 phase A
V_out[8] = opf.V_B.get_values()[2]                                              # load_2 phase B
V_out[9] = opf.V_C.get_values()[2]                                              # load_2 phase C

# re-sort voltage angles by node to compare with OpenDSS
V_angle = [None] * len(node_names)
V_angle[0] = opf.Vang_A.get_values()[0]
V_angle[1] = opf.Vang_B.get_values()[0]
V_angle[2] = opf.Vang_C.get_values()[0]
V_angle[3] = opf.Vang_A.get_values()[1]
V_angle[4] = opf.Vang_A.get_values()[2]
V_angle[5] = opf.Vang_B.get_values()[1]
V_angle[6] = opf.Vang_C.get_values()[1]
V_angle[7] = opf.Vang_A.get_values()[3]
V_angle[8] = opf.Vang_B.get_values()[2]
V_angle[9] = opf.Vang_C.get_values()[2]

total_load_P = load_1_A.P + load_3_A.P + load_2_A.P + load_1_B.P + load_2_B.P + load_1_C.P + load_2_C.P   # sum load P
total_load_Q = load_1_A.Q + load_3_A.Q + load_2_A.Q + load_1_B.Q + load_2_B.Q + load_1_C.Q + load_2_C.Q   # sum load Q

# ----------------------------- Print bus voltages ----------------------------------------------------------------#
if output == "robust":
    # # Extract results from opf object and print
    print("OPF Bus Voltages")
    for i in range(0, len(node_names)):
        print(str(node_names[i]) + " voltage is " + str(round(V_out[i], 2)) + " < " + str(round(V_angle[i]*180/np.pi, 2)) + " deg.")

    print("\n")
    print("Slack Bus Active Power")
    print("Phase A " + str(round(opf.Slack_P_A.value/1000, 2)) + " kW, Phase B " + str(round(opf.Slack_P_B.value/1000, 2)) + " kW, Phase C " + str(round(opf.Slack_P_B.value/1000, 2)) + " kW")
    print("Total generator power is " + str(round((opf.Slack_P_A.value+opf.Slack_P_B.value+opf.Slack_P_B.value)/1000,2)) + " kW")

    print("\n")
    print("Slack Bus Reactive Power")
    print("Phase A " + str(round(opf.Slack_Q_A.value/1000, 2)) + " kvar, Phase B " + str(round(opf.Slack_Q_B.value/1000, 2)) + " kvar, Phase C " + str(round(opf.Slack_Q_B.value/1000, 2)) + " kvar")
    print("Total generator power is " + str(round((opf.Slack_Q_A.value+opf.Slack_Q_B.value+opf.Slack_Q_B.value)/1000,2)) + " kvar")

    print("\n")
    print("Circuit Losses")
    print("Active power losses were " + str(round((opf.Slack_P_A.value+opf.Slack_P_B.value+opf.Slack_P_B.value)/1000 - total_load_P/1000,2)) + " kW")
    print("Reactive power losses were " + str(round((opf.Slack_Q_A.value+opf.Slack_Q_B.value+opf.Slack_Q_B.value)/1000 - total_load_Q/1000,2)) + " kvar")

# Take it from here ... 

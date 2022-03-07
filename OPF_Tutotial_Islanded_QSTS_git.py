# This script demonstrates a QSTS study for an islanded microgrid.
# Written by Nicholas Barry, 2/1/2022

# Import libraries
import pandas as pd
import matplotlib.pyplot as plt
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
# change this path to the location of the islanded_git.dss file
dssFileName = r'C:\Users\nicho\OneDrive - The University of Texas at Austin\Desktop\Research\OPF Nanogrid\OPF Nanogrid Tutorial\OpenDSS Files\islanded_git.dss'

#------------------------------ Load system and extract admittance matrix---------------------------------------------#
DSS.Start(0)                                                                # initiate DSS COM interface
Text = DSS.Text                                                             # name DSS Text interface to control
Circuit = DSS.Circuits                                                      # setup circuit instance
Solution = DSS.Circuits.Solution                                            # setup solution instance

CircuitPath, CircuitName = os.path.split(dssFileName)                       # break file name off from path
CircuitName_No_Ext= os.path.splitext(CircuitName)[0]

Text.Command = r"Clear"                                                     # clear circuit for fresh run
Text.Command = f"Compile ({dssFileName})"                                   # compile DSS file
Text.Command = 'vsource.source.enabled=no'                                  # disable source for Y matrix build
Text.Command = "batchedit load..* enabled=false"                            # disable all loads to build Y matrix
Text.Command = "batchedit PVsystem..* enabled=false"                        # disable all loads to build Y matrix
Text.Command = "batchedit storage..* enabled=false"                         # disable all loads to build Y matrix
Text.Command = "disable vsource.source"                                     # disable voltage source
Text.Command = "CalcV"                                                      # calculate voltage bases
Text.Command = 'Solve mode=yearly stepsize=1m number=1'                     # set solve mode in DSS to one step (snapshot)
Text.Command = "Solve"                                                      # solve circuit

Y = Circuit.SystemY                                                         # collect Y matrix of file
Y = np.array(Y, dtype = float)                                              # convert Y to a numpy array to do math
Y = np.reshape(Y, (int(np.sqrt(np.size(Y)/2)), -1))                         # reshape Y into a nodexnode matrix
# print(Y)
G = Y[: , ::2]                                                              # seperate G from Y (real) for rectangular calcs
B = Y[: , 1::2]                                                             # seperate B from Y (imag)

# Form G matrix for phase A
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

# Form G matrix for phase B
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

# From G matrix for phase C
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

# Form B matrix for phase A
B_A =B
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

# Form B matrix for phase B
B_B  = B
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

# From B matrix for phase C
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

# check to see order of node names for match to Y matrix order
node_names = Circuit.AllNodeNames                                           # collect node names in order
print("Admittance calculation node order:")
print(node_names)                                                           # print results

dss_var = Object()
Text.Command = r"Clear"                                                     # clear circuit for fresh run
Text.Command = f"Compile ({dssFileName})"                                   # compile DSS file
Text.Command = "CalcV"                                                      # calculate voltage bases
Text.Command = 'Solve mode=yearly stepsize=1m number=1'
Text.Command = 'Solve'
Text.Command = 'show voltages LN nodes'                                   # use for validation
Text.Command = 'show powers elements'

# Get voltages
V = np.array(Circuit.AllBusVolts)
V = np.reshape(V, (-1,2))
VC = np.ravel(V[:,::2]) + 1.0j*np.ravel(V[:,1::2])
dss_var.Vang = np.angle(VC)                                                 # in rad
dss_var.Vmag = np.abs(VC)                                                   # in V.
dss_var.Vpu = dss_var.Vmag/120                                              # in pu

# all node names
dss_var.allNodeNames = np.asarray(Circuit.AllNodeNames, dtype=str)
node_names = np.asarray(DSS.Circuits.AllNodeNames, dtype=str)
print("Un-altered node order:")
print(node_names)

dss_var.allNodeNames = node_names
dss_var.Vang = dss_var.Vang                                                 # in rad
dss_var.Vmag = dss_var.Vmag                                                 # in V.
dss_var.Vpu = dss_var.Vpu                                                   # in V.
dss_var.timestep = 60                                                       # time step in seconds
it = 0                                                                      # initiate iteration counter

output = "robust"                                                           # provides voltages by bus in output
# output = "slim"                                                           # just the plots

start_hour = 8                                                              # initialize start time, fractional hour
it = int(start_hour * dss_var.timestep)                                     # calculate iteration number based on starting time
hour = start_hour                                                           # set hour to starting hour

obj = 'max soc max PV min Q'                                                # set objective

############### -------------------- End of Simualation configuration --------------------------------------############

# pull in solar radiation load shape data in the .csv file 
irrad = pd.read_csv(r"C:\Users\nicho\OneDrive - The University of Texas at Austin\Desktop\Research\OPF Nanogrid\OpenDSS Files\Solar_Data.csv")
irrad = irrad.iloc[:, 0].values                                             # convert to list and grab G values

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
PV_A_P = [0]
PV_A_Q = [0]
PV_B_P = [0]
PV_B_Q = [0]
PV_C_P = [0]
PV_C_Q = [0]
V_PV_A = [0]
V_PV_B = [0]
V_PV_C = [0]
V_bus_0_A=[None]
V_bus_0_B=[None]
V_bus_0_C=[None]
load_P = [0]
load_Q = [0]
ES_SOC = [0]
ES_A_P = [0]
ES_B_P = [0]
ES_C_P = [0]
ES_A_Q = [0]
ES_B_Q = [0]
ES_C_Q = [0]
V_load_1_A = [0]
V_load_1_B = [0]
V_load_1_C = [0]
V_load_2_A = [0]
V_load_2_B = [0]
V_load_2_C = [0]
V_load_3 =[0]
irrad_out = [0]

ES = Object()                                                               # setup energy storage object
ES.P_rated = 25000                                                          # ES rated power (W)
ES.A_P_rated = ES.P_rated / 3                                               # Phase A rated ESS Real power
ES.A_S_rated = 1.5 * ES.P_rated / 3                                         # ES rated complex power in VA
ES.B_P_rated = ES.P_rated / 3                                               # Phase A rated ESS Real power
ES.B_S_rated = 1.5 * ES.P_rated / 3                                         # ES rated complex power in VA
ES.C_P_rated = ES.P_rated / 3                                               # Phase A rated ESS Real power
ES.C_S_rated = 1.5 * ES.P_rated / 3                                         # ES rated complex power in VA
ES.E_rated = 25000                                                          # ES rated energy in Wh
ES.P_A = ES.A_P_rated                                                       # set to rated power to start
ES.P_B = ES.B_P_rated                                                       # set to rated power to start
ES.P_C = ES.C_P_rated                                                       # set to rated power to start
ES.pf = 0.5                                                                 # ES power factor
ES.A_Q = np.sqrt((ES.A_P_rated/ES.pf)**2 - ES.A_P_rated**2)                 # set to corresponding Q pf for P discharge
ES.B_Q = np.sqrt((ES.B_P_rated/ES.pf)**2 - ES.B_P_rated**2)                 # set to corresponding Q pf for P discharge
ES.A_Q = np.sqrt((ES.C_P_rated/ES.pf)**2 - ES.C_P_rated**2)                 # set to corresponding Q pf for P discharge
ES.SOC = 0.50                                                               # set ES state of charge (%, 0-1)
ES.SOC_updated = ES.SOC                                                     # setup SoC updated instance
# ES.N = 0.965                                                              # one way efficiency of ESS


# This loop executes while the ESS SoC > 5%, which executes the series of optimizations for each time step
while ES.SOC_updated > 0.05:

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

    PV_A = Object()                                                            # setup the first PV system parameters
    PV_A.P_rated = 8000                                                        # PV1 rated power in W
    PV_A.S_rated = 12000                                                       # PV1 rated complex power in VA
    PV_A.pf = 1                                                                # PV 1 power factor
    PV_A.Q = 0                                                                 # PV 1 reactive power
    PV_A.P0 = PV_A.P_rated * irrad[it]                                         # calculated P given irrad

    PV_B = Object()                                                             # setup the first PV system parameters
    PV_B.P_rated = 8000                                                         # PV1 rated power in W
    PV_B.S_rated = 12000                                                        # PV1 rated complex power in VA
    PV_B.pf = 1                                                                 # PV 1 power factor
    PV_B.Q = 0                                                                  # PV 1 reactive power
    PV_B.P0 = PV_B.P_rated * irrad[it]                                          # calculated P given irrad

    PV_C = Object()                                                             # setup the first PV system parameters
    PV_C.P_rated = 8000                                                         # PV1 rated power in W
    PV_C.S_rated = 12000                                                        # PV1 rated complex power in VA
    PV_C.pf = 1                                                                 # PV 1 power factor
    PV_C.Q = 0                                                                  # PV 1 reactive power
    PV_C.P0 = PV_C.P_rated * irrad[it]                                          # calculated P given irrad

    ES.SOC = ES.SOC_updated                                                     # update SoC for next iteration

    bus_voltage = 120
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

    opf.P_A = Var(opf.n_A, initialize=0)                                        # initialize node real power
    opf.P_B = Var(opf.n_B, initialize=0)                                        # initialize node real power
    opf.P_C = Var(opf.n_C, initialize=0)                                        # initialize node real power
    opf.Q_A = Var(opf.n_A, initialize=0)                                        # initialize node real power
    opf.Q_B = Var(opf.n_B, initialize=0)                                        # initialize node real power
    opf.Q_C = Var(opf.n_C, initialize=0)                                        # initialize node real power

    opf.ES_P_A = Var(initialize=2000)                                           # initialize ES phase A
    opf.ES_P_B = Var(initialize=2000)                                           # initialize ES phase B
    opf.ES_P_C = Var(initialize=2000)                                           # initialize ES phase C
    opf.ES_Q_A = Var(initialize=0)                                              # initialize ES phase A
    opf.ES_Q_B = Var(initialize=0)                                              # initialize ES phase B
    opf.ES_Q_C = Var(initialize=0)

    opf.PV_P_A = Var(initialize=PV_A.P0)                                        # initialize PV A real power
    opf.PV_P_B = Var(initialize=PV_B.P0)                                        # initialize PV B real power
    opf.PV_P_C = Var(initialize=PV_C.P0)                                        # initialize PV C real power
    opf.PV_Q_A = Var(initialize=0)                                              # initialize PV A reactive power
    opf.PV_Q_B = Var(initialize=0)                                              # initialize PV B reactive power
    opf.PV_Q_C = Var(initialize=0)                                              # initialize PV C reactive power


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
            return opf.P_A[0] == opf.ES_P_A
        elif n == 1:
            return opf.P_A[1] == -1*load_3_A.P
        elif n == 2:
            return opf.P_A[2] == -1*load_1_A.P
        elif n == 3:
            return opf.P_A[3] == -1*load_2_A.P + opf.PV_P_A
    opf.constraint_inject_P_A = Constraint(opf.n_A, rule=constraint_inject_P_A)

    def constraint_inject_P_B(opf, n):
        if n == 0:
            return opf.P_B[0] == opf.ES_P_B
        elif n == 1:
            return opf.P_B[1] == -1*load_1_B.P
        elif n == 2:
            return opf.P_B[2] == -1*load_2_B.P + opf.PV_P_B
    opf.constraint_inject_P_B = Constraint(opf.n_B, rule=constraint_inject_P_B)

    def constraint_inject_P_C(opf, n):
        if n == 0:
            return opf.P_C[0] == opf.ES_P_C
        elif n == 1:
            return opf.P_C[1] == -1*load_1_C.P
        elif n == 2:
            return opf.P_C[2] == -1*load_2_C.P + opf.PV_P_C
    opf.constraint_inject_P_C = Constraint(opf.n_C, rule=constraint_inject_P_C)

    def constraint_inject_Q_A(opf, n):
        if n == 0:
            return opf.Q_A[0] == opf.ES_Q_A
        elif n == 1:
            return opf.Q_A[1] == -1*load_3_A.Q
        elif n == 2:
            return opf.Q_A[2] == -1*load_1_A.Q
        elif n == 3:
            return opf.Q_A[3] == -1*load_2_A.Q + opf.PV_Q_A
    opf.constraint_inject_Q_A = Constraint(opf.n_A, rule=constraint_inject_Q_A)

    def constraint_inject_Q_B(opf, n):
        if n == 0:
            return opf.Q_B[0] == opf.ES_Q_B
        elif n == 1:
            return opf.Q_B[1] == -1*load_1_B.Q
        elif n == 2:
            return opf.Q_B[2] == -1*load_2_B.Q + opf.PV_Q_B
    opf.constraint_inject_Q_B = Constraint(opf.n_B, rule=constraint_inject_Q_B)

    def constraint_inject_Q_C(opf, n):
        if n == 0:
            return opf.Q_C[0] == opf.ES_Q_C
        elif n == 1:
            return opf.Q_C[1] == -1*load_1_C.Q
        elif n == 2:
            return opf.Q_C[2] == -1*load_2_C.Q + opf.PV_Q_C
    opf.constraint_inject_Q_C = Constraint(opf.n_C, rule=constraint_inject_Q_C)

    # -----------------------------Constraints for PV system -------------------------------------------------------#
    def constraint_PV_A_max(opf):                                              # max P output constraint
        return opf.PV_P_A <= irrad[it] * PV_A.P_rated
        # return opf.PV_P_A <= irrad * PV_A.P_rated
    opf.constrint_PV_A_max = Constraint(rule=constraint_PV_A_max)

    def constraint_PV_B_max(opf):                                              # max P output constraint
        return opf.PV_P_B <= irrad[it] * PV_B.P_rated
        # return opf.PV_P_B <= irrad * PV_B.P_rated
    opf.constrint_PV_B_max = Constraint(rule=constraint_PV_B_max)

    def constraint_PV_C_max(opf):                                              # max P output constraint
        return opf.PV_P_C <= irrad[it] * PV_C.P_rated
        # return opf.PV_P_C <= irrad * PV_C.P_rated
    opf.constraint_PV_C_max = Constraint(rule=constraint_PV_C_max)

    def constraint_PV_A_min(opf):
        return opf.PV_P_A >= 0
    opf.constraint_PV_A_min = Constraint(rule=constraint_PV_A_min)

    def constraint_PV_B_min(opf):
        return opf.PV_P_B >= 0
    opf.constraint_PV_B_min = Constraint(rule=constraint_PV_B_min)

    def constraint_PV_C_min(opf):
        return opf.PV_P_C >= 0
    opf.constraint_PV_C_min = Constraint(rule=constraint_PV_C_min)

    def constraint_PV_A_S_max(opf):
        return (opf.PV_P_A**2 + opf.PV_Q_A**2 <= PV_A.S_rated**2)
    opf.constraint_PV_A_S_max = Constraint(rule=constraint_PV_A_max)

    def constraint_PV_B_S_max(opf):
        return (opf.PV_P_B**2 + opf.PV_Q_B**2 <= PV_B.S_rated**2)
    opf.constraint_PV_B_S_max = Constraint(rule=constraint_PV_B_max)

    def constraint_PV_C_S_max(opf):
        return (opf.PV_C_B**2 + opf.PV_Q_C**2 <= PV_C.S_rated**2)
    opf.constraint_PV_C_S_max = Constraint(rule=constraint_PV_C_max)

    #------------------------constraints bus voltages and angles ---------------------------------------------------#
    def constraint_bus_V_mag_A_max(opf,n):
        if n == 1 or 2 or 3 or 0:
            return opf.V_A[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_A_max = Constraint(opf.n_A, rule=constraint_bus_V_mag_A_max)

    def constraint_bus_V_mag_B_max(opf,n):
        if n == 1 or 2 or 0:
            return opf.V_B[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_B_max = Constraint(opf.n_B, rule=constraint_bus_V_mag_B_max)

    def constraint_bus_V_mag_C_max(opf,n):
        if n == 1 or 2 or 0:
            return opf.V_C[n] <= 1.05 * bus_voltage                              # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_C_max = Constraint(opf.n_C, rule=constraint_bus_V_mag_C_max)

    def constraint_bus_V_mag_A_min(opf,n):
        if n == 1 or 2 or 3 or 0:
            return opf.V_A[n] >= 0.94 * bus_voltage                             # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_A_min = Constraint(opf.n_A, rule=constraint_bus_V_mag_A_min)

    def constraint_bus_V_mag_B_min(opf,n):
        if n == 1 or 2 or 0:
            return opf.V_B[n] >= 0.94 * bus_voltage                             # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_B_min = Constraint(opf.n_B, rule=constraint_bus_V_mag_B_min)

    def constraint_bus_V_mag_C_min(opf,n):
        if n == 1 or 2 or 0:
            return opf.V_C[n] >= 0.94 * bus_voltage                            # limit bus voltage to 1.05 pu
        else:
            return Constraint.Skip
    opf.constraint_bus_V_mag_C_min = Constraint(opf.n_C, rule=constraint_bus_V_mag_C_min)

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

   # ------------------------------- ESS Constraints --------------------------------------------------------------#
    def constraint_ES_P_A_max(opf):
        return opf.ES_P_A <= ES.A_P_rated                                   # limit active power output
    opf.constraint_ES_A_P_max = Constraint(rule=constraint_ES_P_A_max)

    def constraint_ES_P_B_max(opf):
        return opf.ES_P_B <= ES.B_P_rated                                   # limit active power output
    opf.constraint_ES_B_P_max = Constraint(rule=constraint_ES_P_B_max)

    def constraint_ES_P_C_max(opf):
        return opf.ES_P_C <= ES.C_P_rated                                   # limit active power output
    opf.constraint_ES_C_P_max = Constraint(rule=constraint_ES_P_C_max)

    def constraint_ES_P_A_min(opf):
        return opf.ES_P_A >= -ES.A_P_rated                                   # limit active power charging
    opf.constraint_ES_A_P_min = Constraint(rule=constraint_ES_P_A_min)

    def constraint_ES_P_B_min(opf):
        return opf.ES_P_B >= -ES.B_P_rated                                   # limit active power charging
    opf.constraint_ES_B_P_min = Constraint(rule=constraint_ES_P_B_min)

    def constraint_ES_P_C_min(opf):
        return opf.ES_P_C >= -ES.C_P_rated                                   # limit active power charging
    opf.constraint_ES_C_P_min = Constraint(rule=constraint_ES_P_C_min)

    def constraint_ES_S_A_max(opf):
        return opf.ES_P_A**2 + opf.ES_Q_A**2 <= ES.A_S_rated**2             # limit S output from ES phase A
    opf.constraint_ES_S_A_max = Constraint(rule=constraint_ES_S_A_max)

    def constraint_ES_S_B_max(opf):
        return opf.ES_P_B**2 + opf.ES_Q_B**2 <= ES.B_S_rated**2             # limit S output from ES phase B
    opf.constraint_ES_S_B_max = Constraint(rule=constraint_ES_S_B_max)

    def constraint_ES_S_C_max(opf):
        return opf.ES_P_C**2 + opf.ES_Q_C**2 <= ES.C_S_rated**2             # limit S output from ES phase C
    opf.constraint_ES_S_C_max = Constraint(rule=constraint_ES_S_C_max)

    def constraint_ES_soc_max(opf):
        return (ES.SOC - ((opf.ES_P_A + opf.ES_P_B + opf.ES_P_C) * dss_var.timestep / 3600) / ES.E_rated) <= 1
    opf.constraint_ES_soc_max = Constraint(rule=constraint_ES_soc_max)

    def constraint_ES_soc_min(opf):
        return (ES.SOC - ((opf.ES_P_A + opf.ES_P_B + opf.ES_P_C) * dss_var.timestep / 3600) / ES.E_rated) >= 0.05
    opf.constraint_ES_soc_min = Constraint(rule=constraint_ES_soc_min)


    # ------------------------------Run optimization for power flow ------------------------------------------------#

    # ------------------- Define Objective ------------------------------------------------------------------------#
    def objective_max_soc_max_PV_min_Q(opf):
      return -(ES.SOC - ((opf.ES_P_A + opf.ES_P_B + opf.ES_P_C) * dss_var.timestep / 3600) / ES.E_rated) - (opf.PV_P_A + opf.PV_P_B + opf.PV_P_C) / (PV_A.P_rated + PV_B.P_rated + PV_C.P_rated) + 1000 * (opf.PV_Q_A**2 + opf.PV_Q_B**2 + opf.PV_Q_C**2 + opf.ES_Q_A**2 + opf.ES_Q_B**2 + opf.ES_Q_C**2) / ((PV_A.S_rated**2 + PV_B.S_rated**2 + PV_C.S_rated**2) + (ES.A_S_rated**2 + ES.B_S_rated**2 + ES.C_S_rated**2))

    if obj == 'max soc max PV min Q':
        opf.objective = Objective(rule=objective_max_soc_max_PV_min_Q, sense=minimize)   # objective is to min neg SOC

    # Create the ipopt solver plugin using the ASL interface
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
    V_out[0] = opf.V_A.get_values()[0]                                              # bus 0 phase A
    V_out[1] = opf.V_B.get_values()[0]                                              # bus 0 phase B
    V_out[2] = opf.V_C.get_values()[0]                                              # bus 0 phase C
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

    ES.SOC_updated = (ES.SOC - ((opf.ES_P_A.value + opf.ES_P_B.value + opf.ES_P_C.value) * dss_var.timestep / 3600) / ES.E_rated)

    # ----------------------------- Print bus voltages ----------------------------------------------------------------#
    if output == "robust":
        # # Extract results from opf object and print
        print("OPF Bus Voltages")
        for i in range(0, len(node_names)):
            print(str(node_names[i]) + " voltage is " + str(round(V_out[i], 2)) + " < " + str(round(V_angle[i]*180/np.pi, 2)) + " deg.")

        print("\n")
        print("ES Active Power")
        print("Phase A " + str(round(opf.ES_P_A.value/1000, 2)) + " kW, Phase B " + str(round(opf.ES_P_B.value/1000, 2)) + " kW, Phase C " + str(round(opf.ES_P_C.value/1000, 2)) + " kW")
        print("Total ES real power is " + str(round((opf.ES_P_A.value+opf.ES_P_B.value+opf.ES_P_C.value)/1000,2)) + " kW")

        print("\n")
        print("ES Reactive Power")
        print("Phase A " + str(round(opf.ES_Q_A.value/1000, 2)) + " kvar, Phase B " + str(round(opf.ES_Q_B.value/1000, 2)) + " kvar, Phase C " + str(round(opf.ES_Q_C.value/1000, 2)) + " kvar")
        print("Total ES reactive power is " + str(round((opf.ES_Q_A.value+opf.ES_Q_B.value+opf.ES_Q_C.value)/1000,2)) + " kvar")

        print("\n")
        print("PV Active Power")
        print("Phase A " + str(round(opf.PV_P_A.value/1000, 2)) + " kW, Phase B " + str(round(opf.PV_P_B.value/1000, 2)) + " kW, Phase C " + str(round(opf.PV_P_C.value/1000, 2)) + " kW")
        print("Total PV real power is " + str(round((opf.PV_P_A.value+opf.PV_P_B.value+opf.PV_P_C.value)/1000,2)) + " kW")

        print("\n")
        print("PV Reactive Power")
        print("Phase A " + str(round(opf.PV_Q_A.value/1000, 2)) + " kvar, Phase B " + str(round(opf.PV_Q_B.value/1000, 2)) + " kvar, Phase C " + str(round(opf.PV_Q_C.value/1000, 2)) + " kvar")
        print("Total PV reactive power is " + str(round((opf.PV_Q_A.value+opf.PV_Q_B.value+opf.PV_Q_C.value)/1000,2)) + " kvar")

        print("\n")
        print("Circuit Losses")
        print("Active power losses were " + str(round((opf.ES_P_A.value+opf.ES_P_B.value+opf.ES_P_C.value+opf.PV_P_A.value+opf.PV_P_B.value+opf.PV_P_C.value )/1000 - total_load_P/1000,2)) + " kW")
        print("Reactive power losses were " + str(round((opf.ES_Q_A.value+opf.ES_Q_B.value+opf.ES_Q_C.value+opf.PV_Q_A.value+opf.PV_Q_B.value+opf.PV_Q_C.value)/1000 - total_load_Q/1000, 2)) + " kvar")

        # need SoC updated from original
        print("\n")
        print("The ending SoC is " + str(round(ES.SOC_updated,3)))

    # ----------------------- capture results -------------------------------------------------------------------------#
    # append OPF results for each iteration for plotting
    time.append(hour)
    PV_A_P.append(opf.PV_P_A.value)
    PV_A_Q.append(opf.PV_Q_A.value)
    PV_B_P.append(opf.PV_P_B.value)
    PV_B_Q.append(opf.PV_Q_B.value)
    PV_C_P.append(opf.PV_P_C.value)
    PV_C_Q.append(opf.PV_Q_C.value)

    irrad[it] = irrad[it] * 10000
    irrad_out.append(irrad[it])

    ES_SOC.append(ES.SOC_updated)
    ES_A_P.append(opf.ES_P_A.value)
    ES_B_P.append(opf.ES_P_B.value)
    ES_C_P.append(opf.ES_P_C.value)
    ES_A_Q.append(opf.ES_Q_A.value)
    ES_B_Q.append(opf.ES_Q_B.value)
    ES_C_Q.append(opf.ES_Q_C.value)

    V_bus_0_A.append(opf.V_A[0].value)
    V_bus_0_B.append(opf.V_B[0].value)
    V_bus_0_C.append(opf.V_C[0].value)
    V_load_3.append(V_out[3])                                                     # capture load_3 bus voltage
    V_load_1_A.append(V_out[4])                                                    # load_1 phase A voltage
    V_load_1_B.append(V_out[5])                                                    # load_1 Phase B voltage
    V_load_1_C.append(V_out[6])                                                    # load_1 phase C voltage
    V_load_2_A.append(V_out[7])                                                  # load_2 phase A voltage
    V_load_2_B.append(V_out[8])                                                  # load_2 Phase B voltage
    V_load_2_C.append(V_out[9])                                                  # load_2 phase C voltage


    load_P.append(total_load_P)                                                 # append load real power
    load_Q.append(total_load_Q)                                                 # append load reactive power

    it = it + 1                                                                 # increment iteration counter
    print("Timestep " + str(it))                                               # print iteration number
    hour = hour + dss_var.timestep/60/60                                        # advance hour by one time step

# -------------------------- Plot results------------------------------------------------------------#
plot = 'yes'
if plot == 'yes':

    fig, axs = plt.subplots(2,2)
    # plot active powers
    axs[0,0].plot(time, PV_A_P)
    axs[0,0].plot(time, PV_B_P)
    axs[0,0].plot(time, PV_C_P)
    axs[0,0].plot(time, ES_A_P)
    axs[0,0].plot(time, ES_B_P)
    axs[0,0].plot(time, ES_C_P)
    axs[0,0].legend(['PV A','PV B', "PV C",'ES A','ES B', "ES C"])
    axs[0,0].set_title('PV and ES Active Power')

    # plot reactive powers
    axs[0,1].plot(time, PV_A_Q)
    axs[0,1].plot(time, PV_B_Q)
    axs[0,1].plot(time, PV_C_Q)

    axs[0,1].plot(time, ES_A_Q)
    axs[0,1].plot(time, ES_B_Q)
    axs[0,1].plot(time, ES_C_Q)
    axs[0,1].legend(['PV A','PV B', "PV C",'GEN A','GEN B', "Bus 0 C",'ES A','ES B', "ES C"])
    axs[0,1].set_title('PV and ES Reactive Power')

    axs[1,0].plot(time, ES_SOC)
    axs[1,0].set_xlabel("Hour")
    axs[1,0].set_ylabel("ES SoC")
    axs[1,1].set_title('ES State of Charge')

    axs[1,1].plot(time, load_P)
    axs[1,1].plot(time, load_Q)
    axs[1,1].set_xlabel("Hour")
    axs[1,1].set_ylabel("Load [kw or kvar]")
    axs[1,1].legend(['P Load [kW]', 'Q Load [kvar]'])
    axs[1,1].set_title('Microgrid Load')
    plt.show()

    # plot votlage results
    fig, axs = plt.subplots(2,2)
    # plot phase A voltages
    axs[0,0].plot(time, V_bus_0_A)
    axs[0,0].plot(time, V_load_2_A)
    axs[0,0].plot(time, V_load_1_A)
    axs[0,0].legend(['ES', 'load_3', 'load_2', 'load_1'])
    axs[0,0].set_title('Phase A Voltages')

    # plot phase B voltages
    axs[0,1].plot(time, V_bus_0_B)
    axs[0,1].plot(time, V_load_2_B)
    axs[0,1].plot(time, V_load_1_B)
    axs[0,1].legend(['ES', 'load_3', 'load_2', 'load_1'])
    axs[0,1].set_title('Phase B Voltages')

    # plot ES bus voltages
    # plot phase B voltages
    axs[1,0].plot(time, V_bus_0_C)
    axs[1,0].plot(time, V_load_2_C)
    axs[1,0].plot(time, V_load_1_C)
    axs[1,0].legend(['ES', 'load_3', 'load_2', 'load_1'])
    axs[1,0].set_title('Phase C Voltages')

    # show generator voltages
    axs[1,1].plot(time, V_bus_0_A)
    axs[1,1].plot(time, V_bus_0_B)
    axs[1,1].plot(time, V_bus_0_C)
    axs[1,1].legend(['Gen. A', 'Gen. B', 'Gen. C'])
    axs[1,1].set_title('Generator Bus Voltages')
    plt.show()


    plt.step(time,PV_A_P, marker="+", markevery=15)
    plt.step(time,PV_B_P, marker="x", markevery=25)
    plt.step(time,PV_C_P, marker='2', markevery=25)
    plt.step(time,ES_A_P, marker="+", markevery=15)
    plt.step(time,ES_B_P, marker="x", markevery=25)
    plt.step(time,ES_C_P, marker='2', markevery=25)
    plt.step(time,irrad_out)
    plt.xlabel("Hour of Day")
    plt.ylabel("Real Power [W] / Irradiation [W/m^2]x10")
    plt.legend(['PV A','PV B', "PV C",'ES A','ES B', "ES C", "Irradiation"])
    plt.title("ESS and PV Active Powers")
    plt.show()

    plt.step(time, ES_SOC)
    plt.xlabel("Hour")
    plt.ylabel("SoC")
    plt.title('ESS State of Charge')
    plt.show()

# please take this and continue to go forward

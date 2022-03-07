# OPF-Tutorial
Optimal Power Flow for Unbalanced Three-Phase Microgrids Using Interior Point Optimizer, IPOPT
Link to published paper 

Abstract: 
This work provides a tutorial on formulating and solving optimal power flow (OPF) problems for grid-connected and islanded, unbalanced, three-phase microgrids using an open-source software package. Because of its robustness and ease of use, this paper adopts the IPOPT solver package implemented in Python using PYOMO to formulate the OPF problem. Optimal management of photovoltaic generation and energy storage during islanded and grid-connected operations are used as illustrative OPF problems. Formulation of OPF variables, constraints, objectives, and solutions for snapshot and quasi-static time series simulation implemented in PYOMO will be presented and explained in detail, along with Python code snippets. The IPOPT solver solution of a multi-hour simulation horizon demonstrating optimized operational endurance of a multi-phase, islanded, and unbalanced microgrid will be presented. Methods to validate an IPOPT solution using OpenDSS, an open-source power systems distribution system simulation software, will also be demonstrated.

Required Software: 
1. Anaconda – Python distribution with common software and packages, https://www.anaconda.com/products/individual
    Note: Other Python distributions will not usually support required packages
    
2. An IDE for Anaconnda, Pycharm is an excellent IDE for Python and free with a .edu email:  https://www.jetbrains.com/pycharm/ 
 
3. Pyomo: Optimization package: https://pyomo.readthedocs.io/en/stable/ or use “conda install -c conda-forge pyomo” Please see https://anaconda.org/conda-forge/pyomofor more detail. 

4. IPOPT: Optimization solver: https://coin-or.github.io/Ipopt/   conda install -c conda-forge/label/cf201901 ipopt
Please see https://anaconda.org/conda-forge/ipoptfor more detail. To verify IPOPT is installed properly, navigate to your working directory in the terminal. Type ‘pyomo help -s’. This command will list all installed solvers and their versions. Scroll through the output to locate IPOPT, which if correctly installed will have a ‘+’ in from of its listing and version 3.11.1 listed. 

5. DSS-python: https://pypi.org/project/dss-python/ (replaces win32 COM interface discussed in user docs)

6. Open DSS: https://sourceforge.net/projects/electricdss/ 

7. Notepad ++: This software is used for the improved readability of OpenDSS results. It is available from: https://notepad-plus-plus.org/downloads/.

Directory: 

loadshape.csv: This is the load shape file for the microgrid loads.

Solar Data.csv: This is the solar radiation data used to calculate PV generation for the QSTS simulation.

grid-connected_git.dss: This OpenDSS file contains the grid connected microgrid simulated in Section IV A, snapshot solution, grid connected.
OPF_Tutorial_Validation_git.py: This python file implements the OPF for the grid connected microgrid as in Section IV A. 

islanded_git.dss: This OpenDSS file contains the islanded microgrid model used in Sections IV B and D.
OPF_Tutorial_islanded_git.py: This python file demonstrates the islanded microgrid calcualtions with OPF.

islanded_charging_git.dss: This is the OpenDSS file for section IV C. 
OPF_Tutorial_charging_git.py: This is the python file for the charging of the ESS while islanded, from Section IV C. 

OPF_Tutorial_QSTS_git: This is the Python file for the QSTS simulation in Section IV D. 

Note: 
The loadshape.csv and Solar Data.csv files are required by OpenDSS for all simulations. Ensure to update local directories once installed.

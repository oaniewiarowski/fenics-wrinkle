# fenics-wrinkle

## REPLICATION OF RESULTS
You must have FEniCS (2019) and MOSEK (v9) installed, see:

https://fenics.readthedocs.io/en/latest/installation.html

https://www.mosek.com/downloads/

Next, download the `fenics-optim` package:

https://gitlab.enpc.fr/navier-fenics/fenics-optim) 

Enter the repo:

`cd fenics-optim`

and install the package as editable:

`pip3 install -e`.

Replace the following two files with modified versions in this repo:

`mosek_io.py`

`convex_function.py`

To generate the plots, you will need to have installed matplotlib.

Some visualizations were made externally using Paraview 
www.paraview.org

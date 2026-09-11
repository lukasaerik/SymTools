# SymTools
Tools for handling symmetric models and maps in ChimeraX

## Installation
Save ```symtools.py``` to a desired location (```/path/to/symtools.py```)

Open in ChimeraX:
```
open /path/to/symtools.py
```
If desired, this can be automatically done at ChimeraX startup by adding the above line to Settings -> Startup -> Execute these commands at startup

## Running
After opening, axes are found for map N using command ```symaxes #N``` and for model M with Cn or Dn symmetry (e.g., D2) ```symaxes #M D2```. This creates a new group which holds the model/map, as well as the axis/axes and any labels. For all symmetries but D2, only the principal axis is generated. For D2 symmetry, all three C2 axes are generated and labelled 1, 2, and 3.

To align two (non-D2-symmetric) groups, use command ```symfit #X into #Y```. For two D2 symmetric groups, specify which axes you wish to be colinear, e.g., ```symfit #X ax 2 into #Y ax 1```

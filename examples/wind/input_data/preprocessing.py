#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 31 22:35:39 2021

@author: alexanderniewiarowski
"""
import dolfin as df
import numpy as np

topological_dim = 2
geometrical_dim = 3

COORDS = np.loadtxt("nodes.txt", dtype='float')
CON = np.loadtxt("elements.txt",dtype='uint') - 1  # from MATLAB, start from 0

num_local_vertices = COORDS.shape[0]
num_global_vertices = num_local_vertices  # True if run in serial
num_local_cells = CON.shape[0]
num_global_cells = num_local_cells

# Create mesh object and open editor
mesh = df.Mesh()
editor = df.MeshEditor()
editor.open(mesh, "triangle", topological_dim, geometrical_dim)
editor.init_vertices_global(num_local_vertices, num_global_vertices)
editor.init_cells_global(num_local_cells, num_global_cells)

# Add verticess
for i, coord in enumerate(COORDS):
    editor.add_vertex(i, coord)

# Add cells
for i, cell in enumerate(CON):
    editor.add_cell(i, cell)

# Close editor
editor.close()

# f = df.File('mesh.pvd')
# f << mesh

# %% Original pressures for comparison
W = df.FunctionSpace(mesh, 'DG', 0)
ret_dofmap = W.dofmap()
p = df.Function(W)

PRESSURES = np.loadtxt("pressures.txt", dtype='float')
assert(len(PRESSURES) == num_global_cells)

temparray = np.zeros(num_global_cells)
for c, mesh_cell in enumerate(df.cells(mesh)):
    temparray[ret_dofmap.cell_dofs(mesh_cell.index())] = PRESSURES[c]

p.vector()[:] = temparray

f = df.File('PRESSURES.pvd')
f << p

with open(f'pressures.npy', 'wb') as f:
            np.save(f, PRESSURES)

# %% build list of cell midpoints
R = 10
MID_PTS = []
for c, mesh_cell in enumerate(df.cells(mesh)):
    MID_PTS.append([v for v in mesh_cell.midpoint()])
MID_PTS = np.array(MID_PTS)

# Find intersection of sphere with radius 10
NEW_MID_PTS = np.zeros((num_global_cells, 3))
for i, mp in enumerate(MID_PTS):
    scale_factor = R/np.sqrt(mp.dot(mp))
    corrected = mp*scale_factor
    NEW_MID_PTS[i] = corrected
    # print(np.sqrt(corrected.dot(corrected)))

# %%
# Build inverse mapping to parametric sphere

PRM_COORDS = np.zeros((num_global_cells, 2))
for i, mp in enumerate(NEW_MID_PTS):
    X = mp[0]
    Y = mp[2]
    Z = mp[1]
    t = -1/(Z/R+1)
    PRM_COORDS[i] = [X/R*t, Y/R*t]

with open(f'prm_coords.npy', 'wb') as f:
            np.save(f, PRM_COORDS)
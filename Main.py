# Main.py
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import torch
from plane import Plane
import utils
from tqdm import tqdm

# check GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
#%% Module 1: Import the point cloud from STL file and prepare the data
# Read mesh from STL file and define normals and color
stl_object = '../3_down.stl'
mesh = o3d.io.read_triangle_mesh(stl_object)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0, 1, 0])

# Transform to point cloud numpy 2D matrix
pcd = np.asarray(mesh.vertices)


#%% Module 2: Rotate the main plane so it's parrallel to X-Y plane and calculate planes
# Define main plane object and calculate equation to calculate rotation
Main_plane = Plane()
Main_plane_eq, Main_plane_inliers = Main_plane.fit(pcd,thresh=0.03)

# Find angels to rotate the point cloud and rotate it- currently not necessary 
#angels = utils.extractAngles(Main_plane_eq)
#pcd = utils.Rotate(pcd,angels)

# sort by x and then by y values before order in sample grid- currently not necessary 
#pcd = utils.SortPointCloud(pcd)

# calculate grid, from unstructred to structured grid - currently not necessary 
#Grid = utils.CreateMesh(pcd)

# recalculate the plain- currently not necessary 
#Main_plane = Plane()
#Main_plane_eq, Main_plane_inliers = Main_plane.fit(pcd,thresh=0.03)

#%% Module 3: Idetify the grooves and divide into different point could
# calculate the maximum height in y-axis projection
max_height_y = utils.GetProjection(pcd,Main_plane_eq,axis='y')


# plot single groove
index = np.where(np.logical_and(pcd[:,1] > -20, pcd[:,1] < -18.5))[0]
pcd_groove_1 = pcd[index,:]

utils.DrawPointCloud(pcd_groove_1)





#%% Module 4: Fix artifacts in every groove before calculating volume
groove = pcd






#%% Module 5: Calculate volume and save results    
volume = utils.CalculateIntegral(groove,Main_plane_eq)







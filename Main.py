# Main.py
import open3d as o3d
import numpy as np
import torch
from plane import Plane
import utils
from tqdm import tqdm
import matplotlib.pyplot as plt

# check GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 
#%% Module 1: Import the point cloud from STL file and convert to PointCloud
# Read mesh from STL file and define normals and color
stl_object = '../B14_Bottom.stl'
mesh = o3d.io.read_triangle_mesh(stl_object)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0, 1, 0])

# Transform to point cloud numpy 2D matrix
pcd = np.asarray(mesh.vertices)


#%% Module 2: Optional - Rotation
# Option to Rotate the Point Cloud with R rotation matrix
# (R (numpy.ndarray[float64[3, 3]]) – The rotation matrix)
# R – The rotation matrix
# center (numpy.ndarray[float64[3, 1]]) – Rotation center used for transformation
#PointCloud.rotate(R, R, center)


#%% Module 3: Transform the PointCloud to PointCloud on a structred grid
voxel_size = 0.02
pcd_grid = utils.PointCloud2Grid(pcd,voxel_size)

# calculate the delta_x and delta_y for volume calculation
delta_x = (np.max(pcd[:,0]) - np.min(pcd[:,0])) / len(np.unique(pcd_grid[:,0]))
delta_y = (np.max(pcd[:,1]) - np.min(pcd[:,1])) / len(np.unique(pcd_grid[:,1]))
delta_z = (np.max(pcd[:,2]) - np.min(pcd[:,2])) / len(np.unique(pcd_grid[:,2]))

# redefine pcd
pcd_old = pcd; pcd = pcd_grid;

#%% Module 4: Calculate Main plane and delta
Main_plane = Plane()
Main_plane_eq, Main_plane_inliers = Main_plane.fit(pcd,thresh=0.03)



#%% Module 5: Idetify the grooves and divide into different point could
# calculate the maximum height in y-axis projection
axis = 'x'; Interval_size = 1
axis_intervals,Projection = utils.GetProjection(pcd,Main_plane_eq,Interval_size,axis,True)

# divide into grooves
quantile = 0.19
num_margin_intervals = 20;

grooves = utils.DivideGroovesProjection(pcd,Projection,axis_intervals,quantile,
                                        num_margin_intervals,axis)
pcd_grooves,colors = utils.PlotSegmentedGrooves(grooves)

#%% Module 6: Cut the sides of the point cloud 
# take one groove
grooves_new = []
# define parameters for the function
quantileLow = 0.05; quantileHigh = 0.93; axis = 'y'

for i in range(len(grooves)):
    groove = grooves[i][0]
    groove_analyzed = utils.CutEdges(groove,Main_plane_eq,quantileLow,quantileHigh,axis,False)
    grooves_new.append(groove_analyzed)


#%% Module 7: Calculate the volume of the groove and plot results
volumes = []
grooves_fill,colors_fill = utils.CreateGrooveFill(pcd,pcd,pcd,
                                                  Main_plane_eq,(0,0,0),stone=True)

for i in range(len(grooves_new)):
    print('Groove number %d ' % i)
    groove_new = grooves_new[i]
    volume = utils.CalculateIntegral(groove_new,Main_plane_eq,delta_x,delta_y,delta_z)
    volumes.append(volume)
    grooves_fill,colors_fill = utils.CreateGrooveFill(groove_new,grooves_fill,
                                                      colors_fill,Main_plane_eq,colors[i],
                                                      stone=False)
# Draw results

for volume in volumes: print('\nThe groove volume is: %.2f mm^3' % volume)
grooves_fill = grooves_fill.cpu().numpy()
colors_fill = colors_fill.cpu().numpy()

utils.DrawVoxelGrid(grooves_fill,colors_fill,voxel_size=1)


#%% Module 8: Plot only grooves
stone_length = pcd.shape[0]
grooves_fill_no_stone = grooves_fill[stone_length:,:]
colors_fill_no_stone = colors_fill[stone_length:,:]

utils.DrawVoxelGrid(grooves_fill_no_stone,colors_fill_no_stone,voxel_size=1)



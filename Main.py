# Main.py
import open3d as o3d
import numpy as np
import torch
from plane import Plane
import utils

# check GPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

 
#%% Module 1: Import the point cloud from STL file and prepare the data
# Read mesh from STL file and define normals and color
stl_object = '../C06_Bottom.stl'
mesh = o3d.io.read_triangle_mesh(stl_object)
mesh.compute_vertex_normals()
mesh.paint_uniform_color([0, 1, 0])

# Transform to point cloud numpy 2D matrix
pcd = np.asarray(mesh.vertices)

# Option to Rotate the Point Cloud with R rotation matrix
# (R (numpy.ndarray[float64[3, 3]]) – The rotation matrix)
# R – The rotation matrix
# center (numpy.ndarray[float64[3, 1]]) – Rotation center used for transformation
#PointCloud.rotate(R, R, center)


#%% Module 2: Define main plane object and calculate equation to calculate rotation
Main_plane = Plane()
Main_plane_eq, Main_plane_inliers = Main_plane.fit(pcd,thresh=0.03)
 


#%% Module 3: Idetify the grooves and divide into different point could
# calculate the maximum height in y-axis projection
axis = 'x'
axis_intervals,Projection = utils.GetProjection(pcd,Main_plane_eq,axis)

# divide into grooves
quantile = 0.19
num_margin_intervals = 50;
grooves = utils.DivideGroovesProjection(pcd,Projection,axis_intervals,quantile,
                                        num_margin_intervals,axis)
pcd_grooves = utils.PlotSegmentedGrooves(grooves)

# cut the side of the point cloud 

 
#%% create grid based on point cloud 
Grid = utils.CreateMesh(grooves[0][0])

# calculate new point cloud based on Grid

# calculate plane point cloud based on Grid


# calculate volume based on two point clouds




#%% Voxel Grid Open3D
pointcloud = o3d.geometry.PointCloud()
pointcloud.points = o3d.utility.Vector3dVector(pcd)
VoxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud,0.01)

o3d.visualization.draw_geometries([VoxelGrid])

#%%Voxel Grid pyntcloud
from pyntcloud import PyntCloud

cloud = PyntCloud.from_instance("open3d", mesh)
voxelgrid_id = cloud.add_structure("voxelgrid", n_x=200, n_y=200, n_z=200)
new_cloud = cloud.get_sample("voxelgrid_nearest", voxelgrid_id=voxelgrid_id, as_PyntCloud=True)
a = new_cloud.points


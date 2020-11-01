# Utils functions
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def CalculateIntegral(groove,plane_eq,delta_x,delta_y,delta_z):
    '''
    This function calculate the ablation volume of the groove. 
    The groove is an ordered point cloud, and the plane equation blocks
    The groove from above. The edge of the groove are merging with the plain equation
    The plain the and the stone are parrallel to X-Y plane in the coordinate system

    Parameters
    ----------
    groove : 2D numpy matrix, shape (N,3) float64
        Ordered Point cloud, with interval of delta_x, delta_y.
        more than one z-value is possible
    plane_eq : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0.
    delta_x : float64
        The average distance between points in the scanning in x-axis
    delta_y : float64
        The average distance between points in the scanning in y-axis

    Returns
    -------
    volume : (1,1) float64 
        Ablation volume in units of [mm^3]

    '''
    # take sample grid and calculate plane 3D grid
    # coeff_x*x + coeff_y*y + coeff_z*z + const = 0 to:
    # z = (coeff_x*x + coeff_y*y + const) / (-coeff_z)
    groove_xy = groove[:,0:-1]
    plane = (plane_eq[0] * groove_xy[:,0] + plane_eq[1] * groove_xy[:,1] + plane_eq[3])/(-plane_eq[2]) # calculate z
    
    # (delta_x / 2) * (delta_y / 2) * z = Infinitesimal volume
    volume_infi = delta_x * delta_y * delta_z # dx*dy*dz
    diff = plane - groove[:,2] # vector of different 
    diff = diff.clip(min=0) # zeros values to zero
    volume = np.sum(volume_infi * diff) # dx*dy*dz sum
    
    
    
    
    return volume

def CreateGrooveFill(groove,grooves_fill,colors_fill,plane_eq,color,stone=False):
    '''
    This function create points cloud of every groove in seperate with different 
    Color, based on the stone point cloud, the groove boundries, and main plane
    All calculation are preformed on GPU.

    Parameters
    ----------
    groove : 2D numpy matrix, shape (M,3) float64
        Point cloud, represent only one groove after the edges where cutted
        in case stone is True: the original Point Cloud of the stone
    grooves_fill : 2D Torch matrix, shape (N,3) float64
        Poind Cloud of the stone and the grooves, this variabels move between
        Iterations, each iteration new groove is added to the point cloud
    colors_fill : 2D Torch matrix, shape (N,3) float64
        Matrix of the same size as colors_fill, holds the color for each groove
    plane_eq : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0
    color : 1D Numpy array with size of (1,3)
        Color to assign to the groove
    stone : Boolean
        if stone True, initiate the grooves_fill and colors_fill Tensor on GPU
        if stone is False, create data points of the main plane and groove height
        and add them to grooves_fill, and add points to colors_fill
    Returns
    -------
    grooves_fill : 2D Torch matrix, shape (N,3) float64
        Poind Cloud of the stone and the grooves, this variabels move between
        Iterations, each iteration new groove is added to the point cloud
    colors_fill : 2D Torch matrix, shape (N,3) float64
        Matrix of the same size as colors_fill, holds the color for each groove
    '''
    if stone == True:
        # In this case, the variables grooves_fill and colors_fill, don't exist 
        # and we starting to build the point cloud to display
        
        # In this case, we insert the groove input as the stone point cloud and 
        # assing it to grooves_fill to begin filling
        grooves_fill = groove
        
        # for the stone we assign the color black (0,0,0)
        color = np.zeros((1,3))
        colors_fill = np.repeat(color,grooves_fill.shape[0],axis=0)
        
        # send to GPU
        grooves_fill = torch.from_numpy(grooves_fill).to(device)
        colors_fill = torch.from_numpy(colors_fill).to(device)
        
        
    else:
        # In this case we have already initiled the variables grooves_fill and colors_fill
        # and we need to add new points to them
        
        # Stage 1: Calculate z-axis of the plane
        # the same size as groove_grid
        plane = (plane_eq[0] * groove[:,0] + plane_eq[1] * groove[:,1] + plane_eq[3])/(-plane_eq[2]) 
        # send to GPU
        plane = torch.from_numpy(plane).to(device)
        groove = torch.from_numpy(groove).to(device)
        color = torch.from_numpy(color).to(device)
        
        # create a 2D matrix (N,3) for every point the groove
        for point in tqdm(range(groove.shape[0])):
            # the height of the main plane
            groove_Z = groove[point,2]
            
            # if plane isn't higher than the hight of the groove, fill it
            if groove_Z < plane[point]:
                # Version 1: create all points between stone and plane
                if False:
                    # create vector of z_values to comlete the groove 
                    z_fill = torch.arange(groove_Z+1,plane[point]+1,1).to(device)
                    z_fill = torch.unsqueeze(z_fill,1)
                    
                    # replecate the x,y point (grid points)
                    xy = torch.unsqueeze(groove[point,0:-1],0).to(device)
                    xy = xy.repeat(z_fill.shape[0],1)
                # Version 2: create only twp points, plane and groove hieght
                if True:
                    z_fill = torch.tensor([groove_Z+1,plane[point]]).to(device)
                    z_fill = torch.unsqueeze(z_fill,1)
                    xy = torch.unsqueeze(groove[point,0:-1],0)
                    xy = xy.repeat(2,1).to(device)
                
                # concatenate the x,y point with z-valus
                point_fill = torch.cat((xy,z_fill),1)
                
                # add the points to grooves_fill
                grooves_fill = torch.cat((grooves_fill,point_fill),0)
        
        # after we created all points, we count number of points added, to increase
        # the color matrix in the same size
        num_of_points_added = grooves_fill.shape[0]- colors_fill.shape[0]
        color_new = torch.unsqueeze(color,0).to(device)
        color_new = color_new.repeat(num_of_points_added,1)
        colors_fill = torch.cat((colors_fill,color_new),0)
    
        
    return grooves_fill,colors_fill

def extractAngles(plane):
    '''
    This function calculate the angels to rotate the point cloud
    based in the plane coeeficents, so the plane will be parrallel to X-Y plane

    Parameters
    ----------
    plane : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0.

    Returns
    -------
    angels : list of (1,3) float64 
        angels in [Rad] units

    '''
    A,B,C,D = plane
    
    theta_x = (A / np.sqrt(A**2 + B**2 + C**2))
    theta_y = (B / np.sqrt(A**2 + B**2 + C**2))
    theta_z = (C / np.sqrt(A**2 + B**2 + C**2))
    
    angels = (theta_x,theta_y,theta_z)
    print(angels)
    
    return angels
    

def TensorList2Array(tensorlist):
    '''
    This function take a list of tensors in GPU and transform it to numpy array in CPU

    Parameters
    ----------
    tensorlist : List of Tensor objects float64, the list itself isn't Tensor
        
    Returns
    -------
    array : numpy array in the same size as 'tensorlist' on CPU 

    '''
    
    array = []
    
    for i in range(len(tensorlist)):
        item = tensorlist[i].cpu().numpy()
        array.append(item)
        
    array = np.asarray(array)
    return array




def DrawPointCloud(stone):
    '''
    This function take Point Cloud as numpy 2D matrix (N,3) and transform it
    To point cloud object in Open3D library and draw it

    Parameters
    ----------
    stone : 2D numpy matrix, shape (N,3) float64
        Point cloud in numpy array

    Returns
    -------
    pcd : Point cloud object

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(stone)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    pcd.paint_uniform_color((0,1,0))

    o3d.visualization.draw_geometries([pcd])
    
    return pcd


def DrawVoxelGrid(stone,colors,voxel_size=0.5):
    '''
    his function take Point Cloud as numpy 2D matrix (N,3) and transform it
    To VoxelGrid object in Open3D library and draw it

    Parameters
    ----------
    stone : 2D numpy matrix, shape (N,3) float64
        Point cloud in numpy array
    voxel_size : Size of drawn oxel, float64

    Returns
    -------
    voxelgrid : VoxelGrid object

    '''
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(stone)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    voxelgrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd,voxel_size)
    
    o3d.visualization.draw_geometries([voxelgrid])
    
    return voxelgrid


def GetProjection(pcd,plane_eq,Interval_size=0.01,axis='y',Plot=False):
    '''
    This function calculate the projection of the stone in the Y axis
    In order to detect the grooves. possible for x-axis to.

    Parameters
    ----------
    pcd : 2D numpy matrix, shape (N,3) float64
        Point cloud, without rotation of the plain
    plane_eq: list of (1,4) float64 coefficient
        The main plain equation, to correct the values of the height according to 
        location
    Interval_size: Float64
        The interval size of the axis_intervals array, the smaller, the more precise
        the approxiamtion, but contain less points
    axis : string, optional
        Indicating which axis you want to project, X-axis or Y-axis. The default is 'y'.
    Plot : Boolean
        if True, plot projection, if False, don't plot
    Returns
    -------
    Projection : 1D numpy array, float64
        The maximum height for every interval in Y-projection.
    axis_intervals : 1D numpy array, float64
        A structered 1D grid for Y-axis

    '''
    # Pick the desired axis to calculate the projection
    if axis == 'y':
        pcd_axis = torch.from_numpy(pcd[:,1]).to(device)
        plain_coeff = plane_eq[1]
    elif axis == 'x': 
        pcd_axis = torch.from_numpy(pcd[:,0]).to(device)
        plain_coeff = plane_eq[0] 
    else:
        return None
    
    # calculate axis intervals with intervals of Interval_size
    axis_intervals = torch.arange(torch.min(pcd_axis),torch.max(pcd_axis), Interval_size)
    Projection = torch.zeros(len(axis_intervals) - 1)
    pcd = torch.from_numpy(pcd).to(device)
    
    # Calculate Projection
    for row in tqdm(range(len(axis_intervals) - 1)):
        # find the points in the point cloud, beloging to every interval in axis_intervals
        ind = torch.where(torch.logical_and(pcd_axis >= axis_intervals[row],
                                            pcd_axis < axis_intervals[row+1]))[0]
        # for every interval, take the maximum height (from the points belogons to that inteval)
        # in addition subtract the height of the plane according to main plane equation
        # so the projection will "see" plane parrallel to the XY plane
        Projection[row] = torch.max(pcd[ind,2]) - plain_coeff * axis_intervals[row]
    
    
    # Convert back to Numpy array and CPU
    Projection = Projection.cpu().numpy()
    axis_intervals = axis_intervals.cpu().numpy()
    
    # Plot the projection
    if Plot:
        plt.figure()
        plt.plot(axis_intervals[:-1],Projection)
        plt.title('Max height in %s axis' % axis); plt.grid()
    
    return axis_intervals,Projection
    
    
def CutEdges(groove,plane_eq,quantileLow,quantileHigh,axis='x',plot=False):
    '''
    This function remove the edge of the stone 

    Parameters
    ----------
    groove : 2D numpy matrix, shape (N,3) float64
        Point cloud, represent only one groove.
    Main_plane_eq : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0
    quantileLow : float64 between 0 to 1
        DESCRIPTION.
    quantileHigh : float64 between 0 to 1, higher than quantileLow
        DESCRIPTION.
    axis : string, optional
        Indicating which axis you want to project, X-axis or Y-axis. The default is 'x'.
    plot: Boolean, optional
        if True, plot projection, if False, don't plot

    Returns
    -------
    groove_new : 2D numpy matrix, shape (M,3) float64, where M <= N
        Point cloud, represent only one groove after the edges where cutted

    '''
    # Pick the desired axis to calculate the projection
    if axis == 'y':
        groove_axis = groove[:,1]
    elif axis == 'x': 
        groove_axis = groove[:,0]
    else:
        return None
    
    # Calculate extreme quantile 
    Q1_precent = np.quantile(groove_axis,quantileLow)
    Q4_precent = np.quantile(groove_axis,quantileHigh)
    
    # remove all points with higher than Q4_precent in y-axis and lower than Q1_precent
    ind = np.where(np.logical_and(groove_axis > Q1_precent,groove_axis < Q4_precent))[0]
    groove_new = groove[ind,:]
    
    if plot:
        axis_intervals,projection = GetProjection(groove,plane_eq,1,axis,False)
        axis_intervals_new,projection_new = GetProjection(groove_new,plane_eq,1,axis,False)
        
        plt.figure()
        plt.plot(axis_intervals[1:],projection)
        plt.plot(axis_intervals_new[1:],projection_new)
        plt.grid(); plt.legend(['Original projection','Projection after cutting edge'])
    
    return groove_new



def DivideGroovesProjection(pcd,Projection,axis_intervals,quantile,num_margin_intervals,axis):
    '''
    This function perform segemntation of the point cloud, into different 
    Segements, every segment contain one groove. this calculation is vased on
    Y-axis projection of the point cloud, so you need to call GetProjection function
    To get max_height_y & axis_intervals. 
    Currently - this method don't work, beacuse the side planes aren't parrallel 
    To XZ or YZ plane, so part of the groove is outside the segemntation

    Parameters
    ----------
    pcd : 2D numpy matrix, shape (N,3) float64
        Point cloud, without rotation of the plain.
    Projection : 1D numpy array, float64
        The maximum height for every interval in Y-axis projection.
    axis_intervals : 1D numpy array, float64
        A structered 1D (constant interval) for Y-axis
    quantile : float64, between 0 to 1 
        which quantile to use as a threshold
    num_margin_intervals : integer
        The filtered projection with threshold give us position of the begining of the
        groove, but since the groove isn't orthogonal to the edges, we need to take
        more big margins, how many intervals to take as margin in Y-axis.
    
    Returns
    -------
    grooves : list of 2D numpy array, every element in the list is (N,3) float64
        list of grooves, every element contain a point cloud of a single groove

    '''
    # Filter the Projection array using LPF butter
    # ignore first and last 100 intervals, it's the sides of the stone
    b, a = butter(N=7, Wn=0.2)
    Projection_filtered = filtfilt(b, a,Projection[100:-200])
    #Projection_filtered = Projection
    
    # Find threshold based on quatile
    threshold = np.quantile(Projection_filtered,quantile)
    
    
    # Find positions where the groove starts according to threshold
    # where the projection in (i-1) was above the threshold and in (i) below the threshold
    gt = Projection_filtered[:-1] > threshold
    lt = Projection_filtered[1:] < threshold
    # move the start index num_margin_intervals to the left to take margins
    groove_start = np.where(gt * lt)[0] - num_margin_intervals
    # we found the indexs, now take the y-axis values from projection
    groove_start_y = np.take(Projection_filtered,groove_start)
    
    # Find positions where the groove ends according to the threshold 
    # where the projection in (i-1) was below the threshold and in (i) above the threshold
    gt = Projection_filtered[:-1] < threshold
    lt = Projection_filtered[1:] > threshold
    # move the end index num_margin_intervals to the right to take margins
    groove_end = np.where(gt * lt)[0] + num_margin_intervals
    # we found the indexs, now take the y-axis values from projection
    groove_end_y = np.take(Projection_filtered,groove_end)


    # Plot Groove Segmentation process
    plt.figure('Grooves Segmentation')
    plt.plot(Projection_filtered);  
    plt.scatter(groove_start,groove_start_y,marker='o',color='r')
    plt.scatter(groove_end,groove_end_y,marker='o',color='g')
    #plt.plot(np.ones(np.shape(Projection_filtered))*threshold,color='c');
    plt.legend(['Filtered Projection_filtered','Groove starts','Groove ends'])
    plt.grid()
    
    # Fix indexs, we igonred the first 100 intervals at the segmentation process 
    groove_start = np.int64(groove_start + 100)
    groove_end = np.int64(groove_end + 100)
    
    # Now we have found the indexs where the grooves started and ended, we need to 
    # take the Y-axis values for every groove before partition
    groove_start = np.take(axis_intervals,groove_start)
    groove_end = np.take(axis_intervals,groove_end)
    
    # Choose axis to work on, x or y
    if axis == 'y':
        pcd_axis = pcd[:,1];
    elif axis == 'x': 
        pcd_axis = pcd[:,0]
    else:
        return None
    
    # According to indexs found in axis_intervals, take point cloud and assign to 
    # the different grooves segments
    grooves = []
    for groove in range(len(groove_start)):
        # which indexs from the point cloud?
        ind = np.where(np.logical_and(pcd_axis >= groove_start[groove], 
                                      pcd_axis < groove_end[groove]))
        # take those points
        pcd_groove = pcd[ind,:]
        # add to a list
        grooves.append(pcd_groove)

    
    return grooves



def PlotSegmentedGrooves(grooves):
    '''
    This function plot all grooves in one open3d pointcloud, every groove in different color

    Parameters
    ----------
    grooves : list of 2D numpy array, every element in the list is (N,3) float64
        list of grooves, every element contain a point cloud of a single groove

    Returns
    -------
    pcd_grooves : 
        Point Cloud of all grooves
    colors : 
    '''
    number_grooves = len(grooves)
    pcd_grooves = o3d.geometry.PointCloud()
    colors = np.random.rand(number_grooves,3)
    
    for groove in range(number_grooves):
        groove_pcd = grooves[groove][0]
        color =  np.expand_dims(colors[groove,:],axis=0)
        color = np.repeat(color,groove_pcd.shape[0],axis=0)
        
        # add to numpy array of all point cloud
        if groove == 0:
            grooves_pcd = groove_pcd
            colors_pcd = color
        else:
            grooves_pcd = np.concatenate((grooves_pcd,groove_pcd),axis=0)
            colors_pcd = np.concatenate((colors_pcd,color),axis=0)
            
            
    
    pcd_grooves.points = o3d.utility.Vector3dVector(grooves_pcd)
    pcd_grooves.colors = o3d.utility.Vector3dVector(colors_pcd)
        
    pcd_grooves.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    o3d.visualization.draw_geometries([pcd_grooves])
    
    
    return pcd_grooves,colors
    
def PointCloud2Grid(pcd,voxel_size):
    '''
    This function takes 

    Parameters
    ----------
    pcd : 2D numpy matrix, shape (N,3) float64
        Point cloud
    voxel_size : TYPE
        DESCRIPTION.

    Returns
    -------
    pcd_grid : 2D numpy matrix, shape (M,3) float64, where M <= N
        Point cloud, where the X,Y space is structured on a grid

    '''
    # Create Point Cloud and transform it to VoxelGrid
    pointcloud = o3d.geometry.PointCloud()
    pointcloud.points = o3d.utility.Vector3dVector(pcd)
    VoxelGrid = o3d.geometry.VoxelGrid.create_from_point_cloud(pointcloud,voxel_size)

    # Get all the voxels and transform it to 2D numpy array
    Voxels = VoxelGrid.get_voxels()
    pcd_grid = np.zeros((len(Voxels),3))
    
    for i,voxel in enumerate(Voxels):
        pcd_grid[i,:] = voxel.grid_index

    
    # Check if the Grid is full without holes
    # sort the x and y values, the calculate difference between consecutive values
    x_grid_diff = np.diff(np.sort(np.unique(pcd_grid[:,0])))
    y_grid_diff = np.diff(np.sort(np.unique(pcd_grid[:,1])))
    
    # find where the difference between consecutive values is differenet than 1?
    ind_x = np.where(x_grid_diff != 1)[0]
    ind_y = np.where(y_grid_diff != 1)[0]
    
    # if not, the Grid if full
    if ind_x.size == 0 and ind_y.size == 0:
        print('Grid is Full')
        print('Original Point Cloud Size: %d' % pcd.shape[0])
        print('New Point Cloud Size: %d' % pcd_grid.shape[0])
        precent = pcd_grid.shape[0] / pcd.shape[0] * 100
        print('The number of data point was reduced to: %.2f precent' % precent)
    else:
        print('Grid is not full. fill the holes')
    
    
    return pcd_grid

#%% Plane equations
# 1. Parrallel to z-plane, Ax+By+Cz+D=0 becomes to Ax+By+D=0 (side planes)
# beacuse (A,B,0)*(0,0,1) = 0 = [ like (A,B,C)*k vector] 
# 2. Parrallel to xy plane, Ax+By+Cz+D=0 becomes to Cz+D=0 (Main plane)

# how do I know that the main plane is orthogonal to the side planes?
# (A_main,B_main,C_main) * (A_side,B_side,C_side) = 0
    
    
    
    
# Utils functions
import numpy as np
import torch
import open3d as o3d
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter,filtfilt


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')



def SortPointCloud(stone):
    '''
    This function sort the values of the Point cloud, in lexigraphic otder
    The function first sort the point cloud by the y-values, and then sort
    The point cloud by the x-values. 

    Parameters
    ----------
    stone : 2D numpy matrix, shape (N,3) float64
        Point Cloud unsorted

    Returns
    -------
    stone : 2D numpy matrix, shape (N,3) float64
        Point Cloud sorted in lexigraphic order.

    '''
    # sort PCD by y-value
    order_y = np.argsort(stone[:,1])
    stone = stone[order_y,:]
    
    # now, we can sort by x coordinate, and the array will be sorted lexiographic
    order_x = np.argsort(stone[:,0])
    stone = stone[order_x,:]
    
    return stone


def CalculateIntegral(stone,plane_eq):
    '''
    This function calculate the ablation volume of the groove. 
    The groove is an ordered point cloud, and the plane equation blocks
    The groove from above. The edge of the groove are merging with the plain equation
    The plain the and the stone are parrallel to X-Y plane in the coordinate system

    Parameters
    ----------
    stone : 2D numpy matrix, shape (N,3) float64
        Ordered Point cloud, with interval of delta_x, delta_y.
        more than one z-value is possible
    plane_eq : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0.

    Returns
    -------
    volume : (1,1) float64 
        Ablation volume in units of [mm^3]

    '''
    
    # find delta_x and delta_y
    delta_x, delta_y = stone[1,0,0] - stone[0,0,0], stone[0,1,0] - stone[0,0,0]
    
    # take sample grid and calculate plane 3D grid
    # coeff_x*x + coeff_y*y + coeff_z*z + const = 0 to:
    # z = (coeff_x*x + coeff_y*y + const) / (-coeff_z)
    plane = np.unique(stone[:,:2],axis=0) # take (x,y) grid withput reptitions
    z = (plane_eq[0] * plane[0] + plane_eq[1] * plane[1] + plane_eq[3])/(-plane_eq[2]) # calculate z
    plane = np.concatenate((plane,z),axis=1) # concate to one matrix
    
    # (delta_x / 2) * (delta_y / 2) * z = Infinitesimal volume
    area = (delta_x / 2) * (delta_y / 2) # dx*dy
    diff = plane[:,2] - stone[:,2] # vector of different 
    diff = diff.clip(min=0) # zeros values to zero
    volume = np.sum(area * diff) # dx*dy*dz sum
    
    
    return volume



def extractAngles(plane):
    '''
    This function calculate the angels to rotate the point cloud
    based in the plane coeeficents, so the plane will be parrallel to X-Y plane

    Parameters
    ----------
    plane : list of (1,4) float64 coefficient
        Plain Equation: A*x + B*y + C*z + D=0..

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
    


def GetProjection(pcd,plane_eq,Interval_size=0.01,axis='y',Plot=False):
    '''
    This function calculate the projection of the stone in the Y axis
    In order to detect the grooves. possible for x-axis to.

    Parameters
    ----------
    pcd : 2D numpy matrix, shape (N,3) float64
        Point cloud, without rotation of the plain
    plane_eq: 
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
    
    # calculate axis intervals with intervals of 0.001
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
    groove : TYPE
        DESCRIPTION.
    Main_plane_eq : TYPE
        DESCRIPTION.
    quantileLow : TYPE
        DESCRIPTION.
    quantileHigh : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is 'x'.
    plot: 
        

    Returns
    -------
    groove_new : TYPE
        DESCRIPTION.

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
        axis_intervals,projection = GetProjection(groove,plane_eq,0.02,axis,False)
        axis_intervals_new,projection_new = GetProjection(groove_new,plane_eq,0.02,axis,False)
        
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
    b, a = butter(N=7, Wn=0.08)
    Projection_filtered = filtfilt(b, a,Projection[100:-200])
    
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
    plt.plot(Projection[100:]); 
    plt.scatter(groove_start,groove_start_y,marker='o',color='r')
    plt.scatter(groove_end,groove_end_y,marker='o',color='g')
    #plt.plot(np.ones(np.shape(Projection_filtered))*threshold,color='c');
    plt.legend(['Filtered Projection_filtered','Projection','Groove starts','Groove ends'])
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
    pcd_grooves

    '''
    number_grooves = len(grooves)
    pcd_grooves = o3d.geometry.PointCloud()
    colors = np.random.rand(3,number_grooves)
    
    for groove in range(number_grooves):
        groove_pcd = grooves[groove][0]
        color =  np.expand_dims(colors[:,groove],axis=0)
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
    
    
    return pcd_grooves
    
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
    else:
        print('Grid is not full. fill the holes')
    
    
    return pcd_grid

#%% Plane equations
# 1. Parrallel to z-plane, Ax+By+Cz+D=0 becomes to Ax+By+D=0 (side planes)
# beacuse (A,B,0)*(0,0,1) = 0 = [ like (A,B,C)*k vector] 
# 2. Parrallel to xy plane, Ax+By+Cz+D=0 becomes to Cz+D=0 (Main plane)

# how do I know that the main plane is orthogonal to the side planes?
# (A_main,B_main,C_main) * (A_side,B_side,C_side) = 0
    
    
    
    
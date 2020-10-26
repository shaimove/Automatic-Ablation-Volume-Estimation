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



def Rotate(stone,angels):
    '''
    This function rotate a point cloud so the main plain, is parrallel to 
    X-Y plane. the angels: (theta_x,theta_y,theta_z) = (roll,pitch,yaw) 
    is used in homogenous coordinate to rotate the point cloud easily.
    for more info: see pdf file with explanations about affine transformation

    Parameters
    ----------
    stone : 2D numpy matrix, shape (N,3) float64
        The point cloud
    angels : list of (1,3) float64 
        angels in [Rad] units

    Returns
    -------
    stone_new : 2D numpy matrix, shape (N,3) float64
        The point cloud rotated

    '''
    # if I want to add offset to the plane, I need to add const value 
    # to the last zero in every of the first 3 rows. without the changing the 1!
    # but then we need to divide the (x,y,z) value by the last value to trandform to 
    # in non-homogenous corrdinate, only in the last matrix (after matmul)
    
    
    # Define R_x,R_y,R_z
    theta_x,theta_y,theta_z = angels
    
    R_x = [[1,      0,                  0,     0],
           [0,np.cos(theta_x),-np.sin(theta_x),0],
           [0,np.sin(theta_x), np.cos(theta_x),0],
           [0,      0,                  0,     1]]
    
    R_y = [[np.cos(theta_y),0,np.sin(theta_y), 0],
           [0,      1,                  0,     0],
           [-np.sin(theta_y),0,np.cos(theta_y),0],
           [0,      0,                  0,     1]]
    
    R_z = [[np.cos(theta_z),-np.sin(theta_z),0, 0],
           [np.sin(theta_z), np.cos(theta_z),0, 0],
           [0,      0,                  1,      0],
           [0,      0,                  0,      1]]
    
    # Add 1's to stone matrix to (4*n)
    ones = np.ones((stone.shape[0],1)) # create 1's vector
    stone = np.concatenate((stone,ones),axis=1) # concatenate
    stone = np.transpose(stone)
    
    # stone_new = N*4, Rotation = 4*4, stone 4*N;
    # so 4*N = stone * Rotation - and then transpose
    Rotation = np.matmul(np.matmul(R_x,R_y),R_z)
    stone_new = np.matmul(Rotation,stone)
    stone_new = np.transpose(stone_new)
    
    # remove 1's
    stone_new = stone_new[:,:3]
    
    return stone_new


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



def CreateMesh(point_cloud):
    '''
    This function take point cloud without ordered grid, calculate average 
    Interval between points (assuming the interval is constant in both axises)
    And return the grid, without assiging z-values to the grid. 
    The number of points in the grid is roughly the same as the original point
    Cloud. the comment part is a loop to assign z-values to the grid using nearest neighbor
    on GPU computation. currently the computation time is to long

    Parameters
    ----------
    point_cloud : 2D numpy matrix, shape (N,3) float64
        Point Cloud without ordred mesh

    Returns
    -------
    Grid : 2D numpy matrix, shape (N,2) float64
        Ordered Grid without z-value

    '''
    # Calculate minimum and maximum x and y values
    min_x = np.min(point_cloud[:,0])
    max_x = np.max(point_cloud[:,0]) 
    width_x = max_x - min_x
    min_y = np.min(point_cloud[:,1])
    max_y = np.max(point_cloud[:,1])
    width_y = max_y - min_y
        
    interval = point_cloud.shape[0] / (width_x * width_y)
    interval_square = interval ** 0.5
        
    points_in_y = int(width_y * interval_square)
    points_in_x = int(width_x * interval_square)
        
    X = np.linspace(min_x,max_x,points_in_x)
    Y = np.linspace(min_y,max_y,points_in_y)
    
    X_grid = np.expand_dims(np.tile(X,len(Y)),axis=1)
    Y_grid = np.expand_dims(np.repeat(Y,len(X)),axis=1)
    Grid = np.concatenate((X_grid,Y_grid),axis=1)

    return Grid
    
    #point_cloud = torch.from_numpy(point_cloud).to(device)
    #Grid = torch.from_numpy(Grid).to(device)
    #pcd_close_Grid = torch.zeros(point_cloud.shape).to(device)
    
    #for point_in_pcd in tqdm(range(point_cloud.shape[0])):
    #    x_pcd,y_pcd = point_cloud[point_in_pcd,0],point_cloud[point_in_pcd,1]
    #    dist = torch.sqrt(torch.square(x_pcd-Grid[:,0]) + torch.square(y_pcd-Grid[:,1]))
    #    ind = torch.argmin(dist) # minimum point in Grid that is close to point cloud
    #    pcd_close_Grid[point_in_pcd,:2] = Grid[ind,:]
    
    #pcd_close_Grid[:,2] = point_cloud[:,2]
    
    #return pcd_close_Grid



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
    


def GetProjection(pcd,plane_eq,axis='y'):
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
    axis : string, optional
        Indicating which axis you want to project, X-axis or Y-axis. The default is 'y'.

    Returns
    -------
    Projection : 1D numpy array, float64
        The maximum height for every interval in Y-projection.
    axis_intervals : 1D numpy array, float64
        A structered 1D grid for Y-axis

    '''
    # Pich the desired axis to calculate the projection
    if axis == 'y':
        pcd_axis = torch.from_numpy(pcd[:,1]).to(device)
        plain_coeff = plane_eq[1]
    elif axis == 'x': 
        pcd_axis = torch.from_numpy(pcd[:,0]).to(device)
        plain_coeff = plane_eq[0] 
    else:
        return None
    
    # calculate axis intervals with intervals of 0.001
    axis_intervals = torch.arange(torch.min(pcd_axis),torch.max(pcd_axis), 0.01)
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
    
        
    # Plot the projection
    Max_height = Projection.cpu().numpy()
    axis_intervals = axis_intervals.cpu().numpy()
    plt.figure()
    plt.plot(axis_intervals[:-1],Max_height)
    plt.title('Max height in %s axis' % axis); plt.grid()
    
    return axis_intervals,Projection
    
    

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
    

#%% Plane equations
# 1. Parrallel to z-plane, Ax+By+Cz+D=0 becomes to Ax+By+D=0 (side planes)
# beacuse (A,B,0)*(0,0,1) = 0 = [ like (A,B,C)*k vector] 
# 2. Parrallel to xy plane, Ax+By+Cz+D=0 becomes to Cz+D=0 (Main plane)

# how do I know that the main plane is orthogonal to the side planes?
# (A_main,B_main,C_main) * (A_side,B_side,C_side) = 0
    
    
    
    
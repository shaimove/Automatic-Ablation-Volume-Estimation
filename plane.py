# plane.py
import numpy as np
import random
import torch
from utils import TensorList2Array
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#%% GPU version
          
class Plane:

    def __init__(self):
        self.inliers = []
        self.equation = []
    
    def fit(self,pts,thresh=0.05, min_Points=100, maxIteration=1000):
        """ 
		Find the best equation for a plane.
		:param pts: 3D point cloud as a `np.array (N,3)`.
		:param thresh: Threshold distance from the plane which is considered inlier.
		:param maxIteration: Number of maximum iteration which RANSAC will loop over.
		:returns:
		- `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
		- `self.inliers`: points from the dataset considered inliers
		---
		"""
        n_points = pts.shape[0]
        print(n_points)
        best_eq = []
        best_inliers = []
        
        pts = torch.from_numpy(pts).to(device)
        
        for it in range(maxIteration):
            # Samples 3 random points
            id_samples = random.sample(range(1,n_points-1),3)
            pt_samples = pts[id_samples].to(device)
            
            # We have to find the plane equation described by those 3 points
			# We find first 2 vectors that are part of this plane
			# A = pt2 - pt1
			# B = pt3 - pt1
            vecA = pt_samples[1,:] - pt_samples[0,:]
            vecB = pt_samples[2,:] - pt_samples[0,:]
            
            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = torch.cross(vecA,vecB).to(device)
            
            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = - D
			# We have to use a point to find D
            vecC = torch.true_divide(vecC, torch.norm(vecC)) # to unit vector
            D = - torch.sum(torch.matmul(vecC,pt_samples[1,:]))
            plane_eq  = [vecC[0],vecC[1],vecC[2],D]
            
            # Distance from a point to a plane 
			# https://mathworld.wolfram.com/Point-PlaneDistance.html
            denominator = torch.square(plane_eq[0]**2 + plane_eq [1]**2 + plane_eq[2]**2)
            Numerator = (plane_eq[0]*pts[:,0] + plane_eq[1]*pts[:,1] + plane_eq[2]*pts[:,2] + plane_eq[3])
            dist_pt = torch.true_divide(Numerator,denominator).to(device)
            
            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = [] # list of inliers ids
            pt_id_inliers = torch.where(torch.abs(dist_pt) <= thresh)[0]
            
            if (len(pt_id_inliers) > len(best_inliers)):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            
            self.inliers = best_inliers
            self.equation = best_eq
        
        # print equation
        [a, b, c, d] = self.equation
        print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.2f} = 0")
        
        # transform to numpy
        self.equation = TensorList2Array(self.equation)
        self.inliers = self.inliers.cpu().numpy()
        
        return self.equation, self.inliers
        










#%% CPU version
class PlaneCPU:
    
    def __init__(self):
        self.inliers = []
        self.equation = []
        
    def fit(self,pts,thresh=0.05, min_Points=100, maxIteration=1000):
        """ 
		Find the best equation for a plane.
		:param pts: 3D point cloud as a `np.array (N,3)`.
		:param thresh: Threshold distance from the plane which is considered inlier.
		:param maxIteration: Number of maximum iteration which RANSAC will loop over.
		:returns:
		- `self.equation`:  Parameters of the plane using Ax+By+Cy+D `np.array (1, 4)`
		- `self.inliers`: points from the dataset considered inliers
		---
		"""
        n_points = pts.shape[0]
        print(n_points)
        best_eq = []
        best_inliers = []
        
        for it in range(maxIteration):
            # Samples 3 random points
            id_samples = random.sample(range(1,n_points-1),3)
            pt_samples = pts[id_samples]
            
            # We have to find the plane equation described by those 3 points
			# We find first 2 vectors that are part of this plane
			# A = pt2 - pt1
			# B = pt3 - pt1
            vecA = pt_samples[1,:] - pt_samples[0,:]
            vecB = pt_samples[2,:] - pt_samples[0,:]
            
            # Now we compute the cross product of vecA and vecB to get vecC which is normal to the plane
            vecC = np.cross(vecA,vecB)
            
            # The plane equation will be vecC[0]*x + vecC[1]*y + vecC[0]*z = - D
			# We have to use a point to find D
            vecC = vecC / np.linalg.norm(vecC) # to unit vector
            D = - np.sum(np.multiply(vecC,pt_samples[1,:]))
            plane_eq  = [vecC[0],vecC[1],vecC[2],D]
            
            # Distance from a point to a plane 
			# https://mathworld.wolfram.com/Point-PlaneDistance.html
            denominator = np.sqrt(plane_eq[0]**2 + plane_eq [1]**2 + plane_eq[2]**2)
            Numerator = (plane_eq[0]*pts[:,0] + plane_eq[1]*pts[:,1] + plane_eq[2]*pts[:,2] + plane_eq[3])
            dist_pt = Numerator / denominator
            
            # Select indexes where distance is biggers than the threshold
            pt_id_inliers = [] # list of inliers ids
            pt_id_inliers = np.where(np.abs(dist_pt) <= thresh)[0]
            
            if (len(pt_id_inliers) > len(best_inliers)):
                best_eq = plane_eq
                best_inliers = pt_id_inliers
            
            self.inliers = best_inliers
            self.equation = best_eq
        
        # print equations
        [a, b, c, d] = self.equation
        print(f"Plane equation: {a:.3f}x + {b:.3f}y + {c:.3f}z + {d:.2f} = 0")
        
        
        
        return self.equation, self.inliers


## Automatic Ablation Volume Estimation
----
 The following repo includes python code, that enable to read STL files, results from 3D light scanner of stones with grooves. 
 The grooves were created using laser cutting technology, and the goal is to measure the grooves volume automaticly. 
 
## Requirements: Numpy, Open3D, and Pytorch with Cuda

Numpy:
```sh
conda install numpy
```
Oped3d:
```sh
conda install -c open3d-admin open3d
```
PyTorch with GPU:
```sh
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```

## Algorithm

Algorithm includes the following steps (modules or cells in Main.py):
 1. Read STL file (mesh traingulation) and convert it into Point Cloud
 2. Optional  Rotation of the point cloud. In our dataset, the planes were alligened with the coordinate system, but if your stone planes aren't allign with the coordinate system,     you should allign them, to make the automatic anaylsis possible. your first need to extract the angels from the detected main plane and then rotate the whole point cloud.  
 3. Convert the Point cloud into Voxel Grid, and calculate the calibration of Voxel Grid.
 4. Calculate the main plane equation (Ax+By+Cz+D=0). this equation will help us to correct the point cloud z-position. we use RANSAC method with 1000 iterations. 
 5. Idetify the grooves and extract the points form the point cloud that contain each groove. we use projection (You can use X-axis or Y-axis projection), and using the maximum       height in each interval projection, we can find the begining and end of the groove. we divide the point cloud and plot the result.
 6. Cutting the edges of the projection, beacuse we want to calculate the volume of the stone, using only the main plane, and we need to delete the rest of the planes. 
 7. Calculate the volume of the groove using descrite integral, 
 8. Plot grooves and volumes
 
 ## Example of Results and steps
 
 Stone 3D scan, presented in Point Cloud (mesh traingulation)
 ![Image 1](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Stone%20-%20PointCloud.png)
 
 when we transform the format to Voxel Grid, the stone looks as:
  ![Image 2](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Stone%20-%20Voxel%20Grid.png)
  
 The main plane:
  ![Image 3](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Main%20plane.png)
 
 Then we look on the Y-axis projection to try and find the grooves:
  ![Image 4](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Groove%20Detection.png)
  
 And after some processing we find the beginning and the end of the grooves
  ![Image 5](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Groove%20Detection%202.png)
  
 We remove the unnesessary planes, using the projection at the orthogonal axis:
  ![Image 6](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Remove%20Edges.png)
 
 And we can easily can segment the groove, in both Point cloud:
  ![Image 7](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Segmented%20Grooves%20Point%20Cloud.png)
  
 And in Voxel Grid:
  ![Image 8](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Segmented%20Grooves%20Voxel%20Grid.png)
 
 And we get the following segmentation on the stone, that enable us to easily compute the volumes:
  ![Image 9](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Result%201.png)
  
  ![Image 10](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Result%202.png)
  
  ![Image 11](https://github.com/shaimove/Automatic-Ablation-Volume-Estimation/blob/master/Imgaes/Result%203.png)
  

## License

Private Property of Lumenis LTD. 

## Contact
Developed by Sharon Haimov, Research Engineer at Lumenis.

Email: sharon.haimov@lumenis.com or shaimove@gmail.com

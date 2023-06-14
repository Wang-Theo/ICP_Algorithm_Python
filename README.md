# ICP_Algorithm_Python
This is an SVD-based ICP algorithm to calculate transformation from point cloud
1 to point cloud 2.

**Pointcloud data** : two bunny pointclouds in different color

**Two questions** need to be solved:
1. correspondence is known  
2. correspondence is unknown  

## 1. Prerequisites
1.1 Python 3.8

1.2 OpenCV 4.2

1.3 Numpy

## 2. Principle & Execution
**2.1 ICP Algorithm for Matched Point Cloud**  
- Step 1: calculate centroid
- Step 2: de-centroid of points1 and points2
- Step 3: compute H, which is sum of p1i'*p2i'^T
- Step 4: SVD of H (can use 3rd-part lib), solve R and t
- Step 5: combine R and t into transformation matrix T
- Step 6: calculate transformed point cloud 1 based on T solved above

```
cd ~/ICP_Algorithm_Python/ICP_Matched_PC
sudo chmod +x main.py
./main.py
```
Pointclouds registration result:

<img src="https://github.com/Wang-Theo/ICP_Algorithm_Python/blob/devel/reault_image/matched_pc.png" width="500" alt="matched_pc"/>

**2.2 ICP Algorithm for Unmatched Point Cloud**
- Aim: For all point in points_1, find nearest point in points_2, and generate points_2_nearest
- Step 1: solve icp (Step 1-5 in 2.1)
- Step 2: update accumulated T
- Step 3: update points_1
- **Iterated untill the time set before**

```
cd ~/ICP_Algorithm_Python/ICP_Unmatched_PC
sudo chmod +x main.py
./main.py
```

Pointclouds registration result (30 iterations):

<img src="https://github.com/Wang-Theo/ICP_Algorithm_Python/blob/devel/reault_image/unmatched_pc_iter30.png" width="500" alt="matched_pc"/>

Pointclouds registration result (50 iterations):

<img src="https://github.com/Wang-Theo/ICP_Algorithm_Python/blob/devel/reault_image/unmatched_pc_iter50.png" width="500" alt="matched_pc"/>

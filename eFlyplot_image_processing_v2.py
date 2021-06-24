import numpy as np

'''
function platn
1. rotate, translation, basic functions
2. warping 
3. deal with different neuronal type transition (nrrd to nifti etc.)
4. points to voxels and voxels to points
5.

'''

def rotation(X, angle, axis):
    X_new = []
    if axis.find('xz') != -1:
        for x, y, z in X:
            x_prime = math.cos(angle) * x - math.sin(angle) * z
            y_prime = y
            z_prime = math.sin(angle) * x + math.cos(angle) * z
            X_new.append([x_prime, y_prime, z_prime])
    elif axis.find('xy') != -1:
        for x, y, z in X:
            x_prime = math.cos(angle) * x - math.sin(angle) * y
            y_prime = math.sin(angle) * x + math.cos(angle) * y
            z_prime = z
            X_new.append([x_prime, y_prime, z_prime])
    else:
        for x, y, z in X:
            x_prime = x
            y_prime = math.cos(angle) * y - math.sin(angle) * z
            z_prime = math.sin(angle) * y + math.cos(angle) * z
            X_new.append([x_prime, y_prime, z_prime])
    return np.array(X_new)

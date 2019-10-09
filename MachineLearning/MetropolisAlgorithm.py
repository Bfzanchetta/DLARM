import numpy as np
import matplotlib.pyplot as plt
import time
from tkinter import *

start_time = time.time()

num_samples = 100000
prob_density = 0
mean = np.array([0,0])
cov = np.array([[1,0.7],[0.7,1]])
cov1 = np.matrix(cov)
mean1 = np.matrix(mean)

x_list,y_list= [],[]
accepted_samples_count = 0

normalizer = np.sqrt( ((2*np.pi)**2)*np.linalg.det(cov))

x_initial, y_initial = 0,0
x1,y1 = x_initial, y_initial

for i in range(num_samples):
	mean_trans = np.array([x1,y1])
	cov_trans = np.array([[0.2,0],[0,0.2]])
	x2,y2 = np.random.multivariate_normal(mean_trans,cov_trans).T
	X = np.array([x2,y2])
	X2 = np.array(X)
	X1 = np.array(mean_trans)
	
	mahalnobis_dist2 = (X2 - mean1)*np.linalg.inv(cov)*(X2 - mean1).T
	prob_density2 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist2)
	mahalnobis_dist1 = (X1 - mean1)*np.linalg.inv(cov)*(X1 - mean1).T
	prob_density1 = (1/float(normalizer))*np.exp(-0.5*mahalnobis_dist1)
	
	acceptance_ratio = prob_density2[0,0] / float(prob_density1[0,0])
	
	if(acceptance_ratio >= 1) | ((acceptance_ratio < 1) and (acceptance_ratio >= np.random.uniform(0,1)) ):
		x_list.append(x2)
		y_list.append(y2)
		x1 = x2
		y1 = y2
		accepted_samples_count += 1

end_time = time.time()

print('Time taken to sample ' + str(accepted_samples_count) + ' points ==> ' + str(end_time - start_time) + ' seconds.')
print('Acceptance ratio ===> ' , accepted_samples_count/float(100000))

plt.xlabel('X')
plt.ylabel('Y')
plt.scatter(x_list, y_list, color='black')
print("Mean of the Sampled Points")
print(np.mean(x_list), np.mean(y_list))
print("Covariance matrix of the Sampled Points")
print(np.cov(x_list,y_list))

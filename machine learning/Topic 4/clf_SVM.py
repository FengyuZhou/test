import json
import numpy as np
from sklearn import svm
from sklearn.utils import shuffle
import glob, os
from sklearn.model_selection import cross_val_score
import random
from multiprocessing import Pool
os.chdir("./")

x = []
y = []

for file in glob.glob("train/*.json"):
	with open(file,'r') as data_file:    
		data = json.load(data_file)

	for reg in data['train_clutter_poly']:
		# Make sure that it is not TAIL
		# 0 ROOM
	        # 1 INTERIOR_UNCLASSIFIED
	        # 2 INTERIOR_OBSTACLE			0
	        # 3 INTERIOR_CLUTTER			1
	        # 4 BG_CONNECTED_UNCLASSIFIED
	        # 5 BG_CONNECTED_OBSTACLE		0
	        # 6 BG_CONNECTED_CLUTTER		1
	        # 7 TAIL
		if reg['type']==2 or reg['type']==3 or reg['type']==5 or reg['type']==6:
			x.append([reg['area'],reg['xmin'],reg['xmax'],reg['ymin'],reg['ymax'],\
			reg['centroid_x'],reg['centroid_y'],reg['majorAxisLength'],reg['minorAxisLength'],\
	          	reg['orientation'],reg['eccentricity'],reg['solidity'],reg['is_free'],\
			int(reg['type']<4),reg['compactcness'],reg['long_line_lenth'],reg['n_long_lines'],\
			reg['n_free_pixels'],reg['n_unexplored_pixels'],reg['n_neighbors'],\
			reg['ratio_of_simplified_to_unsimplified_size'],reg['n_doors'],reg['total_door_length']])
			y.append(int(reg['type']==3 or reg['type']==6))

x, y = shuffle(x,y)

# case1 para:
#best_C_para =  8987.35323158
#best_gamma_para = 26.9860243732

# case2 para:
#best_C_para =  7963.0894983
#best_gamma_para = 21.2033350094

best_C_para =  7963.0894983
best_gamma_para = 21.2033350094

clf = svm.SVC(C = best_C_para/5.0, kernel='rbf', gamma = 0.000001*best_gamma_para) 
best_score = np.mean(cross_val_score(clf, x, y, cv=5))
n_sample = 64

def sol(n):
	if(random.random()>0.6):
		C_para = np.random.normal(best_C_para,150)
		gamma_para = np.random.normal(best_gamma_para,2)
		if gamma_para<=0:
			gamma_para = 0.1
	else:
		C_para = random.uniform(000,10000)
		gamma_para = random.uniform(0,40)

	clf = svm.SVC(C = C_para/5.0, kernel='rbf', gamma = 0.000001*gamma_para) 
	scores = cross_val_score(clf, x, y, cv=5)

	return [C_para,gamma_para,np.mean(scores)]



for it in range(5):
	pool = Pool()
	args = range(n_sample)

	results = pool.map(sol, args)
	
	for sample in results:
		if sample[2]>best_score:
			best_score = sample[2]
			best_C_para = sample[0]
			best_gamma_para = sample[1]
	print("####################")
	print(best_score)
	print(best_C_para)
	print(best_gamma_para)
	print("####################")

clf = svm.SVC(C = best_C_para/5.0, kernel='rbf', gamma = 0.000001*best_gamma_para) 

clf.fit(x,y)

x_test = []
y_test = []

for file in glob.glob("test/*.json"):
	with open(file,'r') as data_file:    
		data = json.load(data_file)

	for reg in data['train_clutter_poly']:
		if reg['type']==2 or reg['type']==3 or reg['type']==5 or reg['type']==6:
			x_test.append([reg['area'],reg['xmin'],reg['xmax'],reg['ymin'],reg['ymax'],\
			reg['centroid_x'],reg['centroid_y'],reg['majorAxisLength'],reg['minorAxisLength'],\
	          	reg['orientation'],reg['eccentricity'],reg['solidity'],reg['is_free'],\
			int(reg['type']<4),reg['compactcness'],reg['long_line_lenth'],reg['n_long_lines'],\
			reg['n_free_pixels'],reg['n_unexplored_pixels'],reg['n_neighbors'],\
			reg['ratio_of_simplified_to_unsimplified_size'],reg['n_doors'],reg['total_door_length']])
			y_test.append(int(reg['type']==3 or reg['type']==6))



y_pred =clf.predict(x_test)
print 'Test error rate:', 1-float(sum(y_pred==y_test))/len(x_test)

print(best_C_para)
print(best_gamma_para)


import json
import numpy as np
from sklearn.utils import shuffle
from sklearn import tree
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

best_score = 0
for md in range(20):
	clf = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = md+1)
	scores = cross_val_score(clf, x, y, cv=5)
	print(np.mean(scores))
	if np.mean(scores)>best_score:
		best_score = np.mean(scores)
		md_opt = md
clf = tree.DecisionTreeClassifier(criterion = 'gini',max_depth = md_opt+1)


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
print 'Test error rate: ', 1-float(sum(y_pred==y_test))/len(x_test)

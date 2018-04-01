import json
import numpy as np
from sklearn.utils import shuffle
import glob, os
from sklearn.model_selection import cross_val_score
import random
from multiprocessing import Pool
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
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


x = np.array(x)
y = np.array(y)

#x, x_test, y, y_test = train_test_split(x, y, test_size=0.2, random_state = 2)

# paramters for case 1:
#eta_opt = 8.53653307113
#md_opt = 4
#gamma_opt = 1

# case 2
#eta_opt = 14.15781879
#md_opt = 4
#gamma_opt = 1.07480991723

eta_opt = 12.0363333706
md_opt = 4
gamma_opt = 1.02650749689


clf = XGBClassifier(learning_rate = eta_opt/100.0, max_depth = md_opt, gamma = gamma_opt)
scores = cross_val_score(clf, x, y, cv=5)
best_score = np.mean(scores)

for it in range(10):
	eta_para = np.random.normal(eta_opt,2)
	gamma_para = np.random.normal(gamma_opt,0.3)
	if gamma_para<=0:
		gamma_para = 0
	clf = XGBClassifier(learning_rate = eta_para/100.0, max_depth = md_opt, gamma = gamma_para)
	scores = cross_val_score(clf, x, y, cv=5)
	if np.mean(scores)>best_score:
		best_score = np.mean(scores)
		eta_opt = eta_para
		gamma_opt = gamma_para
	print(best_score)

clf = XGBClassifier(learning_rate = eta_opt/100.0, max_depth = md_opt, gamma = gamma_opt)
clf.fit(x,y)

print(eta_opt)
print(gamma_opt)

x_test = []
y_test = []
info = []

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
			info.append([file,reg['id'],reg['type']])

y_pred =clf.predict(x_test)
print 'Optimal cross validation error rate:', 1-best_score
print 'testing error rate:', 1-float(sum(y_pred==y_test))/len(x_test)

print 'Mistakes: (format: filename, region id, region type)'

for k in range(len(info)):
	if y_pred[k] != y_test[k]:
		print(info[k])

inter_count = 0
inter_correct = 0
BG_count = 0
BG_correct = 0

inter_wall = 0
BG_wall = 0
inter_wall_correct = 0
BG_wall_correct = 0

for k in range(len(info)):
	if x_test[k][13]==1:
		inter_count = inter_count + 1
		if y_pred[k] == y_test[k]:
			inter_correct = inter_correct + 1
		if y_test[k] == 0:
			inter_wall = inter_wall + 1
			if y_pred[k] == y_test[k]:
				inter_wall_correct = inter_wall_correct + 1
		
	if x_test[k][13]==0:
		BG_count = BG_count + 1
		if y_pred[k] == y_test[k]:
			BG_correct = BG_correct + 1
		if y_test[k] == 0:
			BG_wall = BG_wall + 1
			if y_pred[k] == y_test[k]:
				BG_wall_correct = BG_wall_correct + 1

print 'There are ', inter_count, ' interior regions,'
print 'among which, ', inter_correct, ' regions are correctly classified'
print 'There are ', BG_count, ' BG connected regions,'
print 'among which, ', BG_correct, ' regions are correctly classified'

print 'detailed info:'
print 'There are ', inter_wall, ' interior walls,'
print 'among which, ', inter_wall_correct, ' regions are correctly classified'

print 'There are ', inter_count - inter_wall, ' interior clutters,'
print 'among which, ', inter_correct - inter_wall_correct, ' regions are correctly classified'

print 'There are ', BG_wall, ' BG connected walls,'
print 'among which, ', BG_wall_correct, ' regions are correctly classified'

print 'There are ', BG_count - BG_wall, ' BG clutters,'
print 'among which, ', BG_correct - BG_wall_correct, ' regions are correctly classified'

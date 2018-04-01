import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier
import json
import numpy as np
from sklearn import svm
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


x = np.array(x)
y = np.array(y)

# Build a forest and compute the feature importances
forest = ExtraTreesClassifier(n_estimators=250,
                              random_state=0)

forest.fit(x, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

# Print the feature ranking
print("Feature ranking:")

for f in range(x.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(x.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")

name = np.array(['area','xmin','xmax','ymin','ymax','centroid_x','centroid_y','majorAxisLength','minorAxisLength',\
'orientation','eccentricity','solidity','is_free','is_interior','compactcness','long_line_lenth',\
'n_long_lines','n_free_pixels','n_unexplored_pixels','n_neighbors','ratio_of_simplifiede',\
'n_doors','total_door_length'])

plt.xticks(range(x.shape[1]), [name[i] for i in indices],rotation=50,ha = 'right')
plt.xlim([-1, x.shape[1]])
plt.subplots_adjust(left=0.1, right=0.9, bottom=0.3, top=0.9)
plt.savefig('plots/FeatureImportance')
plt.show()

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



y_pred =forest.predict(x_test)
print(float(sum(y_pred==y_test))/len(x_test))



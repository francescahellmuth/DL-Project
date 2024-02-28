# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#import needed libraries for project
import numpy as np # for maths 
import matplotlib # for plotting 
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd # pandas is a popular library in industry for manipulating large data tables

#directory specific to each person
#Sinead's ddir
ddir = '/Users/MYMTeam/Desktop/universe_10/'
#Tika's ddir
#ddir = ''
#Chess' ddir
#ddir = ''


# configure notebook for plotting
mpl.style.use('seaborn-colorblind') # colourblind-friendly colour scheme

# define default plot settings
matplotlib.rcParams['image.origin'] = 'lower'
matplotlib.rcParams['figure.figsize']=(8.0,6.0)    #(6.0,4.0)
matplotlib.rcParams['font.size']=16              #10 
matplotlib.rcParams['savefig.dpi']= 300             #72 

import warnings
warnings.filterwarnings('ignore')

'''         Mapping Galaxies within Each Camera Angle         '''
from sklearn.cluster import KMeans
    
pmin,pmax = -60, 60 # minimum and maximum pixel number in each image
#constrain if other galaxies within axis or noise to remove
#1st row = x constrains
#2nd row = y constrains
#3rd row = number of clusters

# hard to tell if galaxys
BOTTOM = [
    [ [pmin,-15],   [40,pmax]   ],
    [ [pmin,pmax],  [pmin,-30]  ],
    [     1,            1       ]
    ]

# partition 0 over to LEFT partition 1
TOP = [
    [ [pmin, -10], [-10,-6],  [00,10],    [36,40]   ],
    [ [pmin,-40],  [-35,-25], [-23,-18], [-42,-36]  ],
    [      1,           1,        1,          1     ]
    ]


5
LEFT = [
    [ [pmin,-30], [-30,-10], [00,6],    [6.2,11],  [11,14], [13,15.5], [31.5,35.5] ],
    [ [20,30],    [40,pmax], [15,18.5], [18,22.8], [21.5,25], [-2,2],  [16.5,21]   ],
    [     8,          1,        1,         14,        2,       1,           1      ]
    ]

#may be some more to document
RIGHT = [
    [ [-25,-20], [00,5],     [5,10],      [5,12.5],    [13.8,20] ],
    [ [10,20],   [pmin,-40], [-15,-10],   [-23,-17.6], [-23,-15] ],
    [    1,          1,           1,          6,          10     ]
    ]


BACK = [
    [ [pmin,-25], [30,40] ],
    [ [pmin,0],   [20,25] ],
    [     7,        4     ]
    ]

#maybe break up partion 6 into smaller
FRONT = [
    [ [-42,-34],   [-31,-25],  [-25,-10], [-18.5,-16], [-15,-10], [-15,-11], [-10,-8], [23.5,25],   [24,26.5], [27,27.5], [28.4,29], [27,27.5] ],
    [ [pmin, -10], [-21,-18],  [20,pmax], [3, 8],      [6.5,10],  [4,6],     [4,9],    [-1,-0.3],   [00,1.6],  [1.5,2],   [0.2,0.6], [-2,-1.3] ],
    [    1,            1,          1,       5,           9,         2,        4,         2,            5,       1,          1,          1      ]
    ]

DEFAULT = [
    [ [pmin,pmax] ],
    [ [pmin,pmax] ],
    [ 10 ]
    ]

###                 Boolean Sections                       ###
#change to 'True' to run section or 'False' to pass that section
#change camera and cameraData angle to change plots shown
camera = "Front"; cameraData = FRONT
plotPartitions = True # whether to plot clustering scatters

'''                 BEGIN ACTUAL CLUSTERING SECTION            '''

stars = pd.read_csv(ddir + camera + '/Star_Data.csv') 

if plotPartitions:
    # plot all the stars
    plt.scatter(stars.X,stars.Y, s = 0.1)
    plt.xlabel('x (pixels)')
    plt.ylabel('y (pixels)')
    plt.title(f'Star plot ({camera.lower()} camera)')
    plt.grid()
    plt.show() 

gno = 0 # how many galaxies already found
for i in range(len(cameraData[0])):
    xmin,xmax = cameraData[0][i]
    ymin,ymax = cameraData[1][i]
    ngalaxies = cameraData[2][i]
    
    partitionMask = (xmin <= stars.X) * (stars.X < xmax) * \
                    (ymin <= stars.Y) * (stars.Y < ymax) 
    starPartition = stars[ partitionMask ]
    
    # perform clustering fit
    R = np.array([ i for i in zip(*[starPartition.X,starPartition.Y]) ]) # (x,y) coords
    km = KMeans(n_clusters = ngalaxies, random_state = 0)
    km.fit(R)
    
    # update master dataframe with the new galaxy
    starPartition["Galaxy"] = gno + km.labels_
    gno += np.max(km.labels_) + 1
    stars.loc[partitionMask, "Galaxy"] = starPartition.Galaxy
    
    if plotPartitions:
        cmap = plt.cm.get_cmap('tab20')
        plt.scatter( starPartition.X, starPartition.Y, c = starPartition.Galaxy, s = 5,
                     cmap = cmap, alpha = 0.5)
        plt.title(f"{camera} camera, Partition {i}")
        plt.xlabel("x (pix)"); plt.ylabel("y (pix)")
        plt.axis('equal')
        plt.grid()
        plt.show()
    
stars.Galaxy = stars.Galaxy.astype(int)
galaxies = set(stars.Galaxy)
"""
Radiograph segmentation tutorial
===================================


"""


#%%
#Importing modules

import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import median, rank

#from xripl.data import shot
from xripl.reader import openRadiograph
from xripl.contrast import equalize
from xripl.clean import cleanArtifacts, flatten
from xripl.segmentation import detectShock
# import xripl.pltDefaults

from xripl.datasets import shot81431



#%%
# Plot

foreground, background = shot81431()
plt.imshow(foreground)
plt.title('Marble VC shot 81431')
plt.show()

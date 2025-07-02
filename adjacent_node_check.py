#!/usr/bin/env python3.10
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

from pydrake.all import *#!/usr/bin/env python3.8
import numpy as np
import matplotlib.pyplot as plt

import shapely

from shapely.geometry import Point, LineString

from pydrake.all import *


# CONNECTING ADJACENT NODES

def check_rV_list(coord, r_V_list):
    
    for v_pol in r_V_list:
        if v_pol.PointInSet(coord) == False:
            return False
    return True
            

def connect_centers(center_list, r_V_list):
    connect_pairs = []
    for center_idx in range(len(center_list)):
        for center_el in center_list:

            if center_list[center_idx] == center_el:
                continue
            
            line = shapely.get_coordinates(LineString(center_list[center_idx], center_el)).tolist()

            valid = True
            for coord in line:
                valid = check_rV_list(coord, r_V_list)
                if valid == False:
                    break
            if valid == True:
                connect_pairs.append([center_list[center_idx], center_el])

    return connect_pairs

center_list = [[17, 4], [5, 14], [26, 16], [4, 4], [25, 15], [16, 11], [28, 5], [17, 10], [17, 11], [28, 6], [28, 11]]

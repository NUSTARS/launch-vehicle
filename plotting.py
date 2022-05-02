import numpy as np 
from matplotlib import pyplot as plt
from scipy.ndimage.filters import uniform_filter1d


def import_data(data_path, variable, n_avg=0):

    assert(isinstance(data_path,str)),"data_path must be a string"
    assert(isinstance(variable,str)),"variable must be a string"
    assert(isinstance(n_avg,int)),"n_avg must be an integer"
    
    variables_dict = {
        "altitude": 1,
        "pressure": 2,
        "velocity": 3,
        "temperature": 4,
        "voltage": 5,
    }

    data_index = variables_dict[variable]

    data = []
    time = []
    first = True
    
    with open(data_path,"r") as d:
        for row in d:
            row = row.split(",")    
            if first is False:
                data.append(float(row[data_index]))
                time.append(float(row[0]))
            else:
                first = False
        
    if n_avg != 0:
        data = uniform_filter1d(data,n_avg)

    return time,data
    

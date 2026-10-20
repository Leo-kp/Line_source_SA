import numpy as np

from scipy.interpolate import PchipInterpolator as Pchip

#----------------------------
#interpolation
def objective_function(npyresult,datafield):

    t_source=np.array(npyresult["timevalues"])
    p_source=np.array(npyresult["values"]) #values, either pressure or other variable
    p_field=datafield["Pint_downhole [MPa]_s"]*1e6 #Pa values

    # sort_idx= np.argsort(t_source) #sortingin case no order in output
    # t_source=t_source[sort_idx]
    # p_source=p_source[sort_idx]

    f= Pchip(t_source,p_source)
    p_interp=f(datafield["Zeit [s]"])

    rsme=np.sqrt(np.mean((p_interp-p_field)**2))

    return rsme


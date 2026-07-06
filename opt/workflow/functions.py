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

#-----------------------------------------
# keff calculator

def calculate_keff(factors):
    pjack=factors['pjack']
    p1=factors['p1']
    p2=factors['p2']
    wr=max(factors['wr'], 1e-6)
    k01=factors['k01']
    k02=factors['k02']

    prev=np.linspace(p1,p2,50)

    tanh_term = np.tanh((prev - pjack) / wr)
    c=(k02 - k01) * 0.5
    k_value = k01 + c* (1 + tanh_term)

    sf0=factors['sf0']
    beta_dimen=factors['b_dim']
    beta=pjack*beta_dimen

    X= np.sqrt(3)*np.sqrt(k_value)/k_value
    Y= c*(1-tanh_term**2)/wr
    s_value=  sf0 + beta*X*Y

    keff=k_value*s_value/sf0
    return keff
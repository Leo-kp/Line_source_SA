import numpy as np
import ogstools as ot
from pathlib import Path

def extract_values(OUT_DIR): 

    y=-40.6
    r_st=0.038 
    coords = np.array([[r_st, y, 1e-18]])
    
    pvd_files = list(Path(OUT_DIR).glob("*.pvd"))
    if not pvd_files:
        raise FileNotFoundError(f"No .pvd file found in {OUT_DIR}")
    
    pvd_path = pvd_files[0]
    ms = ot.MeshSeries(pvd_path)
    pressure=ot.variables.pressure
    pressure= pressure.replace(output_unit="MPa")
    
    ms_probes=ms.probe(points=coords)
    raw_pressure_array = ms_probes['pressure']
    clean_p = np.squeeze(raw_pressure_array) #or raw_pressure_array()[:,0] taking the one point scalar

    data_bundle = {
        'values': clean_p, 
        'timevalues': np.array(ms.timevalues),
        'metadata': {
            'variable_name': 'pressure',
            'unit': 'MPa',
            'time_unit': 's',
            'coordinates': coords,
            'source_file': str(pvd_path)
        }
    }

    return data_bundle


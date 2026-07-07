import sys
import numpy as np
import ogstools as ot
from pathlib import Path

def extract_values(OUT_DIR): 

    y=-40.6
    r_st=0.038 
    coords = np.array([[r_st, y, 1e-18]])
    
    pvd_files = list(Path(OUT_DIR).glob("*.pvd")) #avoidin locking memory so use glob()
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
        "values": clean_p.tolist() if isinstance(clean_p,np.ndarray) else [float(clean_p)],
        "timevalues": np.array(ms.timevalues).tolist()
    } #optimising saving and internal processing of results

    return data_bundle

if __name__ == "__main__": #standalone execution
    if len(sys.argv)>2:
        target_out_dir=sys.argv[1] #output OGS files
        destination_npy_path=sys.argv[2] #save npy extracted

        try: 
            extract_bundle = extract_values(target_out_dir)
            np.save(destination_npy_path,extract_bundle)
            sys.exit(0)
        except Exception as err:
            print(f"Error: {str(err)}")
            sys.exit(1)




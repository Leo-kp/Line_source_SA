import os
import sys
import shutil
from pathlib import Path

import logging
import multiprocessing

#-------------------------------------------
# python and jupyter neutral
if '__file__' in locals() or '__file__' in globals():
    CODE_DIR=Path(__file__).resolve().parent
else:
    CODE_DIR= Path.cwd().resolve()
BASE_DIR=CODE_DIR.parent if CODE_DIR.name=="workflow" else CODE_DIR


# ----------------------------------------
#OGS init and env

MANUAL_OGS_PATH= r"C:\OGS_Binary\ogs-6.5.7-Windows-10.0.26200-python-3.13.7-utils\bin"
OGS_PYTHON_EXE= r"C:\Miniforge-Data\envs\ogs_exc_313\python.exe" #When split env ogs and other, use path to env of python #sys.executable # when alltogether env
def set_ogs_environment():

    if MANUAL_OGS_PATH:
        bin_dir=Path(MANUAL_OGS_PATH).resolve()
        if bin_dir.exists() and bin_dir.is_dir():
            return bin_dir
        
    env_root= Path(sys.prefix)
    for sub_folder in ["bin","Library/bin"]:
        probing_dir = env_root/sub_folder
        if probing_dir.exists():
            if (probing_dir/"ogs").exists() or (probing_dir/"ogs.exe").exists():
                return probing_dir.resolve()
            
    system_match=shutil.which("ogs")
    if system_match:
        return  Path(system_match).resolve().parent
    
    return None

OGS_BIN_DIR = set_ogs_environment() 

if OGS_BIN_DIR:
    os.environ["OGS_BIN_PATH"] = str(OGS_BIN_DIR)
    os.environ["PATH"]= f"{OGS_BIN_DIR}{os.pathsep}{os.environ.get('PATH','')}"

    OGS_BINARY = OGS_BIN_DIR/("ogs.exe") if os.name== "nt" else "ogs"
else:
    raise FileNotFoundError(
        "Not OGS binary directory.\n"
        "Please,verify paths to /bin containing 'ogs'"
    )

TOTAL_LOGICAL_CORES=multiprocessing.cpu_count()
try:
    import psutil
    TOTAL_PHYSICAL_CORES= psutil.cpu_count(logical=False) or TOTAL_LOGICAL_CORES
except ImportError:
    TOTAL_PHYSICAL_CORES=max(1,TOTAL_LOGICAL_CORES//2)
USE_NUM_THREADS=max(1, TOTAL_PHYSICAL_CORES-2)

os.environ["OMP_NUM_THREADS"]= str(USE_NUM_THREADS)
os.environ["MKL_NUM_THREADS"]= str(USE_NUM_THREADS)
os.environ["OPENBLAS_NUM_THREADS"]=str(USE_NUM_THREADS)

print(f"[CONFIG] Env loaded. OGS threads manually set to: {USE_NUM_THREADS}")


#---------------------------------------
#Folder three and dirs with files
MESH_DIR = BASE_DIR/"mesh"
OUT_DIR = BASE_DIR/"out" #simpler than Path(os.environ.get()) for local execution
RUN_DIR= BASE_DIR/"run"
DATA_DIR=BASE_DIR/"data"
MORRIS_DIR=BASE_DIR.parent

VERSION="v12"
RESULTS_DIR=MORRIS_DIR/f"results_{VERSION}"

MORRIS_SAMPLES_CSV=RESULTS_DIR/f"morris_samples_{VERSION}.csv"
MORRIS_RAW_DATA_DIR=RESULTS_DIR/"results/raw_data"

TEMPLATE_PRJ = CODE_DIR/ "BH10_20180718_40.6_opt.prj"
WORKFLOW_DATA = CODE_DIR/"BH10_20180718_40.6_SR_v2.csv"

RUNTIME_PRJ  = OUT_DIR/ "BH10_20180718_40.6_temp.prj"
FIELD_DATA_PATH = DATA_DIR/"BH10_20180718_40.6_SR_v2.csv"

MESH_FILENAME= "symmetric_cylinder_3D.msh"
STATIC_MESH_PATH =MESH_DIR / MESH_FILENAME
DYNAMIC_MESH_PATH= OUT_DIR/MESH_FILENAME

IS_MESH_DYNAMIC=False
ACTIVE_MESH_PATH= DYNAMIC_MESH_PATH if IS_MESH_DYNAMIC else STATIC_MESH_PATH
STATIC_MESH_PREFIX=f"../{MESH_DIR.name}"

def initialize_project_folders(): #not hanging execution, so wrapped in function
    for folder in [MESH_DIR, OUT_DIR, RUN_DIR,  DATA_DIR]:
        folder.mkdir(parents=True, exist_ok=True)

    if WORKFLOW_DATA.exists() and not  FIELD_DATA_PATH.exists():
        shutil.copy2(WORKFLOW_DATA,FIELD_DATA_PATH)

initialize_project_folders()

#------------------------------------------
#bounds and factors to optmise

OPTIMISATION_CONFIG={
        "bounds":{
            "pjack":(3.1e6,3.6e6),
            "wr":(0.2,0.5)
        },
        "initial_guess":[3.43e6,0.4e6]

}


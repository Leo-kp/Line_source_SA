import ogstools as ot
from pathlib import Path

#--------------------------------------------------
#prj updater function

def temp_prj(prj_in: Path, prj_out:Path,factors:dict, is_dynamic:bool,static_prefix:str): 

    keff=factors['keff']

    values_str = " ".join(map(str, keff))

    model = ot.Project(input_file=prj_in,output_file=prj_out)
    xpath='./curves/curve[name="k_curve"]/values'
    medium=1

    root_xml=model.root
    meshes_block=root_xml.find("./meshes")

    try:
        model.replace_text(values_str,xpath)
        
        if meshes_block is not None:
            for mesh_element in meshes_block.findall('mesh axially_symmetric="true"'):
                current_mesh_name= mesh_element.text.strip()
                raw_filename= Path(current_mesh_name).name

                if is_dynamic:
                    mesh_element.text= raw_filename
                else:
                    mesh_element.text=f"{static_prefix}{raw_filename}"
        else:
            print("Warning: No <meshes> block in project file structure")


    except Exception as e:
        print(f"CRITICAL ERROR in PRJ update: {e}")
        raise # Stop the loop if the input file is not correctly updated

    model.write_input(prj_out)

if __name__=="__main__":
    import sys
    import json
    if len(sys.argv)>5:
        p_in=sys.argv[1]
        p_out=sys.argv[2]
        factors=json.loads(sys.argv[3])
        is_dyn=sys.argv[4].lower()=='true'
        prefix=sys.argv[5]

        temp_prj(p_in,p_out,factors,is_dyn,prefix)
        sys.exit(0)
    sys.exit(1)
import ogstools as ot
from pathlib import Path

#--------------------------------------------------
#prj updater function

def temp_prj(prj_in: Path, prj_out:Path,factors:dict, is_dynamic:bool,static_prefix:str): 

    keff=factors['keff']
    sf0=factors['sf0'] #*******************

    values_str = " ".join(map(str, keff))

    model = ot.Project(input_file=prj_in,output_file=prj_out)
    xpath='./curves/curve[name="k_curve"]/values'
    medium=1

    try:
        model.replace_text(values_str,xpath)
        model.replace_medium_property_value(mediumid=medium, name='storage', value=sf0, propertytype='Constant', valuetag='value') #3rd parameter

        xml_tree=model.tree
        meshes_containers= xml_tree.findall(".//meshes/mesh")

        for mesh_tag in meshes_containers:
            if mesh_tag.text:
                current_mesh_name=mesh_tag.text.strip()
                raw_filename=Path(current_mesh_name).name

                if is_dynamic:
                    new_mesh_name=raw_filename
                else:
                    new_mesh_name=f"{static_prefix}{raw_filename}"

                mesh_tag.text=new_mesh_name

        model.write_input(prj_out)

    except Exception as e:
        print(f"CRITICAL: Failed to modify ogstools XML tree: {e}")
        raise e


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
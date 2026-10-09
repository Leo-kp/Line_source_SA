import numpy as np
from scipy import special as sc 

import meshio


#----------------------------------------------------------------------

def save_combined_mesh(msh_file, output_path, fracture_label="fracture"):
    """
    Reads MSH and saves a single VTU with all elements and MaterialIDs.
    Focus on changing the ID of the fracture from domain
    Inputs.
        msh_file: Mesh file (path/file.msh)
        output_path: New vtu file (path/file.msh)
        fracture_label: name of element to change
    Output.
        vtu file with combined elements (rock+fracture) with different IDs
    """
   
    msh = meshio.read(msh_file)
    try:
        target_id = msh.field_data[fracture_label][0]
    except KeyError:
        print(f"Warning: '{fracture_label}' not found. Combined mesh may lack IDs.")
        return

    valid_cells = []
    valid_material_ids = []

    for i, cell_block in enumerate(msh.cells):
        if cell_block.type in ["line", "triangle", "quad"]:
            n_cells = len(cell_block.data)
            block_ids = np.zeros(n_cells, dtype=np.int32)
            
            if i < len(msh.cell_data.get("gmsh:physical", [])):
                ids_in_block = msh.cell_data["gmsh:physical"][i]
                block_ids[ids_in_block == target_id] = 1
            
            valid_cells.append(cell_block)
            valid_material_ids.append(block_ids)

    combined_mesh = meshio.Mesh(
        points=msh.points,
        cells=valid_cells,
        cell_data={"MaterialIDs": valid_material_ids}
    )
    combined_mesh.write(output_path)
    # print(f"Combined mesh saved to: {output_path}") #silent in multimesh evaluation


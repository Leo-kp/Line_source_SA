from pathlib import Path

import gmsh
import meshio
import ogstools as ot
import pyvista as pv

import numpy as np

import config 


#---------------------------------------
#mesh gen and extraction

def create_rectangle_frac_mesh_v3(
    filepath: Path,
    radius: float,
    height: float,
    mesh_size: float,
    center_z: float = 0.0,
    r_well: float = 0.01,
    length: float = 8.0,
    refine_well: float = 0.05,  # Element size at the well
    refine_frac: float = 0.02   # Element size along the fracture
) -> None:

    if refine_well >= mesh_size or refine_frac >= mesh_size:
        raise ValueError("Refinement sizes must be strictly smaller than the global mesh_size.")

    gmsh.initialize()
    
    try:
        gmsh.option.setNumber("General.Verbosity", 0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
        gmsh.model.add(filepath.stem)
        
        z = center_z
        r_start = r_well
        r_end = radius + r_start

        p1 = gmsh.model.occ.addPoint(r_start, z + height / 2, 0.0, 0.0) # Top-Well
        p2 = gmsh.model.occ.addPoint(r_end,   z + height / 2, 0.0, 0.0) # Top-Right
        p3 = gmsh.model.occ.addPoint(r_end,   z - height / 2, 0.0, 0.0) # Bot-Right
        p4 = gmsh.model.occ.addPoint(r_start, z - height / 2, 0.0, 0.0) # Bot-Well
        
        p5 = gmsh.model.occ.addPoint(r_start, z, 0.0, 0.0)              # Intersection
        p6 = gmsh.model.occ.addPoint(length,  z, 0.0, 0.0)              # Frac-Tip
        p7 = gmsh.model.occ.addPoint(r_end,   z, 0.0, 0.0)              # Far-Mid
        
        gmsh.model.occ.synchronize()
        
        l_well_top  = gmsh.model.occ.addLine(p1, p5)
        l_well_bot  = gmsh.model.occ.addLine(p5, p4)
        l_frac      = gmsh.model.occ.addLine(p5, p6)
        l_top       = gmsh.model.occ.addLine(p2, p1)
        l_bot       = gmsh.model.occ.addLine(p4, p3)
        l_right_top = gmsh.model.occ.addLine(p7, p2)
        l_right_bot = gmsh.model.occ.addLine(p3, p7)
        l_connector = gmsh.model.occ.addLine(p6, p7)
        
        gmsh.model.occ.synchronize()
        
        cl1 = gmsh.model.occ.addCurveLoop([l_well_top, l_frac, l_connector, l_right_top, l_top])
        surf1 = gmsh.model.occ.addPlaneSurface([cl1])
        
        cl2 = gmsh.model.occ.addCurveLoop([l_well_bot, l_bot, l_right_bot, -l_connector, -l_frac])
        surf2 = gmsh.model.occ.addPlaneSurface([cl2])
        
        gmsh.model.occ.synchronize()
        
        gmsh.model.occ.synchronize()
        
        all_surfs = gmsh.model.occ.getEntities(2)
        gmsh.model.occ.fragment(all_surfs, []) 
        gmsh.model.occ.synchronize()

        # --- Dynamic Fields Calculations - local refinements---
        limiting_dim = min(radius, height)

        d_well = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_well, "CurvesList", [l_well_top, l_well_bot])
        gmsh.model.mesh.field.setNumber(d_well, "Sampling", 100)

        t_well = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_well, "InField", d_well)
        gmsh.model.mesh.field.setNumber(t_well, "SizeMin", refine_well)
        gmsh.model.mesh.field.setNumber(t_well, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(t_well, "DistMin", refine_well * 2.0)
        gmsh.model.mesh.field.setNumber(t_well, "DistMax", min(limiting_dim * 0.25, radius * 0.3))

        d_frac = gmsh.model.mesh.field.add("Distance")
        gmsh.model.mesh.field.setNumbers(d_frac, "CurvesList", [l_frac])
        gmsh.model.mesh.field.setNumber(d_frac, "Sampling", 200) # Higher sampling for high aspect line

        t_frac = gmsh.model.mesh.field.add("Threshold")
        gmsh.model.mesh.field.setNumber(t_frac, "InField", d_frac)
        gmsh.model.mesh.field.setNumber(t_frac, "SizeMin", refine_frac)
        gmsh.model.mesh.field.setNumber(t_frac, "SizeMax", mesh_size)
        gmsh.model.mesh.field.setNumber(t_frac, "DistMin", refine_frac * 3.0)
    
        gmsh.model.mesh.field.setNumber(t_frac, "DistMax", height * 0.15) 

        f_min = gmsh.model.mesh.field.add("Min")
        gmsh.model.mesh.field.setNumbers(f_min, "FieldsList", [t_well, t_frac])
        
        gmsh.model.mesh.field.setAsBackgroundMesh(f_min)

        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        #--------------

        gmsh.model.addPhysicalGroup(2, [surf1, surf2], name="bulk_mesh")
        gmsh.model.addPhysicalGroup(1, [l_well_top, l_well_bot], name="well")
        gmsh.model.addPhysicalGroup(1, [l_frac], name="fracture")
        gmsh.model.addPhysicalGroup(1, [l_top], name="top")
        gmsh.model.addPhysicalGroup(1, [l_bot], name="bottom")
        gmsh.model.addPhysicalGroup(1, [l_right_top, l_right_bot], name="boundary_R")
        
        gmsh.model.addPhysicalGroup(0, [p5], name="intersection_point")
        gmsh.model.addPhysicalGroup(0, [p6], name="fracture_tip")

        gmsh.model.mesh.generate(2)
        gmsh.write(str(filepath.with_suffix(".msh")))

    except Exception as e:
        print(f"[ERROR] Mesh generation failed: {e}")
        raise e
    finally:
        gmsh.finalize()

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

    #----------------------------------------------------

def generate_optimization_mesh(MSH_FILE=None):#wraper for safe execution in modules, None combined with if ...is None ensures dynamic udpate
    if MSH_FILE is None:
        MSH_FILE=config.ACTIVE_MESH_PATH
    MSH_FILE= Path(MSH_FILE)
    h=0.7 #mesh as in field data
    create_rectangle_frac_mesh_v3(
        MSH_FILE,
        radius= 100,
        height= h,
        mesh_size= h/4,
        center_z=-40.6,
        r_well = 0.038,
        length = 0.5,
        refine_well = h/20,  # Element size at the well
        refine_frac = h/30   # Element size along the fracture
    ) 

    meshes = ot.Meshes.from_gmsh(MSH_FILE, log=False)
    for name, mesh in meshes.items():
        vtu_path = (config.MESH_DIR / f"rectangle_{name}.vtu").as_posix()
        pv.save_meshio(vtu_path, mesh)
        #print(f"Saved {vtu_path}")

    combined_vtu = (config.MESH_DIR / "combined_fracture_mesh.vtu").as_posix()
    save_combined_mesh(MSH_FILE, combined_vtu)

if __name__=="__main__":
    import sys
    if len(sys.argv)>1:
        active_mesh_path=sys.argv[1]
        generate_optimization_mesh(active_mesh_path)
        sys.exit(0)
    sys.exit(0)
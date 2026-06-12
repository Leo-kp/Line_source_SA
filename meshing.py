from pathlib import Path
import gmsh
import math

def create_scylindre_mesh(
    filepath: Path,
    radius: float,
    thickness: float,
    mesh_size: float,
    r_well: float = 0.0,      # Starting radius (a>0 for line, af>a for finite)
    refine_size: float=0.1, 
    center_y: float = 0.0
) -> None:
    
    is_invalid_scale = refine_size >= mesh_size

    if is_invalid_scale:
        raise ValueError(
            f"Aborting mesh generation: 'refine_size' ({refine_size}) must be "
            f"strictly smaller than 'mesh_size' ({mesh_size}) for local refinement to work."
        )
    
    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.add(filepath.stem)

    z = center_y
    r_start = r_well 
    r_end = r_well + radius

    p1 = gmsh.model.occ.addPoint(r_start,      z+thickness/2, 0, refine_size)
    p2 = gmsh.model.occ.addPoint(r_end,   z+thickness/2, 0, mesh_size)
    p3 = gmsh.model.occ.addPoint(r_end,   z-thickness/2, 0, mesh_size)
    p4 = gmsh.model.occ.addPoint(r_start,      z-thickness/2, 0,  refine_size)

    l1 = gmsh.model.occ.addLine(p2, p1)
    l2 = gmsh.model.occ.addLine(p3, p2)
    l3 = gmsh.model.occ.addLine(p4, p3)
    l4 = gmsh.model.occ.addLine(p1, p4)
    gmsh.model.occ.synchronize()

    cl   = gmsh.model.occ.addCurveLoop([l4, l3, l2, l1])
    surf = gmsh.model.occ.addPlaneSurface([cl])
    gmsh.model.occ.synchronize()

    pg_domain = gmsh.model.addPhysicalGroup(2, [surf])
    gmsh.model.setPhysicalName(2, pg_domain, "domain")
    bcs = [("top", l1), ("boundary_R", l2), ("bottom", l3), ("well", l4)]
    for name, line in bcs:
        pg = gmsh.model.addPhysicalGroup(1, [line])
        gmsh.model.setPhysicalName(1, pg, name)

    # --- local refinement ---
    d_id = gmsh.model.mesh.field.add("Distance")
    gmsh.model.mesh.field.setNumbers(d_id, "CurvesList", [l4])
    gmsh.model.mesh.field.setNumber(d_id, "Sampling", 100) 

    t_id = gmsh.model.mesh.field.add("Threshold")
    gmsh.model.mesh.field.setNumber(t_id, "InField", d_id)
    gmsh.model.mesh.field.setNumber(t_id, "SizeMin", refine_size) 
    gmsh.model.mesh.field.setNumber(t_id, "SizeMax", mesh_size)   

    limiting_dimension = min(radius, thickness)

    dynamic_dist_min = min(refine_size * 2.0, limiting_dimension * 0.05)
    dynamic_dist_max = limiting_dimension * 0.35

    min_runway_needed = mesh_size * 2.0
    if (dynamic_dist_max - dynamic_dist_min) < min_runway_needed:
        dynamic_dist_max = min(dynamic_dist_min + min_runway_needed, radius * 0.9)

    if dynamic_dist_max <= dynamic_dist_min:
        dynamic_dist_max = radius * 0.5
        dynamic_dist_min = radius * 0.1

    dynamic_dist_min = refine_size 
    
    if refine_size > 0:
        ratio = mesh_size / refine_size
        dynamic_dist_max = dynamic_dist_min + (mesh_size * math.log1p(ratio))
    else:
        dynamic_dist_max = radius * 0.25 

    max_allowed_transition = radius * 0.4
    dynamic_dist_max = min(dynamic_dist_max, max_allowed_transition)

    gmsh.model.mesh.field.setNumber(t_id, "DistMin", dynamic_dist_min)
    gmsh.model.mesh.field.setNumber(t_id, "DistMax", dynamic_dist_max)
    gmsh.model.mesh.field.setAsBackgroundMesh(t_id)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
# ----------------------------------------
    gmsh.model.mesh.generate(2)
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

#------------------------------------------------------------------------
def create_rectangle_frac_mesh(
    filepath: Path,
    radius: float,
    height: float,
    mesh_size: float,
    center_z: float = 0.0,
    r_well: float = 0.01,
    length:float=8.,
    mode="domain",
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", mesh_size)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    gmsh.model.add(filepath.stem)
    
    z = center_z
    r_start = r_well
    r_end=radius+r_start

    p1 = gmsh.model.occ.addPoint(r_start, z + height / 2, 0.0, mesh_size)
    p2 = gmsh.model.occ.addPoint(r_end, z + height / 2, 0.0, mesh_size/10)
    p3 = gmsh.model.occ.addPoint(r_end, z - height / 2, 0.0, mesh_size/10)
    p4 = gmsh.model.occ.addPoint(r_start, z - height / 2, 0.0, mesh_size)
    
    p5 = gmsh.model.occ.addPoint(r_start, z, 0.0, mesh_size/10)
    p6 = gmsh.model.occ.addPoint(length, z, 0.0, mesh_size/10)
    p7= gmsh.model.occ.addPoint(r_end, z, 0.0, mesh_size)
    gmsh.model.occ.synchronize()
    
    l1 = gmsh.model.occ.addLine(p2, p1)
    l2 = gmsh.model.occ.addLine(p1, p5)
    l3 = gmsh.model.occ.addLine(p5, p4)
    l4 = gmsh.model.occ.addLine(p4, p3)
    l5 = gmsh.model.occ.addLine(p3, p7)
    l6 = gmsh.model.occ.addLine(p7, p2)

    l7 = gmsh.model.occ.addLine(p5, p6)
    l8 = gmsh.model.occ.addLine(p6, p7)
    gmsh.model.occ.synchronize()
    
    cl1 = gmsh.model.occ.addCurveLoop([l1,l2,l7,l8,l6])
    surf1 = gmsh.model.occ.addPlaneSurface([cl1])
    cl2 = gmsh.model.occ.addCurveLoop([l3,l4,l5,-l8,-l7])
    surf2 = gmsh.model.occ.addPlaneSurface([cl2])
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    if mode == "BC":
        bcs = [("top", l1), ("bottom", l4)]
        for name, line in bcs:
            pg = gmsh.model.addPhysicalGroup(1, [line])
            gmsh.model.setPhysicalName(1, pg, name)
        gmsh.model.addPhysicalGroup(1, [l5, l6], name="boundary_R")
        gmsh.model.addPhysicalGroup(1, [l2, l3], name="well")
      
    elif mode == "domain":
        gmsh.model.addPhysicalGroup(2, [surf1, surf2], name="surf")
        gmsh.model.addPhysicalGroup(1, [l7], name="fracture")
        
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

#---------------------------------------------------------------------------------
def create_rectangle_frac_mesh_v2(
    filepath: Path,
    radius: float,
    height: float,
    mesh_size: float,
    center_z: float = 0.0,
    r_well: float = 0.01,
    length: float = 8.0,
) -> None:

    gmsh.initialize()
    gmsh.option.setNumber("General.Verbosity", 0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.0)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", mesh_size)
    
    gmsh.model.add(filepath.stem)
    
    z = center_z
    r_start = r_well
    r_end = radius + r_start

    p1 = gmsh.model.occ.addPoint(r_start, z + height / 2, 0.0, mesh_size) # Top-Well
    p2 = gmsh.model.occ.addPoint(r_end, z + height / 2, 0.0, mesh_size)   # Top-Right
    p3 = gmsh.model.occ.addPoint(r_end, z - height / 2, 0.0, mesh_size)   # Bot-Right
    p4 = gmsh.model.occ.addPoint(r_start, z - height / 2, 0.0, mesh_size) # Bot-Well
    
    p5 = gmsh.model.occ.addPoint(r_start, z, 0.0, 0.0)              # INTERSECTION
    p6 = gmsh.model.occ.addPoint(length, z, 0.0, mesh_size)               # Frac-Tip
    p7 = gmsh.model.occ.addPoint(r_end, z, 0.0, mesh_size)                # Far-Mid
    
    gmsh.model.occ.synchronize()
    
    l_well_top = gmsh.model.occ.addLine(p1, p5)
    l_well_bot = gmsh.model.occ.addLine(p5, p4)
    
    l_frac = gmsh.model.occ.addLine(p5, p6)
    
    l_top = gmsh.model.occ.addLine(p2, p1)
    l_bot = gmsh.model.occ.addLine(p4, p3)
    l_right_top = gmsh.model.occ.addLine(p7, p2)
    l_right_bot = gmsh.model.occ.addLine(p3, p7)
    l_connector = gmsh.model.occ.addLine(p6, p7)
    
    gmsh.model.occ.synchronize()
    
    cl1 = gmsh.model.occ.addCurveLoop([l_top, l_right_top, l_connector, l_frac, l_well_top])
    surf1 = gmsh.model.occ.addPlaneSurface([cl1])
    
    cl2 = gmsh.model.occ.addCurveLoop([-l_frac, -l_connector, l_right_bot, l_bot, l_well_bot])
    surf2 = gmsh.model.occ.addPlaneSurface([cl2])
    
    #-------------------------------
    all_entities = (gmsh.model.occ.getEntities(2) + 
                gmsh.model.occ.getEntities(1) + 
                gmsh.model.occ.getEntities(0))
    gmsh.model.occ.fragment(all_entities, []) 
    #------------------------------
    
    gmsh.model.occ.synchronize()
    gmsh.model.mesh.generate(2)

    gmsh.model.addPhysicalGroup(2, [surf1, surf2], name="bulk_mesh")
    gmsh.model.addPhysicalGroup(1, [l_well_top, l_well_bot], name="well")
    gmsh.model.addPhysicalGroup(1, [l_frac], name="fracture")
    gmsh.model.addPhysicalGroup(1, [l_top], name="top")
    gmsh.model.addPhysicalGroup(1, [l_bot], name="bottom")
    gmsh.model.addPhysicalGroup(1, [l_right_top, l_right_bot], name="boundary_R")
    
    gmsh.model.addPhysicalGroup(0, [p5], name="intersection_point")
    gmsh.model.addPhysicalGroup(0, [p6], name="fracture_tip")
    
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.write(str(filepath.with_suffix(".msh")))
    gmsh.finalize()

#------------------------
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
    #-------------------------------------------
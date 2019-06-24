import bpy
import bmesh


def run(mmvt):
    for obj in bpy.data.objects:
        obj.select = False
    bm = bmesh.new()
    for root_folder in ['Subcortical_structures', 'Subcortical_fmri_activity_map', 'Subcortical_meg_activity_map']:
        for obj in bpy.data.objects[root_folder].children:
            mesh = obj.data
            bm.from_mesh(mesh)
            print('Recalc normals for {}'.format(obj.name))
            bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
            bm.to_mesh(mesh)
            bm.clear()
            mesh.update()
    bm.free()

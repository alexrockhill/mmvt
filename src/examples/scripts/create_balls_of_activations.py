import bpy
import os.path as op
import numpy as np
import time
import glob
import os.path as op


def run(mmvt):
    mu = mmvt.utils
    if bpy.context.scene.balls_of_activations_all_folder:
        csv_files = glob.glob(op.join(mu.get_real_fname('balls_of_activations_folder'), '*.csv'))
        colors = mmvt.colors_utils.get_distinct_colors_names(len(csv_files))
        for csv_fname, color in zip(csv_files, colors):
            print('{}: {}'.format(mu.namebase(csv_fname), color))
            rgb = mmvt.colors_utils.name_to_rgb(color)
            import_csv(mmvt, csv_fname, {1: rgb}, suffix=mu.namebase(csv_fname))
    else:
        csv_fname = mu.get_real_fname('balls_of_activations_csv_fname')
        if not op.isfile(csv_fname):
            print('Can\'t find csv fname! {}!'.format(csv_fname))
            return False
        import_csv(mmvt, csv_fname, {1: 'blue', 2: 'red'})
    mmvt.appearance.show_electrodes(True)


def import_csv(mmvt, csv_fname, balls_c=None, balls_r=None, suffix=''):
    mu = mmvt.utils
    if balls_c is None:
        balls_c = {1: 'blue', 2: 'red'}
    if balls_r is None:
        balls_r = {1: 0.1, 2: 0.2}
    lines = list(mu.csv_file_reader(csv_fname, find_encoding=True))
    now = time.time()
    for ind, line in enumerate(lines):
        mu.time_to_go(now, ind, len(lines), runs_num_to_print=10)
        mni305_ras = np.array([float(x) for x in line[:3]])
        cond = int(line[3]) if len(line) > 3 else 1
        primary = int(line[4]) if len(line) > 4 else 2
        # subject_tkreg_ras = mmvt.where_am_i.mni305_ras_to_subject_tkreg_ras(mni305_ras)
        if bpy.context.scene.balls_of_activations_pos_to_current_inflation:
            subject_tkreg_ras = mmvt.where_am_i.pos_to_current_inflation(subject_tkreg_ras, subject_tkreg_ras=True)
        else:
            import mathutils
            matrix_world = mu.get_matrix_world()
            subject_tkreg_ras = mathutils.Vector(mni305_ras) * matrix_world

        subject_tkreg_ras[0] *= -1
        subject_tkreg_ras[1] *= -1
        ball_name = 'peak_{}_{}_{}{}'.format(ind, cond, primary, '_{}'.format(suffix) if suffix != '' else '')
        mmvt.data.create_electrode(
            subject_tkreg_ras, ball_name, balls_r[primary], color=balls_c[cond], subject_tkreg_ras=True)


bpy.types.Scene.balls_of_activations_csv_fname = bpy.props.StringProperty(subtype='FILE_PATH')
bpy.types.Scene.balls_of_activations_folder = bpy.props.StringProperty(subtype='DIR_PATH')
bpy.types.Scene.balls_of_activations_pos_to_current_inflation = bpy.props.BoolProperty(default=True)
bpy.types.Scene.balls_of_activations_all_folder = bpy.props.BoolProperty(default=False)


def draw(self, context):
    layout = self.layout
    if bpy.context.scene.balls_of_activations_all_folder:
        layout.prop(context.scene, 'balls_of_activations_folder', text='Folder')
    else:
        layout.prop(context.scene, 'balls_of_activations_csv_fname', text='csv')
    layout.prop(context.scene, 'balls_of_activations_all_folder', text='All folder')
    layout.prop(context.scene, 'balls_of_activations_pos_to_current_inflation', text='Project to surface')

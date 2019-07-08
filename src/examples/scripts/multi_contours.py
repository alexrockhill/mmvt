import bpy
import glob
import os.path as op
from scripts_panel import ScriptsPanel
from coloring_panel import ColoringMakerPanel as coloring_panel
import mne


def _mmvt():
    return ScriptsPanel.addon


def run(mmvt):
    import importlib
    mu = mmvt.mmvt_utils
    mu.add_mmvt_code_root_to_path()
    from src.preproc import meg
    stc_name = mmvt.coloring.get_meg_files()
    t = mmvt.coloring.get_current_time()
    thresholds_min = 10
    thresholds_dx = 3
    #colorbar_name = 'YlOrRd'
    colorbar_name = 'PuBu'
    importlib.reload(meg)
    # to do: if threshold min > colorbar max throw reasonable error
    _, all_contours = meg.stc_to_contours(mu.get_user(), stc_name, pick_t=t,
                                          thresholds_min=thresholds_min,
                                          thresholds_dx=thresholds_dx)
    clusters_root_fol = op.join(mu.get_user_fol(), 'meg', 'clusters')
    #for threshold, contours in all_contours.items():
    output_fname = '{}_contoures_{}'.format(stc_name, t)
    mmvt.meg.plot_activity_contours(output_fname, colorbar_name)
    #mmvt.labels.color_contours(specific_color=(1,0,0), labels_contours=contours)
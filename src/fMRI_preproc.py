try:
    from surfer import Brain
    from surfer import viz
    # from surfer import project_volume_data
    SURFER = True
except:
    SURFER = False
    print('no pysurfer!')
import os
import os.path as op
import nibabel as nib
import mne.stats.cluster_level as mne_clusters
import mne
# from mne import spatial_tris_connectivity, grade_to_tris

import numpy as np
# import pickle
# import math
# import glob
# import fsfast
from sklearn.neighbors import BallTree
import shutil

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from src import utils
from src import freesurfer_utils as fu

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
FREE_SURFER_HOME = utils.get_link_dir(LINKS_DIR, 'freesurfer', 'FREESURFER_HOME')
BLENDER_ROOT_DIR = op.join(LINKS_DIR, 'mmvt')
FMRI_DIR = utils.get_link_dir(LINKS_DIR, 'fMRI')
os.environ['FREESURFER_HOME'] = FREE_SURFER_HOME
os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# SUBJECTS_DIR = '/homes/5/npeled/space3/subjects'
# # SUBJECTS_DIR = '/autofs/space/lilli_001/users/DARPA-MEG/freesurfs'
# # SUBJECTS_DIR =  '/home/noam/subjects/mri'
# # SUBJECT = 'ep001'
# os.environ['SUBJECTS_DIR'] = SUBJECTS_DIR
# ROOT_DIR = [f for f in ['/homes/5/npeled/space3/fMRI/MSIT', '/home/noam/fMRI/MSIT'] if op.isdir(f)][0]
# BLENDER_DIR = '/homes/5/npeled/space3/visualization_blender'
# FREE_SURFER_HOME = utils.get_exisiting_dir([os.environ.get('FREESURFER_HOME', ''),
#     '/usr/local/freesurfer/stable5_3_0', '/home/noam/freesurfer'])

conds = ['congruent-v-base', 'incongruent-v-base',  'congruent-v-incongruent', 'task.avg-v-base']
x, xfs = {}, {}
# show_fsaverage = False

# MRI_FILE_RH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.rh/congruent-v-base/sig.nii.gz'
# MRI_FILE_LH_FS = '/homes/5/npeled/space3/ECR_fsaverage/hc001/bold/congruence.sm05.lh/congruent-v-base/sig.nii.gz'

# fMRI_FILE = '/homes/5/npeled/space3/ECR/hc001/bold/congruence.sm05.{}/congruent-v-base/sig_mg79.mgz'

# x = nib.load('/homes/5/npeled/Desktop/sig_fsaverage.nii.gz')
# xfs = nib.load('/homes/5/npeled/Desktop/sig_subject.nii.gz')


def get_hemi_data(subject, hemi, source, surf_name='pial', name=None, sign="abs", min=None, max=None):
    brain = Brain(subject, hemi, surf_name, curv=False, offscreen=True)
    print('Brain {} verts: {}'.format(hemi, brain.geo[hemi].coords.shape[0]))
    hemi = brain._check_hemi(hemi)
    # load data here
    scalar_data, name = brain._read_scalar_data(source, hemi, name=name)
    print('fMRI contrast map vertices: {}'.format(len(scalar_data)))
    min, max = brain._get_display_range(scalar_data, min, max, sign)
    if sign not in ["abs", "pos", "neg"]:
        raise ValueError("Overlay sign must be 'abs', 'pos', or 'neg'")
    surf = brain.geo[hemi]
    old = viz.OverlayData(scalar_data, surf, min, max, sign)
    return old, brain


def save_fmri_colors(subject, hemi, fmri_file, surf_name, threshold=2):
    old, brain = get_hemi_data(subject, hemi, fmri_file.format(hemi), surf_name)
    x = old.mlab_data

    # Do some sanity checks
    verts, faces = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
    print('{}.pial.ply vertices: {}'.format(hemi, verts.shape[0]))
    if verts.shape[0] != brain.geo[hemi].coords.shape[0]:
        raise Exception("Brain and ply objects doesn't have the same verices number!")
    _save_fmri_colors(subject, hemi, x, threshold, verts=verts)


def _save_fmri_colors(subject, hemi, x, threshold, output_file, verts=None):
    if verts is None:
        verts, _ = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
    if len(x) != verts.shape[0]:
        raise Exception("fMRI contrast map and the hemi doens't have the same vertices number!")

    colors = utils.arr_to_colors_two_colors_maps(x, cm_big='YlOrRd', cm_small='PuBu',
        threshold=threshold, default_val=1)
    colors = np.hstack((x.reshape((len(x), 1)), colors))
    if output_file != '':
        op.join(BLENDER_SUBJECT_DIR, 'fmri_{}.npy'.format(hemi))
    np.save(output_file, colors)


def find_clusters(subject, contrast_name, atlas, load_from_annotation=False, n_jobs=1):
    input_fname = op.join(BLENDER_ROOT_DIR, SUBJECT, 'fmri', 'fmri_{}_{}.npy'.format(contrast_name, '{hemi}'))
    cluster_labels = {}
    for hemi in utils.HEMIS:
        fmri_fname = input_fname.format(hemi=hemi)
        if utils.file_type(input_fname) == 'npy':
            x = np.load(fmri_fname)
            contrast = x[:, 0]
        else:
            # try nibabel
            x = nib.load(fmri_fname)
            contrast = x.get_data().ravel()
        verts, faces = utils.read_ply_file(op.join(SUBJECTS_DIR, subject, 'surf', '{}.pial.ply'.format(hemi)))
        connectivity = mne.spatial_tris_connectivity(faces)
        clusters, _ = mne_clusters._find_clusters(contrast, 2, connectivity=connectivity)
        output_file = op.join(BLENDER_SUBJECT_DIR, 'fmri', 'blobs_{}_{}.npy'.format(contrast_name, hemi))
        save_clusters_for_blender(clusters, contrast, output_file)
        cluster_labels[hemi] = find_clusters_overlapped_labeles(
            subject, clusters, contrast, atlas, hemi, verts, load_from_annotation, n_jobs)
    utils.save(cluster_labels, op.join(BLENDER_ROOT_DIR, subject, 'fmri', 'clusters_labels_{}.npy'.format(contrast_name)))


def save_clusters_for_blender(clusters, contrast, output_file):
    vertices_num = len(contrast)
    data = np.ones((vertices_num, 4)) * -1
    colors = utils.get_spaced_colors(len(clusters))
    for ind, (cluster, color) in enumerate(zip(clusters, colors)):
        x = contrast[cluster]
        cluster_max = max([abs(np.min(x)), abs(np.max(x))])
        cluster_data = np.ones((len(cluster), 1)) * cluster_max
        cluster_color = np.tile(color, (len(cluster), 1))
        data[cluster, :] = np.hstack((cluster_data, cluster_color))
    np.save(output_file, data)


def find_clusters_overlapped_labeles(subject, clusters, contrast, atlas, hemi, verts, load_from_annotation=False, n_jobs=1):
    cluster_labels = []
    annot_fname = op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format(hemi, atlas))
    if load_from_annotation and op.isfile(annot_fname):
        labels = mne.read_labels_from_annot(subject, annot_fname=annot_fname, surf_name='pial')
    else:
        # todo: read only the labels from the current hemi
        labels = utils.read_labels_parallel(subject, SUBJECTS_DIR, atlas, n_jobs)
        labels = [l for l in labels if l.hemi == hemi]

    if len(labels) == 0:
        print('No labels!')
        return None
    for cluster in clusters:
        x = contrast[cluster]
        cluster_max = np.min(x) if abs(np.min(x)) > abs(np.max(x)) else np.max(x)
        inter_labels = []
        for label in labels:
            overlapped_vertices = np.intersect1d(cluster, label.vertices)
            if len(overlapped_vertices) > 0:
                if 'unknown' not in label.name:
                    inter_labels.append(dict(name=label.name, num=len(overlapped_vertices)))
        if len(inter_labels) > 0:
            max_inter = max([(il['num'], il['name']) for il in inter_labels])
            cluster_labels.append(dict(vertices=cluster, intersects=inter_labels, name=max_inter[1],
                coordinates=verts[cluster], max=cluster_max, hemi=hemi))
        else:
            print('No intersected labels!')
    return cluster_labels


def show_fMRI_using_pysurfer(subject, input_file, hemi='both'):
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    if hemi=='both':
        for hemi in ['rh', 'lh']:
            print('adding {}'.format(input_file.format(hemi=hemi)))
            brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)
    else:
        print('adding {}'.format(input_file.format(hemi=hemi)))
        brain.add_overlay(input_file.format(hemi=hemi), hemi=hemi)


def mri_convert_hemis(contrast_file_template, contrasts):
    for hemi in ['rh', 'lh']:
        for contrast in contrasts.keys():
            contrast_fname = contrast_file_template.format(hemi=hemi, contrast=contrast, format='{format}')
            mri_convert(contrast_fname, 'nii.gz', 'mgz')


def mri_convert(volume_fname, from_format='nii', to_format='mgz'):
    try:
        print('convert {} to {}'.format(volume_fname.format(format=from_format), volume_fname.format(format=to_format)))
        utils.run_script('mri_convert {} {}'.format(volume_fname.format(format=from_format),
                                                    volume_fname.format(format=to_format)))
    except:
        print('Error running mri_convert!')


def calculate_subcorticals_activity(volume_file, subcortical_codes_file='', aseg_stats_file_name='',
        method='max', k_points=100, do_plot=False):
    x = nib.load(volume_file)
    x_data = x.get_data()

    if do_plot:
        fig = plt.figure()
        ax = Axes3D(fig)

    sig_subs = []
    if subcortical_codes_file != '':
        subcortical_codes = np.genfromtxt(subcortical_codes_file, dtype=str, delimiter=',')
        seg_labels = map(str, subcortical_codes[:, 0])
    elif aseg_stats_file_name != '':
        aseg_stats = np.genfromtxt(aseg_stats_file_name, dtype=str, delimiter=',', skip_header=1)
        seg_labels = map(str, aseg_stats[:, 0])
    else:
        raise Exception('No segmentation file!')
    # Find the segmentation file
    aseg_fname = op.join(SUBJECTS_DIR, SUBJECT, 'mri', 'aseg.mgz')
    aseg = nib.load(aseg_fname)
    aseg_hdr = aseg.get_header()
    out_folder = op.join(SUBJECTS_DIR, SUBJECT, 'subcortical_fmri_activity')
    if not op.isdir(out_folder):
        os.mkdir(out_folder)
    sub_cortical_generator = utils.sub_cortical_voxels_generator(aseg, seg_labels, 5, False, FREE_SURFER_HOME)
    for pts, seg_name, seg_id in sub_cortical_generator:
        print(seg_name)
        verts, _ = utils.read_ply_file(op.join(SUBJECTS_DIR, SUBJECT, 'subcortical', '{}.ply'.format(seg_name)))
        vals = np.array([x_data[i, j, k] for i, j, k in pts])
        is_sig = np.max(np.abs(vals)) >= 2
        print(seg_name, seg_id, np.mean(vals), is_sig)
        pts = utils.transform_voxels_to_RAS(aseg_hdr, pts)
        # plot_points(verts,pts)
        verts_vals = calc_vert_vals(verts, pts, vals, method=method, k_points=k_points)
        print('verts vals: {}+-{}'.format(verts_vals.mean(), verts_vals.std()))
        if sum(abs(verts_vals)>2) > 0:
            sig_subs.append(seg_name)
        verts_colors = utils.arr_to_colors_two_colors_maps(verts_vals, threshold=2)
        verts_data = np.hstack((np.reshape(verts_vals, (len(verts_vals), 1)), verts_colors))
        np.save(op.join(out_folder, seg_name), verts_data)
        if do_plot:
            plot_points(verts, colors=verts_colors, fig_name=seg_name, ax=ax)
        # print(pts)
    utils.rmtree(op.join(BLENDER_SUBJECT_DIR, 'subcortical_fmri_activity'))
    shutil.copytree(out_folder, op.join(BLENDER_SUBJECT_DIR, 'subcortical_fmri_activity'))
    if do_plot:
        plt.savefig('/home/noam/subjects/mri/mg78/subcortical_fmri_activity/figures/brain.jpg')
        plt.show()


def calc_vert_vals(verts, pts, vals, method='max', k_points=100):
    ball_tree = BallTree(pts)
    dists, pts_inds = ball_tree.query(verts, k=k_points, return_distance=True)
    near_vals = vals[pts_inds]
    # sig_dists = dists[np.where(abs(near_vals)>2)]
    cover = len(np.unique(pts_inds.ravel()))/float(len(pts))
    print('{}% of the points are covered'.format(cover*100))
    if method=='dist':
        n_dists = 1/(dists**2)
        norm = 1/np.sum(n_dists, 1)
        norm = np.reshape(norm, (len(norm), 1))
        n_dists = norm * n_dists
        verts_vals = np.sum(near_vals * n_dists, 1)
    elif method=='max':
        verts_vals = near_vals[range(near_vals.shape[0]), np.argmax(abs(near_vals), 1)]
    return verts_vals


def plot_points(verts, pts=None, colors=None, fig_name='', ax=None):
    if ax is None:
        fig = plt.figure()
        ax = Axes3D(fig)
    colors = 'tomato' if colors is None else colors
    # ax.plot(verts[:, 0], verts[:, 1], verts[:, 2], 'o', color=colors, label='verts')
    ax.scatter(verts[:, 0], verts[:, 1], verts[:, 2], s=20, c=colors, label='verts')
    if pts is not None:
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], 'o', color='blue', label='voxels')
        plt.legend()
    if ax is None:
        plt.savefig('/home/noam/subjects/mri/mg78/subcortical_fmri_activity/figures/{}.jpg'.format(fig_name))
        plt.close()


def project_on_surface(subject, volume_file, colors_output_fname, surf_output_fname,
                       target_subject=None, threshold=2, overwrite_surf_data=False, overwrite_colors_file=False):
    for hemi in ['rh', 'lh']:
        print('project {} to {}'.format(volume_file, hemi))
        if not op.isfile(surf_output_fname.format(hemi=hemi)) or overwrite_surf_data:
            surf_data = fu.project_volume_data(volume_file, hemi, subject_id=subject, surf="pial", smooth_fwhm=3,
                target_subject=target_subject, output_fname=surf_output_fname.format(hemi=hemi))
            nans = np.sum(np.isnan(surf_data))
            if nans > 0:
                print('there are {} nans in {} surf data!'.format(nans, hemi))
            np.save(surf_output_fname.format(hemi=hemi), surf_data)
        else:
            surf_data = np.load(surf_output_fname.format(hemi=hemi))
        if not op.isfile(colors_output_fname.format(hemi=hemi)) or overwrite_colors_file:
            print('Calulating the activaton colors for {}'.format(surf_output_fname))
            _save_fmri_colors(target_subject, hemi, surf_data, threshold, colors_output_fname.format(hemi=hemi))
        shutil.copyfile(colors_output_fname.format(hemi=hemi), op.join(BLENDER_ROOT_DIR, subject, 'fmri',
            'fmri_'.format(op.basename(colors_output_fname.format(hemi=hemi)))))


def load_images_file(image_fname):
    for hemi in ['rh', 'lh']:
        x = nib.load(image_fname.format(hemi=hemi))
        nans = np.sum(np.isnan(np.array(x.dataobj)))
        if nans > 0:
            print('there are {} nans in {} image!'.format(nans, hemi))


def mask_volume(volume, mask, masked_volume):
    vol_nib = nib.load(volume)
    vol_data = vol_nib.get_data()
    mask_nib = nib.load(mask)
    mask_data = mask_nib.get_data().astype(np.bool)
    vol_data[mask_data] = 0
    vol_nib.data = vol_data
    nib.save(vol_nib, masked_volume)


def load_and_show_npy(subject, npy_file, hemi):
    x = np.load(npy_file)
    brain = Brain(subject, hemi, "pial", curv=False, offscreen=False)
    brain.toggle_toolbars(True)
    brain.add_overlay(x[:, 0], hemi=hemi)


def project_volue_to_surface(subject, data_fol, volume_name, target_subject='',
                             overwrite_surf_data=True, overwrite_colors_file=True):
    if target_subject == '':
        target_subject = subject
    volume_fname_template = op.join(data_fol, '{}.{}'.format(volume_name, '{format}'))
    if not op.isfile(volume_fname_template.format(format='mgz')) or overwrite_volume_mgz:
        mri_convert(volume_fname_template, 'mgh', 'mgz')
    volume_fname = volume_fname_template.format(format='mgz')
    shutil.copyfile(volume_fname, op.join(BLENDER_ROOT_DIR, subject, 'freeview', op.basename(volume_fname)))
    target_subject_prefix = '_{}'.format(target_subject) if subject != target_subject else ''
    colors_output_fname = op.join(data_fol, '{}{}_{}.npy'.format(volume_name, target_subject_prefix, '{hemi}'))
    surf_output_fname = op.join(data_fol, '{}{}_{}.mgz'.format(volume_name, target_subject_prefix, '{hemi}'))
        
    project_on_surface(target_subject, volume_fname, colors_output_fname, surf_output_fname,
                       target_subject, overwrite_surf_data, overwrite_colors_file)
    # fu.transform_mni_to_subject('colin27', data_fol, volume_fname, '{}_{}'.format(target_subject, volume_fname))
    # load_images_file(surf_output_fname)


if __name__ == '__main__':
    SUBJECT = 'colin27'
    os.environ['SUBJECT'] = SUBJECT
    TASK = 'ARC'#'MSIT'
    atlas = 'laus250'

    BLENDER_SUBJECT_DIR = op.join(BLENDER_ROOT_DIR, SUBJECT)

    contrast_name = 'interference'
    contrasts  ={'non-interference-v-base': '-a 1', 'interference-v-base': '-a 2',
                 'non-interference-v-interference': '-a 1 -c 2', 'task.avg-v-base': '-a 1 -a 2'}
    contrast_file_template = op.join(FMRI_DIR, TASK, SUBJECT, 'bold',
        '{contrast_name}.sm05.{hemi}'.format(contrast_name=contrast_name, hemi='{hemi}'), '{contrast}', 'sig.{format}')
    contrast = 'non-interference-v-interference'
    contrast_file = contrast_file_template.format(
        contrast=contrast, hemi='{hemi}', format='mgz')
    TR = 1.75

    # show_fMRI_using_pysurfer(SUBJECT, '/homes/5/npeled/space3/fMRI/ECR/hc004/bold/congruence.sm05.lh/congruent-v-incongruent/sig.mgz', 'rh')

    # fsfast.run(SUBJECT, root_dir=ROOT_DIR, par_file = 'msit.par', contrast_name=contrast_name, tr=TR, contrasts=contrasts, print_only=False)
    # fsfast.plot_contrast(SUBJECT, ROOT_DIR, contrast_name, contrasts, hemi='rh')
    # mri_convert_hemis(contrast_file_template, contrasts)


    overwrite_volume_mgz = False
    data_fol = op.join(FMRI_DIR, TASK, 'pp009')
    # contrast = 'pp009_ARC_High_Risk_Linear_Reward_contrast'
    contrast = 'pp009_ARC_PPI_highrisk_L_VLPFC'
    # project_volue_to_surface(SUBJECT, data_fol, contrast)
    find_clusters(SUBJECT, contrast, atlas,  load_from_annotation=True, n_jobs=6)



    # show_fMRI_using_pysurfer(SUBJECT, input_file=contrast_file, hemi='lh')
    # root = op.join('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003')
    # volume_file = op.join(root, 'sig.anat.mgz')
    # mask_file = op.join(root, 'VLPFC.mask.mgz')
    # masked_file = op.join(root, 'sig.anat.masked.mgz')
    # contrast_file = op.join(root, 'sig.{hemi}.mgz')
    # contrast_masked_file = op.join(root, 'sig.masked.{hemi}.mgz')

    # for hemi in ['rh', 'lh']:
    #     save_fmri_colors(SUBJECT, hemi, contrast_masked_file.format(hemi=hemi), 'pial', threshold=2)
    # Show the fRMI in pysurfer
    # show_fMRI_using_pysurfer(SUBJECT, input_file=contrast_masked_file, hemi='both')

    # load_and_show_npy(SUBJECT, '/homes/5/npeled/space3/visualization_blender/mg79/fmri_lh.npy', 'lh')

    # mask_volume(volume_file, mask_file, masked_file)
    # show_fMRI_using_pysurfer(SUBJECT, input_file='/autofs/space/franklin_003/users/npeled/fMRI/MSIT/pp003/sig.{hemi}.masked.mgz', hemi='both')
    # calculate_subcorticals_activity('/homes/5/npeled/space3/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/autofs/space/franklin_003/users/npeled/MSIT/mg78/aseg_stats.csv')
    # calculate_subcorticals_activity('/home/noam/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig.anat.mgh',
    #              '/home/noam/fMRI/MSIT/mg78/aseg_stats.csv')
    # volume_file = nib.load('/autofs/space/franklin_003/users/npeled/fMRI/MSIT/mg78/bold/interference.sm05.mni305/non-interference-v-interference/sig_subject.mgz')
    # vol_data, vol_header = volume_file.get_data(), volume_file.get_header()

    # contrast_file=contrast_file_template.format(
    #     contrast='non-interference-v-interference', hemi='mni305', format='mgz')
    # calculate_subcorticals_activity(volume_file, subcortical_codes_file=op.join(BLENDER_DIR, 'sub_cortical_codes.txt'),
    #     method='dist')

    # SPM_ROOT = '/homes/5/npeled/space3/spm_subjects'
    # for subject_fol in utils.get_subfolders(SPM_ROOT):
    #     subject = utils.namebase(subject_fol)
    #     print(subject)
    #     contrast_masked_file = op.join(subject_fol, '{}_VLPFC_{}.mgz'.format(subject, '{hemi}'))
    #     show_fMRI_using_pysurfer(SUBJECT, input_file=contrast_masked_file, hemi='rh')
    # brain = Brain('fsaverage', 'both', "pial", curv=False, offscreen=False)

    print('finish!')




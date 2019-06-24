import numpy as np
from numpy.ma import masked_array
import glob
import os.path as op
import re
import mne

from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def calc_powers_abs_minmax(powers, label_norm_powers_files=None, both_min_and_max=True):

    # -- Using mean of lablels
    # labels_norm_powers = np.concatenate(
    #     [np.mean(np.load(f), axis=0, keepdims=True) for f in label_norm_powers_files])
    # powers_min = np.min(labels_norm_powers, axis=0)
    # powers_max = np.max(labels_norm_powers, axis=0)

    # -- Using the same vertice
    # min_vertice = np.argmin(np.min(powers, axis=(1, 2)))
    # max_vertice = np.argmax(np.max(powers, axis=(1, 2)))
    # powers_max = np.max(powers, axis=(1, 2))[max_vertice]
    # powers_min = np.min(powers, axis=(1, 2))[min_vertice]
    # minmax_vertice = max_vertice # if powers_max > abs(powers_min) else min_vertice
    # print('max vertice {} -> {}'.format(minmax_vertice, powers_max))
    # print(np.unravel_index(powers[minmax_vertice].argmax(), powers[minmax_vertice].shape))
    # return powers[minmax_vertice]

    # -- Can be different vertice for each time and freq
    max_vertices = powers.reshape(powers.shape[0], -1).argmax(0).reshape((powers.shape[1], -1))
    min_vertices = powers.reshape(powers.shape[0], -1).argmin(0).reshape((powers.shape[1], -1))
    powers_min = np.min(powers, axis=0)  # over vertices
    powers_max = np.max(powers, axis=0)  # over vertices
    print('minmin: {}, maxmax: {}'.format(np.min(powers_min), np.max(powers_max)))
    if both_min_and_max:
        return powers_min, powers_max, min_vertices, max_vertices
    else:
        min_indices = np.where(np.abs(powers_min) > powers_max)
        powers_abs_minmax = powers_max
        powers_abs_minmax[min_indices] = powers_min[min_indices]
        return powers_abs_minmax


def concatenate_powers(fol, return_file_names=False):
    print('Concatenate powers in {}'.format(fol))
    powers_files = glob.glob(op.join(fol, 'epilepsy_*_induced_power.npy'))
    if len(powers_files) == 0:
        print('No files in {}!'.format(fol))
        return None
    # if len(powers_files) != 62: # Should calc number of lables
    #     print('{}: Not all the files were created!'.format(fol))
    #     return None
    powers = np.concatenate([np.load(powers_fname).astype(np.float32) for powers_fname in powers_files])
    if return_file_names:
        return powers, powers_files
    else:
        return powers


def calc_masked_negative_and_positive_powers(norm_powers_min, norm_powers_max, percentiles=[5, 95]):
    minmax_powers = np.zeros(norm_powers_max.shape)
    min_inds = np.where(norm_powers_min < np.percentile(norm_powers_min, percentiles[0]))
    max_inds = np.where(norm_powers_max > np.percentile(norm_powers_max, percentiles[1]))
    minmax_powers[max_inds] = norm_powers_max[max_inds]
    minmax_powers[min_inds] = norm_powers_min[min_inds]
    negative_powers = masked_array(minmax_powers, minmax_powers < np.percentile(norm_powers_min, percentiles[0]))
    positive_powers = masked_array(minmax_powers, minmax_powers > np.percentile(norm_powers_max, percentiles[1]))
    return negative_powers, positive_powers


def get_window_times(window_fname, downsample=2):
    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % downsample == 0 else \
        evoked.times[:-(downsample - 1)]
    return utils.downsample(times, downsample)


def nans(shape, dtype=np.float32):
    x = np.empty(shape, dtype)
    x.fill(np.nan)
    return x


def calc_vertices_lookup_tables(subject, modality, window, inverse_method, labels, inv):
    output_fname = op.join(MMVT_DIR, subject, 'vertices_lookup_tables.pkl')
    # if op.isfile(output_fname):
    #     vertices_ind_to_no_lookup, vertices_no_to_ind_lookup, vertices_labels_lookup = utils.load(output_fname)
    #     return vertices_lookup, vertices_labels_lookup
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    powers_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, window))
    powers_files = glob.glob(op.join(powers_fol, 'epilepsy_*_induced_power.npy'))
    start_ind = 0
    vertice_label = None
    vertices_ind_to_no_lookup = {}
    vertices_labels_lookup = {}
    for file_ind, powers_fname in enumerate(powers_files):
        label_name = utils.namebase(powers_fname).split('_')[1]
        label = [l for l in labels if l.name == label_name][0]
        vertno, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        for vert_ind, vert_num in zip(range(start_ind, start_ind + len(src_sel)), vertno[0] if label.hemi == 'lh' else vertno[1]):
            vertices_ind_to_no_lookup[vert_ind] = vert_num
            vertices_labels_lookup[vert_ind] = label
        # if start_ind <= vertices_ind < start_ind + len(src_sel):
        #     vertice_label = label
        #     break
        start_ind += len(src_sel)
    utils.save((vertices_ind_to_no_lookup, vertices_labels_lookup), output_fname)
    return vertices_ind_to_no_lookup, vertices_labels_lookup


def find_vertices(subject, run_num, modality='meg', atlas='aparc.DKTatlas'):
    from src.utils import labels_utils as lu
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    vertices = []
    for ind, label in enumerate(labels):
        _, vertno = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        vertices.extend(vertno)
    np.save(op.join(MMVT_DIR, subject, 'labels_verts.npy'), np.array(vertices))
    return vertices


def fix_amplitude_fnames(subject, bands):
    stcs_files = glob.glob(op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*-zvals-?h.stc'.format(subject))) + \
                 glob.glob(op.join(MMVT_DIR, subject, 'eeg', '{}-epilepsy-*-zvals-?h.stc'.format(subject)))
    for stc_fname in stcs_files:
        stc_name = utils.namebase(stc_fname)[len('{}-epilepsy-'.format(subject)):-len('-zvals-lh')]
        if stc_name.endswith('amplitude'):
            continue
        if not any([stc_name.endswith(band) for band in bands]):
            stc_name += '_amplitude'
            stc_end = '-zvals-lh.stc' if stc_fname.endswith('-zvals-lh.stc') else '-zvals-rh.stc'
            new_stc_fname = op.join(
                utils.get_parent_fol(stc_fname), '{}-epilepsy-{}{}'.format(subject, stc_name, stc_end))
            print('{} -> {}'.format(stc_fname, new_stc_fname))
            utils.rename_files(stc_fname, new_stc_fname)


def add_run_number_to_files(subject, run):
    run_num = re.sub('\D', ',', run).split(',')[-1]
    files = glob.glob(op.join(MEG_DIR, subject, '{}-epilepsy-*.fif'.format(subject))) + \
            glob.glob(op.join(EEG_DIR, subject, '{}-epilepsy-*.fif'.format(subject)))
            # glob.glob(op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*.fif'.format(subject))) + \
            # glob.glob(op.join(MMVT_DIR, subject, 'eeg', '{}-epilepsy-*.fif'.format(subject)))
    for fname in files:
        new_fname = fname.replace('-epilepsy-', '-epilepsy{}-'.format(run_num))
        print('{} - > {}'.format(fname, new_fname))
        utils.rename_files(fname, new_fname)


def create_evokeds_links(subject, windows):
    fol = utils.make_dir(op.join(MMVT_DIR, subject, 'evoked'))
    for window_fname in windows:
        new_window_fname = op.join(fol, utils.namebase_with_ext(window_fname))
        if op.isfile(new_window_fname) or op.islink(new_window_fname):
            continue
        utils.make_link(window_fname, new_window_fname)


def move_non_zvals_stcs(subject, modality):
    # Move not zvals stc files
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    non_zvlas_fol = utils.make_dir(op.join(modality_fol, 'non-zvals'))
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*.stc'))
                 if '-epilepsy-' in utils.namebase(f) and not '-zvals-' in utils.namebase(f)]
    for stc_fname in stc_files:
        utils.move_file(stc_fname, non_zvlas_fol, overwrite=True)

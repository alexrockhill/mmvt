from src.preproc import eeg
from src.preproc import meg
from src.utils import utils
import glob
import os.path as op
import mne
import numpy as np
import re

from src.examples.epilepsy import utils as epi_utils
from src.examples.epilepsy import plots
from src.examples.epilepsy import power_spectrums_plots as psplots

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels, overwrite_inv=False,
                 overwrite_fwd=False):
    # python -m src.preproc.eeg -s nmr00857 -f calc_inverse_operator,make_forward_solution
    #     --overwrite_inv 0 --overwrite_fwd 0 -t epilepsy
    #     --raw_fname  /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_01_raw.fif
    #     --empty_fname /autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123/5241495_roomnoise_raw.fif
    #     --use_empty_room_for_noise_cov 1
    #     --bad_channels EEG061,EEG02,EEG042,MEG0112,MEG0113
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_inverse_operator,make_forward_solution',
        task='epilepsy',
        inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
        fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
        fwd_usingEEG=modality in ['eeg', 'meeg'],
        overwrite_inv=overwrite_inv,
        overwrite_fwd=overwrite_fwd,
        use_empty_room_for_noise_cov=True,
        bad_channels=bad_channels,
        raw_fname=raw_fname,
        empty_fname=empty_fname,
    ))
    module.call_main(args)


def check_inv_fwd(subject, modality, run_num):
    import mne.minimum_norm
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    fwd_fname = op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality))
    fwd = mne.read_forward_solution(fwd_fname)
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    fwd_meg_channels = [c for c in fwd['sol']['row_names'] if c.startswith('MEG')]
    fwd_eeg_channels = [c for c in fwd['sol']['row_names'] if c.startswith('EEG')]
    inv_meg_channels = [c for c in inv['info']['ch_names'] if c.startswith('MEG')]
    inv_eeg_channels = [c for c in inv['info']['ch_names'] if c.startswith('EEG')]
    print('{}: using {}/{} EEG sensors and {}/{} MEG sensors'.format(
        modality, len(inv_eeg_channels), len(fwd_eeg_channels), len(inv_meg_channels), len(fwd_meg_channels)))


def calc_amplitude(subject, modality, run_num, windows_fnames, inverse_method='dSPM', overwrite=False, n_jobs=4):
    params = [(subject, window_fname, modality, run_num, windows_fnames, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_parallel, params, n_jobs)


def _calc_amplitude_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc -i dSPM -t epilepsy
    #   --evo_fname /autofs/space/frieda_001/users/valia/mmvt_root/meg/00857_EPI/sz_evolution/43.9s.fif
    #   --overwrite_stc 1
    subject, window_fname, modality, run_num, windows_fnames, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_stc',
        task='epilepsy',
        inverse_method=inverse_method,
        inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
        fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
        fwd_usingEEG=modality in ['eeg', 'meeg'],
        evo_fname=window_fname,
        overwrite_stc=overwrite,
        n_jobs=1,
    ))
    module.call_main(args)


def calc_amplitude_zvals(subject, windows_fnames, baseline_name, modality, from_index=None, to_index=None,
                         inverse_method='dSPM', parallel=True, overwrite=False):
    params = [(subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_amplitude_zvals_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_amplitude_zvals_parallel(p):
    # python3 -m src.preproc.meg -s nmr00857 -f calc_stc_zvals --stc_name nmr00857-epilepsy-dSPM-meeg-43.9s
    #   --baseline_stc_name nmr00857-epilepsy-dSPM-meeg-37.3_BGprSzs --use_abs 1 --overwrite_stc 1
    subject, modality, window_fname, baseline_name, from_index, to_index, inverse_method, overwrite = p
    module = eeg if modality == 'eeg' else meg
    window = utils.namebase(window_fname)
    stc_template = '{}-epilepsy-{}-{}-{}{}'.format(subject, inverse_method, modality, '{window}', '{suffix}')
    window_stc_name = stc_template.format(window=window, suffix='')
    args = module.read_cmd_args(dict(
        subject=subject,
        mri_subject=subject,
        function='calc_stc_zvals',
        task='epilepsy',
        stc_name=window_stc_name,
        baseline_stc_name=stc_template.format(window=baseline_name, suffix=''),
        stc_zvals_name=stc_template.format(window=window, suffix='_amplitude-zvals'),
        from_index=from_index,
        to_index=to_index,
        use_abs=1,
        overwrite_stc=overwrite
    ))
    module.call_main(args)


def calc_sensors_power(subject, windows_fnames, modality, inverse_method='dSPM', bad_channels=[],
                       high_gamma_max=120, downsample=2, parallel=False, overwrite=False):
    params = [(subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_sensors_power_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_sensors_power_parallel(p):
    from mne.time_frequency import tfr_array_morlet

    subject, window_fname, modality, inverse_method, bad_channels, downsample, high_gamma_max, overwrite = p

    root_dir = utils.make_dir(op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject))
    output_fname_template = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    # bad_channels = bad_channels.split(',')
    n_cycles = freqs / 2.

    window = utils.namebase(window_fname)
    output_fname = output_fname_template.format(window=window)
    if op.isfile(output_fname) and not overwrite:
        print('{} already exist'.format(output_fname))
    evoked = mne.read_evokeds(window_fname)[0]
    if modality == 'eeg':
        picks = mne.pick_types(evoked.info, meg=False, eeg=True, exclude=bad_channels)
    elif modality == 'meg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=False, exclude=bad_channels)
    elif modality == 'meeg':
        picks = mne.pick_types(evoked.info, meg=True, eeg=True, exclude=bad_channels)
    else:
        raise Exception('Wrong modality!')

    evoked_data = evoked.data[np.newaxis, picks, :]
    powers = tfr_array_morlet(
        evoked_data, sfreq=evoked.info['sfreq'], freqs=freqs, n_cycles=n_cycles, output='power')
    powers = np.squeeze(powers)
    if powers.shape[2] % 2 == 1:
        powers = powers[:, :, :-1]
    if downsample > 1:
        powers = utils.downsample_3d(powers, downsample)
    powers_db = 10 * np.log10(powers)  # dB/Hz should be baseline corrected!!!
    print('Saving {}'.format(output_fname))
    np.save(output_fname, powers_db.astype(np.float16))


def calc_induced_power(subject, run_num, windows_fnames, modality, inverse_method='dSPM', check_for_labels_files=True,
                       overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    module = eeg if modality == 'eeg' else meg
    output_fname = op.join(MMVT_DIR, 'eeg' if modality == 'eeg' else 'meg', '{}-epilepsy-{}-{}-{}_{}'.format(
        subject, inverse_method, modality, '{window}', '{band}'))
    for window_fname in windows_fnames:
        if all([utils.stc_exist(output_fname.format(window=utils.namebase(window_fname), band=band))
                for band in bands]) and not overwrite:
            continue
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, utils.namebase(window_fname)))
        if op.isdir(fol) and not check_for_labels_files:
            print('{} already exist'.format(fol))
            continue
        args = module.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc',
            task='epilepsy',
            inverse_method=inverse_method,
            inv_fname=op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality)),
            fwd_fname=op.join(root_dir, '{}-epilepsy{}-{}-fwd.fif'.format(subject, run_num, modality)),
            calc_source_band_induced_power=True,
            fwd_usingEEG=modality in ['eeg', 'meeg'],
            evo_fname=window_fname,
            n_jobs=1,
            overwrite_stc=overwrite
        ))
        module.call_main(args)


def calc_max_powers(subject, windows_fnames, modality, inverse_method='dSPM', overwrite=False, parallel=True):
    params = [(subject, window_fname, modality, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_max_powers_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_max_powers_parallel(p):
    subject, window_fname, modality, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_max_power.npy'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if op.isfile(output_fname) and not overwrite:
        return
    fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if not op.isdir(fol):
        print('{} does not exist!'.format(fol))
        return
    powers_files = glob.glob(op.join(fol, 'epilepsy_*_induced_power.npy'))
    if len(powers_files) != 62: # Should calc number of lables
        print('{}: Not all the files were created!'.format(fol))
        return
    print('Calculating max power for {}'.format(fol))
    max_powers = np.max(epi_utils.concatenate_powers(fol), axis=0)
    print('Saving to {}'.format(output_fname))
    np.save(output_fname, max_powers)


def calc_induced_power_zvals(
        subject, windows_fnames, baseline_name, modality, bands, from_index=None, to_index=None, inverse_method='dSPM',
        parallel=True, overwrite=False):
    params = [(subject, modality, window_fname, baseline_name, bands, from_index, to_index, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_calc_induced_power_zvals_parallel, params, len(windows_fnames) if parallel else 1)


def _calc_induced_power_zvals_parallel(p):
    subject, modality, window_fname, baseline_name, bands, from_index, to_index, inverse_method, overwrite = p
    module = eeg if modality == 'eeg' else meg
    stc_template = '{}-epilepsy-{}-{}-{}_{}'.format(subject, inverse_method, modality, '{window}', '{band}')
    root_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    if all([utils.stc_exist(op.join(root_fol, '{}-zvals'.format(
            stc_template.format(window=utils.namebase(window_fname), band=band)))) \
            for band in bands]) and not overwrite:
        return
    for band in bands:
        window_stc_name = stc_template.format(window=utils.namebase(window_fname), band=band)
        args = module.read_cmd_args(dict(
            subject=subject,
            mri_subject=subject,
            function='calc_stc_zvals',
            task='epilepsy',
            stc_name=window_stc_name,
            baseline_stc_name=stc_template.format(window=baseline_name, band=band),
            from_index=from_index,
            to_index=to_index,
            use_abs=1,
            overwrite_stc=overwrite
        ))
        module.call_main(args)


def calc_stc_power_specturm(subject, modality, window_fname, baseline_window, run_num, inverse_method='dSPM',
                            atlas='aparc.DKTatlas', high_gamma_max=120):
    from src.utils import labels_utils as lu
    from collections import defaultdict
    import time

    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    window = utils.namebase(window_fname)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
        subject, inverse_method, modality, '{window}'))
    window_output_fname = output_fname.format(window=window)
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])

    d = np.load(window_output_fname)
    min_indeices, max_indices = d['min_vertices'], d['max_vertices']
    times = epi_utils.get_window_times(window_fname, downsample=2)
    t_max = np.where(times > 0.15)[0][0]
    t_min = np.where(times > 0)[0][0]
    x = d['max'][:12, t_min:t_max]
    max_val = np.max(x)
    f, t = np.unravel_index(x.argmax(), x.shape)
    t += t_min
    vertices_ind = max_indices[f, t]

    inv_fname = op.join(root_dir, '{}-epilepsy{}-{}-inv.fif'.format(subject, run_num, modality))
    inv = mne.minimum_norm.read_inverse_operator(inv_fname)
    labels = lu.read_labels(subject, SUBJECTS_DIR, atlas)
    # max_vert_label = find_vert_label(vertices_ind, modality, window, labels, inv)
    vertices_ind_to_no_lookup, vertices_labels_lookup = \
        epi_utils.calc_vertices_lookup_tables(subject, modality, window, inverse_method, labels, inv)
    max_vert_label = vertices_labels_lookup[vertices_ind]
    print('{:.4f} in {}Hz {:.4f}s in {}'.format(max_val, freqs[f], times[t], max_vert_label.name))

    labels_norm_data_fol = utils.make_dir(op.join(root_dir, 'labels_norm_all_baseline'))
    baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(baseline_window)))
    baseline_files = glob.glob(op.join(baseline_fol, 'epilepsy_*_induced_power.npy'))
    powers_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, window))
    powers_files = glob.glob(op.join(powers_fol, 'epilepsy_*_induced_power.npy'))
    all_files_exist = True
    for powers_fname in powers_files:
        label_name = utils.namebase(powers_fname).split('_')[1]
        label_fname = op.join(labels_norm_data_fol, '{}-epilepsy-{}-{}-{}-{}-norm-induced_power.npy'.format(
            subject, inverse_method, modality, window, label_name))
        all_files_exist = all_files_exist and op.isfile(label_fname)

    if not all_files_exist:
        baseline = epi_utils.concatenate_powers(baseline_fol) # (vertices x freqs x time)
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the mean (over time) of log baseline values
        powers = epi_utils.concatenate_powers(powers_fol)
        norm_powers = (powers - baseline_mean) / baseline_std

    # norm_labels_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
    #     subject, inverse_method, modality, window))
    vertices, vertices_data = defaultdict(list), defaultdict(list)
    # labels_data = []
    start_ind = 0
    now = time.time()
    for file_ind, (baseline_fname, powers_fname) in enumerate(zip(baseline_files, powers_files)):
        if not all_files_exist:
            utils.time_to_go(now, file_ind, len(baseline_files), 1)
        baseline_label_name = utils.namebase(baseline_fname).split('_')[1]
        powers_label_name = utils.namebase(powers_fname).split('_')[1]
        if baseline_label_name != powers_label_name:
            raise Exception('ASDGF@#Q%EGF#Q$T')
        label_name = powers_label_name
        label = [l for l in labels if l.name == label_name][0]
        label_fname = op.join(labels_norm_data_fol, '{}-epilepsy-{}-{}-{}-{}-norm-induced_power.npy'.format(
            subject, inverse_method, modality, window, label_name))
        vertno, src_sel = mne.minimum_norm.inverse.label_src_vertno_sel(label, inv['src'])
        if not op.isfile(label_fname):
            print('Reading norm_powers ({}) from {}:{} for label {}'.format(
                norm_powers.shape[0], start_ind, start_ind+len(src_sel), label.name))
            label_powers = norm_powers[start_ind: start_ind+len(src_sel)]
            np.save(label_fname, label_powers)
        else:
            label_powers = np.load(label_fname, mmap_mode='r')
        labels_powers_per_freq_per_time = label_powers[:, f, t]
        if start_ind <= vertices_ind < start_ind+len(src_sel):
            print('vertices_ind: label {}, for {}Hz and {:.3f}s: {}'.format(
                label_name, freqs[f], times[t], label_powers[vertices_ind - start_ind, f, t]))
            if label_powers[vertices_ind - start_ind, f, t] != max_val:
                raise Exception('@#%$#$%@#$%@#$%@%$')
        if np.isclose(max_val, np.max(labels_powers_per_freq_per_time)):
            print('isclose: label {}, for {}Hz and {:.3f}s: {}'.format(
                label_name, freqs[f], times[t], np.max(labels_powers_per_freq_per_time)))
        start_ind += len(src_sel)
        vertices[label.hemi].extend(vertno[0] if label.hemi == 'lh' else vertno[1])
        vertices_data[label.hemi].extend(labels_powers_per_freq_per_time)
        del label_powers

    combined_stc = meg.creating_stc_obj(
        vertices_data, vertices, subject, tmin=times[0], tstep=times[1] - times[0])
    output_stc_fname = op.join(
        MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg',
        'epilepsy-{}-{}-{:.3f}s-{}Hz-{}-induced_norm_power'.format(
            modality, window, times[t], freqs[f], inverse_method))
    print('Saving stc file for t {} and f {} in {}'.format(t, f, output_stc_fname))
    combined_stc.save(output_stc_fname)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_template,
         inverse_method='dSPM', specific_window='', no_runs=False, n_jobs=4):
    # run_num = re.sub('\D', ',', run).split(',')[-1].zfill(2)
    if no_runs:
        windows = glob.glob(op.join(evokes_fol, '*.fif'))
        baseline_windows = glob.glob(op.join(evokes_fol, '*{}*.fif'.format(baseline_template)))
    else:
        windows = glob.glob(op.join(evokes_fol, '{}_*.fif'.format(run)))
        baseline_windows = glob.glob(op.join(evokes_fol, '{}_{}*.fif'.format(run, baseline_template)))
    for baseline_window in baseline_windows:
        if baseline_window in windows:
            windows.remove(baseline_window)
    if specific_window != '':
        windows = [w for w in windows if specific_window in utils.namebase(w)]
        if len(windows) == 0:
            print('No windows!')
            return
    baseline_window = utils.select_one_file(baseline_windows, 'baseline')
    baseline_windows = [baseline_window]
    windows_with_baseline = windows + baseline_windows
    baseline_name = utils.namebase(baseline_window)
    overwrite_inv = False
    overwrite_fwd = False
    overwrite_evokes = True
    overwrite_plots = False
    check_for_labels_files = False
    overwrite_induced_power_zvals = False
    overwrite_stc = False
    overwrite_modalities_figures = False
    from_index, to_index = 2000, 10000
    max_t = 0 #7500
    high_gamma_max = 120
    percentiles = [5, 95]
    sig_threshold = 2
    figures_type = 'jpg'
    save_fig = False
    plot_baseline_stat = False
    bad_channels = bad_channels.split(',')

    # create_evokeds_links(subject, windows_with_baseline)
    for modality in modalities:
        # 1) Sensors
        # plots.plot_evokes(subject, modality, windows, bad_channels, n_jobs > 1, overwrite_evokes)
        # plots.plot_topomaps(subject, modality, windows, bad_channels, parallel=n_jobs > 1)

        # calc_sensors_power(subject, windows_with_baseline, modality, inverse_method, bad_channels,
        #                    high_gamma_max=high_gamma_max, downsample=2, parallel=n_jobs > 1, overwrite=True)
        # psplots.plot_sensors_powers(
        #     subject, windows, baseline_window, modality, inverse_method, high_gamma_max=high_gamma_max,
        #     percentiles=percentiles, sig_threshold=sig_threshold, save_fig=save_fig,
        #     plot_baseline_stat=plot_baseline_stat, bad_channels=bad_channels, overwrite=True, parallel=False)

        # 2) calc fwd and inv
        # calc_fwd_inv(subject, modality, run_num, raw_fname, empty_fname, bad_channels,
        #              overwrite_inv=overwrite_inv, overwrite_fwd=overwrite_fwd)
        # check_inv_fwd(subject, modality, run_num)

        # 3) Amplitude
        # calc_amplitude(subject, modality, run_num, windows_with_baseline, inverse_method, overwrite_stc, n_jobs)
        # calc_amplitude_zvals(
        #     subject, windows, baseline_name, modality, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)

        # 4) Induced power
        # calc_induced_power(subject, run_num, windows_with_baseline, modality, inverse_method, check_for_labels_files,
        #                    overwrite=True)
        # psplots.plot_powers(subject, windows, modality, inverse_method, high_gamma_max, figures_type,
        #         overwrite=False)
        # psplots.plot_baseline_source_powers(
        #     subject, baseline_window, modality, inverse_method, high_gamma_max, figures_type, overwrite_plots)
        psplots.plot_norm_powers(
            subject, windows, baseline_window, modality, inverse_method, overwrite=True, figures_type=figures_type)
        # psplots.plot_norm_powers_per_label(subject, windows, baseline_window, modality, inverse_method,
        #                            calc_also_non_norm_powers=False, overwrite=True, n_jobs=n_jobs)
        # calc_stc_power_specturm(subject, modality, windows[0], baseline_window, run_num)
        pass

    # find_vertices(subject, run_num)
    # for window_fname in windows:
    #     figure_name = '' #'{}-modalities-power-spectrum-with-grpahs'.format(utils.namebase(window_fname))
    #     psplots.plot_modalities_power_spectrums_with_graph(subject, modalities, window_fname, figure_name, file_type='eps')

    # files = glob.glob('/autofs/space/thibault_001/users/npeled/EEG/nmr01321/nmr01321-epilepsy-dSPM-eeg-run1_szMEG_213s-induced_power/epilepsy_*_induced_norm_power.npy')
    # calc_powers_abs_minmax(None, label_norm_powers_files=files)

    # Old stuff
    # for modality in modalities:
        # calc_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=True)
        # plot_max_powers(subject, windows_with_baseline, modality, inverse_method, overwrite=False, parallel=False)
        # calc_induced_power_zvals(
        #     subject, windows, baseline_name, modality, bands, from_index, to_index, inverse_method,
        #     parallel=n_jobs > 1, overwrite=overwrite_induced_power_zvals)
        # move_non_zvals_stcs(subject, modality)
        # plot_stcs_files(subject, modality, n_jobs)
        # plot_windows(subject, windows, modality, bands, inverse_method)
        # plot_freqs(subject, temporal_windows, modality, bands, inverse_method, max_t)

    # plot_modalities(subject, windows, modalities, bands, inverse_method, max_t, overwrite_modalities_figures, n_jobs)
    # plot_activity_modalities(subject, windows, modalities, inverse_method, overwrite=overwrite_modalities_figures)
    # plot_baseline(subject, baseline_name)
    # fix_amplitude_fnames(subject, bands)


if __name__ == '__main__':
    from src.examples.epilepsy import init_files

    modalities = ['meg', 'eeg', 'meeg']
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma', 'high_gamma']
    inverse_method = 'dSPM'
    subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name, no_runs = init_files.subject_nmr01325()
    run_files = [utils.namebase(f).split('_')[0] for f in glob.glob(op.join(evokes_fol, 'run*_*.fif'))]
    evokes_files = glob.glob(op.join(evokes_fol, '*.fif'))
    runs = []
    if len(run_files) == len(evokes_files):
        runs = set(run_files)
    if len(runs) == 0 or no_runs:
        print('No run were found!')
        runs = ['01']
    n_jobs = 1 # utils.get_n_jobs(-5)
    print('n_jobs: {}'.format(n_jobs))
    specific_window = 'sz_1.3s' # '550_20sec'#  #'bl_474s' #  #' # 'sz_1.3s' #'550_20sec' #  'bl_474s' # 'run2_bl_248s'
    for run in runs:
        # if run != 'run1':
        #     continue
        raw_fname, run_num = init_files.find_raw_fname(meg_fol, run)
        if len(runs) > 0:
            if not op.isfile(raw_fname):
                continue
        main(subject, run, modalities, bands, evokes_fol, raw_fname, empty_fname, bad_channels, baseline_name,
             inverse_method, specific_window, no_runs, n_jobs)
    print('Finish!')
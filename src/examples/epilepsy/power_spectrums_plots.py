import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import os.path as op
import numpy as np
import glob
import mne

from src.utils import utils
from src.examples.epilepsy import utils as epi_utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def plot_sensors_powers(subject, windows_fnames, baseline_window_fname, modality, inverse_method='dSPM',
                        high_gamma_max=120, percentiles=[5, 95], sig_threshold=2, plot_non_norm_powers=False,
                        plot_baseline_stat=False, save_fig=True, bad_channels=[], overwrite=False, parallel=True):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    input_fname_template = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    norm_powers_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_norm_power_minmax.npz'.format(
        subject, inverse_method, modality, '{window}'))
    basealine_powers_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-sensors_power_minmax.npz'.format(
        subject, inverse_method, modality, '{window}'))
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'sensors-power-spectrum'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-{}-sensors-power.jpg'.format(
            subject, inverse_method, modality, '{window}', '{method}'))

    baseline_window = utils.namebase(baseline_window_fname)
    if not op.isfile(input_fname_template.format(window=baseline_window)):
        print('No baseline powers! {}'.format(input_fname_template.format(window=baseline_window)))
        return
    baseline_fname = basealine_powers_fname.format(window=utils.namebase(baseline_window))
    if op.isfile(baseline_fname) and not overwrite:
        d = np.load(baseline_fname)
        baseline_mean, baseline_std = d['mean'], d['std']
    else:
        baseline = np.load(input_fname_template.format(window=baseline_window)).astype(np.float32)
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time and vertices) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        np.savez(baseline_fname, mean=baseline_mean, std=baseline_std)
    if plot_baseline_stat and modality != 'meeg':
        plot_sensors_baseline_powers(
            baseline_mean, baseline_std, modality, baseline_window, baseline_window_fname, figures_template,
            bad_channels, high_gamma_max, overwrite)
    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        times = epi_utils.get_window_times(window_fname, downsample=2)
        fname = norm_powers_fname.format(window=window)
        if op.isfile(fname) and not overwrite:
            print('Loading {} ({})'.format(utils.namebase(fname), utils.file_modification_time(fname)))
            d = np.load(fname)
            norm_powers_min, norm_powers_max = d['min'], d['max']
        else:
            # dividing by the mean of baseline values, taking the log, and
            #           dividing by the standard deviation of log baseline values
            #           ('zlogratio')
            if not op.isfile(input_fname_template.format(window=window)):
                print('No window powers! {}'.format(input_fname_template.format(window=window)))
                continue
            powers = np.load(input_fname_template.format(window=window)).astype(np.float32)
            if plot_non_norm_powers and not op.isfile(figures_template.format(window=window, method='max')) \
                    and not overwrite:
                powers_max = np.max(powers, axis=0)  # over vertices
                plot_power_spectrum(
                    powers_max, times, figures_template.format(window=window, method='max'),
                    baseline_correction=False, high_gamma_max=high_gamma_max)

            if baseline_std.shape[1] < powers.shape[1]:
                print('baseline freqs dim is {} and powers freqs dim is {}!'.format(baseline.shape[1], powers.shape[1]))
                powers = powers[:, :baseline.shape[1], :]
            norm_powers = (powers - baseline_mean) # / baseline_std
            # norm_powers = np.diff(norm_powers, 5)
            # norm_powers = (norm_powers[:, :, 170:] - norm_powers[:, :, :-170])
            # times = times[:-170]
            norm_powers_min, norm_powers_max, min_vertices, max_vertices = epi_utils.calc_powers_abs_minmax(
                norm_powers, both_min_and_max=True)

            # freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
            # band_freqs = [1, 4]
            # idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            # band_power = np.mean(norm_powers_max[idx, :], axis=0)
            # plt.plot(band_power.T)
            # plt.show()


            # norm_powers_max = np.max(norm_powers, axis=0) # over vertices
            # norm_powers_min = np.min(norm_powers, axis=0)  # over vertices
            np.savez(fname, min=norm_powers_min, max=norm_powers_max)

        print('***** {} is baseline corrected using {} *******'.format(window, utils.namebase(baseline_window)))
        figure_fname = figures_template.format(window=window, method='pos_and_neg') if save_fig else ''
        plot_positive_and_negative_power_spectrum(
            norm_powers_min, norm_powers_max, times,  '{} {}'.format(modality, window),
            figure_fname=figure_fname, high_gamma_max=high_gamma_max, show_only_sig_in_graph=True)


def plot_sensors_baseline_powers(
        baseline_mean, baseline_std, modality, baseline_window, baseline_fname, figures_template,
        bad_channels=[], high_gamma_max=120, overwrite=False):
    info = mne.read_evokeds(baseline_fname)[0].info
    if modality == 'meg':
        sensors_picks = {
            sensor_type: mne.io.pick.pick_types(info, meg=sensor_type, eeg=False, exclude=bad_channels)
            for sensor_type in ['mag', 'grad']}
    else:
        sensors_picks = {sensor_type: mne.io.pick.pick_types(info, meg=False, eeg=True, exclude=bad_channels)
                         for sensor_type in ['eeg']}
    for sensors_type, sensors_idx in sensors_picks.items():
        sensors_idx = np.arange(len(np.unique(sensors_idx)))
        sensors_idx = np.argsort(baseline_mean[sensors_idx].squeeze().max(1))
        sensors = np.arange(len(sensors_idx))
        baseline_mean_figure_fname = figures_template.format(
            window=baseline_window, method='{}-mean'.format(sensors_type))
        baseline_std_figure_fname = figures_template.format(
            window=baseline_window, method='{}-std'.format(sensors_type))
        if not op.isfile(baseline_std_figure_fname) or overwrite:
            plot_power_spectrum(
                baseline_std[sensors_idx], sensors, baseline_std_figure_fname, xlabel='#sensor',
                baseline_correction=False, high_gamma_max=high_gamma_max)
        if not op.isfile(baseline_mean_figure_fname) or overwrite:
            plot_power_spectrum(
                baseline_mean[sensors_idx], sensors, baseline_mean_figure_fname, xlabel='#sensor',
                baseline_correction=False, high_gamma_max=high_gamma_max)


def plot_powers(subject, windows_fnames, modality, inverse_method='dSPM', high_gamma_max=120, figures_type='jpg',
                overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power.npz'.format(
        subject, inverse_method, modality, '{window}'))
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_power.{}'.format(
            subject, inverse_method, modality, '{window}', figures_type))
    for window_fname in windows_fnames:
        window_name = utils.namebase(window_fname)
        figure_fname = figures_template.format(window=utils.namebase(window_name))
        # if op.isfile(figure_fname) and not overwrite:
        #     continue
        if op.isfile(output_fname.format(window=window_name)) and not overwrite:
            d = np.load(output_fname.format(window=window_name))
            powers_max, times = d['max'], d['times']
        else:
            fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
                subject, inverse_method, modality, window_name))
            powers = epi_utils.concatenate_powers(fol) # (vertices x freqs x time)
            powers_max = np.max(powers, axis=0)  # over vertices

            times = epi_utils.get_window_times(window_fname, downsample=2)
            np.savez(output_fname.format(window=window_name),  max=powers_max, times=times)
        plot_power_spectrum(
            powers_max, times, figure_fname, baseline_correction=False, high_gamma_max=high_gamma_max)


def plot_baseline_source_powers(subject, baseline_fname, modality, inverse_method='dSPM', high_gamma_max=120,
                                figures_type='jpg', overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    baseline_name = utils.namebase(baseline_fname)
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-{}-induced_power.{}'.format(
            subject, inverse_method, modality, baseline_name, '{method}', figures_type))
    baseline_stat_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power_stat.npz'.format(
        subject, inverse_method, modality, baseline_name))
    if not op.isfile(baseline_stat_fname) or overwrite:
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, baseline_name))
        baseline = epi_utils.concatenate_powers(fol)  # (vertices x freqs x time)
        baseline_mean = np.mean(baseline, axis=2, keepdims=True)
        baseline_std = np.std(baseline, axis=2, keepdims=True)
        np.savez(baseline_stat_fname, mean=baseline_mean, std=baseline_std)
    else:
        d = np.load(baseline_stat_fname)
        baseline_mean, baseline_std = d['mean'], d['std']
    baseline_std_figure_fname = figures_template.format(method='std')
    baseline_mean_figure_fname = figures_template.format(method='mean')
    vertices = np.arange(baseline_mean.shape[0])
    vertices_idx = np.argsort(baseline_mean.squeeze().max(1))
    if True: # not op.isfile(baseline_std_figure_fname) or overwrite:
        plot_power_spectrum(
            baseline_std[vertices_idx], vertices, baseline_std_figure_fname, xlabel='#vertice',
            baseline_correction=False, high_gamma_max=high_gamma_max)
    if True: #not op.isfile(baseline_mean_figure_fname) or overwrite:
        plot_power_spectrum(
            baseline_mean[vertices_idx], vertices, baseline_mean_figure_fname, xlabel='#vertice',
            baseline_correction=False, high_gamma_max=high_gamma_max)


def plot_norm_powers(subject, windows_fnames, baseline_window, modality, inverse_method='dSPM', figures_type='jpg',
                     calc_also_non_norm_powers=False, use_norm_labels_powers=False, high_gamma_max=120, overwrite=False):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    if use_norm_labels_powers:
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power_labels.npy'.format(
            subject, inverse_method, modality, '{window}'))
    else:
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
    baseline_stat_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power_stat.npz'.format(
            subject, inverse_method, modality, utils.namebase(baseline_window)))
    not_norm_output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_minmax_power.npy'.format(
        subject, inverse_method, modality, '{window}'))
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figs_fol_not_norm = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum-not-norm'))
    figures_template = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_{}_norm_power.{}'.format(
            subject, inverse_method, modality, '{window}', '{method}', figures_type))
    figures_template_not_norm = op.join(figs_fol_not_norm, '{}-epilepsy-{}-{}-{}-induced_mean_norm_power.jpg'.format(
            subject, inverse_method, modality, '{window}'))
    # if not all([op.isfile(output_fname.format(window=utils.namebase(window_fname)))
    #             for window_fname in windows_fnames]) or overwrite:
    #          or not op.isfile(figures_template.format(window=utils.namebase(baseline_window)))
    if not op.isfile(baseline_stat_fname) or overwrite:
        baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
                subject, inverse_method, modality, utils.namebase(baseline_window)))
        baseline = epi_utils.concatenate_powers(baseline_fol) # (vertices x freqs x time)
        if baseline is None:
            return
        baseline_std = np.std(baseline, axis=2, keepdims=True) # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True) # the mean (over time) of log baseline values
        np.savez(baseline_stat_fname, mean=baseline_mean, std=baseline_std)
        # baseline_mean_over_vertices = np.mean(baseline, axis=0)
        # plot_power_spectrum(baseline_mean_over_vertices, figures_template.format(window=utils.namebase(baseline_window)),
        #                     remove_non_sig=False)
    else:
        d = np.load(baseline_stat_fname)
        baseline_mean, baseline_std = d['mean'], d['std']

    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        window_output_fname = output_fname.format(window=window)
        window_not_norm_fname = not_norm_output_fname.format(window=window)
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(subject, inverse_method, modality, window))
        if not op.isfile(window_output_fname) or overwrite or \
                (not op.isfile(window_not_norm_fname) and calc_also_non_norm_powers):
            powers = epi_utils.concatenate_powers(fol)
            # if calc_also_non_norm_powers:
            #     freqs_norm = np.max(powers, axis=(0, 2), keepdims=True)
            #     freqs_norm_powers = powers / freqs_norm
            #     powers_abs_minmax = epi_utils.calc_powers_abs_minmax(freqs_norm_powers)
            #     np.save(window_not_norm_fname, powers_abs_minmax)
            # dividing by the mean of baseline values, taking the log, and  dividing by the standard deviation of
            # log baseline values ('zlogratio')
            norm_powers = (powers - baseline_mean) / baseline_std
            # if use_norm_labels_powers:
            #     fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            #         subject, inverse_method, modality, window))
            #     label_norm_powers = glob.glob(op.join(fol, 'epilepsy_*_induced_norm_power.npy'))
            # else:
            #     label_norm_powers = None

            # norm_powers_abs_minmax = epi_utils.calc_powers_abs_minmax(norm_powers, both_min_and_max=False)#label_norm_powers)
            # np.save(window_output_fname, norm_powers_abs_minmax)
            norm_powers_min, norm_powers_max, min_vertices, max_vertices = epi_utils.calc_powers_abs_minmax(
                norm_powers, both_min_and_max=True)
            np.savez(window_output_fname.replace('npy', 'npz'), min=norm_powers_min, max=norm_powers_max,
                     min_vertices=min_vertices, max_vertices=max_vertices)
        else:
            # norm_powers_abs_minmax = np.load(window_output_fname)
            d = np.load(window_output_fname.replace('npy', 'npz'))
            norm_powers_min, norm_powers_max = d['min'], d['max']
            if calc_also_non_norm_powers:
                powers_abs_minmax = np.load(window_not_norm_fname)

        # percentiles = [5, 95]
        # negative_powers, positive_powers = epi_utils.calc_masked_negative_and_positive_powers(
        #     norm_powers_min, norm_powers_max, percentiles)

        # x1, x2 = nans(norm_powers_max.shape), nans(norm_powers_max.shape)
        # x1[max_inds] = norm_powers_max[max_inds]
        # x2[min_inds] = norm_powers_min[min_inds]
        times = epi_utils.get_window_times(window_fname, downsample=2)

        figure_fname = figures_template.format(window=window, method='pos_and_neg')
        plot_positive_and_negative_power_spectrum(
            norm_powers_min, norm_powers_max, times,  '{} {}'.format(modality, window),
            figure_fname=figure_fname, high_gamma_max=high_gamma_max, show_only_sig_in_graph=True)


        # figure_fname = figures_template.format(window=window, method='minmax_two_layers')
        # plot_power_spectrum_two_layers(
        #     negative_powers, positive_powers, times, '{} {}'.format(modality, window), figure_fname)

        # plot_power_spectrum(norm_powers_abs_minmax, figures_template.format(window=window, method='minmax'), baseline_correction=False)
        # plot_power_spectrum(norm_powers_min, figures_template.format(window=window, method='min'), baseline_correction=False)
        # plot_power_spectrum(norm_powers_max, figures_template.format(window=window, method='max'), baseline_correction=False)
        if calc_also_non_norm_powers:
            plot_power_spectrum(
                powers_abs_minmax, times, figures_template_not_norm.format(window=window), vmax=1,
                baseline_correction=False, remove_non_sig=False)


def plot_norm_powers_per_label(subject, windows_fnames, baseline_window, modality, inverse_method='dSPM',
                               calc_also_non_norm_powers=False, overwrite=False, n_jobs=4):
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    baseline_fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
        subject, inverse_method, modality, utils.namebase(baseline_window)))

    baseline_labels = glob.glob(op.join(baseline_fol, 'epilepsy_*_induced_power.npy'))
    indices = np.array_split(np.arange(len(baseline_labels)), n_jobs)
    chunks = [([baseline_labels[ind] for ind in chunk_indices], subject, windows_fnames, inverse_method, modality,
               baseline_window, calc_also_non_norm_powers, overwrite) for chunk_indices in indices]
    utils.run_parallel(_plot_norm_powers_per_label_parallel, chunks, n_jobs)


def _plot_norm_powers_per_label_parallel(p):
    (baseline_labels_chunk, subject, windows_fnames, inverse_method, modality, baseline_window,
     calc_also_non_norm_powers, overwrite) = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'labels-power-spectrum'))
    figs_no_norm_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'labels-power-spectrum-no-norm'))

    for label_baseline_fname in baseline_labels_chunk:
        label_name = utils.namebase(label_baseline_fname).split('_')[1]
        label_figs_fol = utils.make_dir(op.join(figs_fol, label_name))
        label_figs_no_norm_fol = utils.make_dir(op.join(figs_no_norm_fol, label_name))
        figures_template = op.join(label_figs_fol, '{}-epilepsy-{}-{}-{}-{}-induced_power.jpg'.format(
            subject, inverse_method, modality, '{window}', label_name))
        figures_template_no_norm = op.join(label_figs_no_norm_fol, '{}-epilepsy-{}-{}-{}-{}-induced_power.jpg'.format(
            subject, inverse_method, modality, '{window}', label_name))
        baseline = np.load(label_baseline_fname).astype(np.float32)
        baseline_std = np.std(baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
        baseline_mean = np.mean(baseline, axis=2, keepdims=True)  # the standard deviation (over time) of log baseline values
        _calc_label_power_over_windows(
            subject, inverse_method, label_name, modality, baseline_mean, baseline_std, windows_fnames, root_dir,
            figures_template, figures_template_no_norm, calc_also_non_norm_powers, overwrite)


def _calc_label_power_over_windows(
        subject, inverse_method, label_name, modality, baseline_mean, baseline_std, windows_fnames, root_dir,
        figures_template, figures_template_no_norm, calc_also_non_norm_powers, overwrite):
    for window_fname in windows_fnames:
        window = utils.namebase(window_fname)
        fol = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_power'.format(
            subject, inverse_method, modality, window))
        label_powers_fname = op.join(fol, 'epilepsy_{}_induced_power.npy'.format(label_name))
        label_norm_powers_fname = op.join(fol, 'epilepsy_{}_induced_norm_power.npy'.format(label_name))
        if op.isfile(figures_template.format(window=window)) and op.isfile(label_norm_powers_fname) and not overwrite:
            print('{} already exist'.format(figures_template.format(window=window)))
            return
        if op.isfile(label_norm_powers_fname) and not overwrite:
            norm_powers = np.load(label_norm_powers_fname)
        else:
            label_powers = np.load(label_powers_fname).astype(np.float32)
            # if calc_also_non_norm_powers and not op.isfile(figures_template_no_norm.format(window=window)) \
            #         and not overwrite:
            #     freqs_norm = np.max(label_powers, axis=(0, 2), keepdims=True)
            #     freqs_norm_powers = label_powers / freqs_norm
            #     powers_abs_minmax = calc_powers_abs_minmax(freqs_norm_powers)
            #     plot_power_spectrum(
            #         powers_abs_minmax, figures_template_no_norm.format(window=window), vmax=1,
            #         baseline_correction=False, remove_non_sig=False)
            norm_powers = (label_powers - baseline_mean) / baseline_std
            print('Saving {} norm powers to {}'.format(label_name, label_norm_powers_fname))
            np.save(label_norm_powers_fname, norm_powers.astype(np.float16))
        # norm_powers_abs_minmax = calc_powers_abs_minmax(norm_powers)
        # plot_power_spectrum(norm_powers_abs_minmax, figures_template.format(window=window))


def plot_max_powers(subject, windows_fnames, modality, inverse_method='dSPM', overwrite=False, parallel=True):
    params = [(subject, window_fname, modality, inverse_method, overwrite)
              for window_fname in windows_fnames]
    utils.run_parallel(_plot_max_powers_parllel, params, len(windows_fnames) if parallel else 1)


def _plot_max_powers_parllel(p):
    subject, window_fname, modality, inverse_method, overwrite = p
    root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
    fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_max_power.npy'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    if not op.isfile(fname):
        print('Can\'t find {}!'.format(fname))
        return
    figs_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum'))
    figure_fname = op.join(figs_fol, '{}-epilepsy-{}-{}-{}-induced_max_power.jpg'.format(
        subject, inverse_method, modality, utils.namebase(window_fname)))
    # if op.isfile(figure_fname) and not overwrite:
    #     return
    powers = np.load(fname)
    plot_power_spectrum(powers, times, figure_fname)


# @utils.tryit()
def plot_power_spectrum(powers, times, figure_fname, remove_non_sig=True, baseline_correction=True,
                        xlabel='Time (s)', high_gamma_max=120):
    # powers: (freqs x time)
    powers = powers.astype(np.float32).squeeze()
    F, T = powers.shape
    if F not in [88, 52]:
        powers = powers.T
        F, T = powers.shape
    min_t, max_t = times[0], times[-1]

    print('Plotting {}'.format(figure_fname))
    if baseline_correction:
        powers -= np.mean(powers[:, 0])

    if F == 88:
        freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, 305, 5)])
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[55, 120],
                     hfo=[120, 300])
    elif F == 52:
        freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
        bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    else:
        raise Exception('Not supported number of freqs ({})!'.format(F))
    if F != len(freqs):
        print('powers.shape[0] != len(freqs)!!!')
        return
    vmax, vmin = np.max(powers), np.min(powers)
    print('vmin: {}, vmax: {}'.format(vmin, vmax))
    cmap, vmin, vmax = get_cm(vmin, vmax)
    if remove_non_sig:
        powers[np.where(np.abs(powers) < 2)] = 0

    fig, (powers_ax, graph_ax) = plt.subplots(2)  # , sharex=True)
    im = powers_ax.imshow(
        np.flip(powers, 0), vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest',
        extent=extents(times) + extents(freqs), cmap=cmap)
    powers_ax.set_ylabel('Frequency (Hz)')
    add_colorbar(powers_ax, im)
    for band_name, band_freqs in bands.items():
        idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        band_power = np.mean(powers[idx, :], axis=0)
        graph_ax.plot(times, band_power.T, label=band_name)
    graph_ax.set_xlim([min_t, max_t])
    graph_ax.set_xlabel(xlabel)
    # graph_ax.legend() #    add_legend(graph_ax)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=0.92, top=None, wspace=None, hspace=None)
    if figure_fname != '':
        plt.savefig(figure_fname, dpi=300)
        plt.close()
    else:
        plt.show()


def plot_power_spectrum_two_layers(powers_negative, powers_positive, times, title='', figure_fname='',
                                   only_power_spectrum=True, high_gamma_max=120):
    if not only_power_spectrum:
        fig, (ax1, ax2, ax3) = plt.subplots(3)#, sharex=True)
    else:
        fig, ax1 = plt.subplots()
    F, T = powers_negative.shape
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55],
                 high_gamma=[65, high_gamma_max])

    im1 = _plot_powers(powers_negative, ax1, times, freqs)
    # cba = plt.colorbar(im1, shrink=0.25)
    im2 = _plot_powers(powers_positive, ax1, times, freqs)
    cbb = plt.colorbar(im2, shrink=0.25)
    plt.ylabel('frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title(title)

    if not only_power_spectrum:
        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_negative[idx, :], axis=0)
            ax2.plot(band_power.T, label=band_name)
        ax2.set_xlim([0, T])
        # ax2.legend()

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_positive[idx, :], axis=0)
            ax3.plot(band_power.T, label=band_name)
        ax3.set_xlim([0, T])
        # ax3.legend()

    if figure_fname != '':
        print('Saving figure to {}'.format(figure_fname))
        plt.savefig(figure_fname, dpi=300)
        plt.close()
    else:
        plt.show()


def _plot_powers(powers, ax, xaxis=None, freqs=None, cmap_vmin_vmax=None, high_gamma_max=120):

    def extents(f):
        delta = f[1] - f[0]
        return [f[0] - delta / 2, f[-1] + delta / 2]

    times = np.arange(powers.shape[1]) if xaxis is None else xaxis
    if freqs is None:
        freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    if powers.shape[0] != len(freqs):
        print('powers.shape[0] != len(freqs)!!!')
        return

    if cmap_vmin_vmax is None:
        if isinstance(powers, np.ndarray):
            vmax, vmin = np.max(powers), np.min(powers)
        else:
            vmax, vmin = np.ma.masked_array.max(powers), np.ma.masked_array.min(powers)
        cmap, vmin, vmax = get_cm(vmin, vmax)
    else:
        cmap, vmin, vmax = cmap_vmin_vmax
        if vmin > vmax > 0:
            vmax = vmin + 1
        elif vmax < vmin < 0:
            vmin = vmax - 1

    # powers[np.where(powers == 0)] = np.nan
    # cmap.set_bad(color='white')
    im = ax.imshow(np.flip(powers, 0), vmin=vmin, vmax=vmax, aspect='auto', interpolation='nearest',
               extent=extents(times) + extents(freqs), cmap=cmap)
    return im


def plot_positive_and_negative_power_spectrum(
        powers_negative, powers_positive, times, title='', figure_fname='',
        only_power_spectrum=True, show_only_sig_in_graph=True, sig_threshold=2, high_gamma_max=120):
    from src.utils import color_maps_utils as cmu
    YlOrRd = cmu.create_YlOrRd_cm()
    PuBu = cmu.create_PuBu_cm()

    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, high_gamma_max + 5, 5)])
    bands = dict(delta=[1, 4], gamma=[30, 55], theta=[4, 8], alpha=[8, 15], beta=[15, 30])
    if high_gamma_max <= 120:
        bands['high_gamma'] = [55, high_gamma_max]
    else:
        bands['high_gamma'] = [55, 120]
        bands['hfo'] = [120, high_gamma_max]

    if show_only_sig_in_graph:
        powers_negative[np.where(np.abs(powers_negative) < sig_threshold)] = 0
        powers_negative[np.where(np.abs(powers_positive) < powers_positive)] = 0

    min_t, max_t = round(times[0], 1), round(times[-1], 1)
    fig, axs = plt.subplots(2, 2)
    pos_powers_ax, neg_powers_ax = axs[0, 0], axs[0, 1]
    pos_graph_ax, neg_graph_ax = axs[1, 0], axs[1, 1]
    pos_powers_ax.title.set_text('Positives')
    neg_powers_ax.title.set_text('Negatives')
    neg_cmap_vmin_vmax = (PuBu, np.min(powers_negative), -2 if show_only_sig_in_graph else 0)
    pos_cmap_vmin_vmax = (YlOrRd, 2 if show_only_sig_in_graph else 0, np.max(powers_positive))
    im1 = _plot_powers(powers_negative, neg_powers_ax, times, freqs, neg_cmap_vmin_vmax)
    # cba = plt.colorbar(im1, ax=neg_powers_ax,  shrink=0.25)
    im2 = _plot_powers(powers_positive, pos_powers_ax, times, freqs, pos_cmap_vmin_vmax)
    # cbb = plt.colorbar(im2, ax=pos_graph_ax, shrink=0.25)

    for band_name, band_freqs in bands.items():
        idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        band_power = np.mean(powers_negative[idx, :], axis=0)
        neg_graph_ax.plot(times, band_power.T, label=band_name)
    neg_graph_ax.set_xlim([min_t, max_t])
    neg_graph_ax.set_ylim([None, -2 if show_only_sig_in_graph else 0])
    neg_graph_ax.legend()

    for band_name, band_freqs in bands.items():
        band_power = calc_band_power(powers_positive, freqs, band_freqs)
        pos_graph_ax.plot(times, band_power.T, label=band_name)
    pos_graph_ax.set_xlim([min_t, max_t])
    pos_graph_ax.set_ylim([2 if show_only_sig_in_graph else 0, None])

    # pos_graph_ax.legend()
    plt.suptitle(title)

    if figure_fname != '':
        print('Saving figure to {}'.format(figure_fname))
        plt.savefig(figure_fname, dpi=300)
        plt.close()
    else:
        plt.show()


def calc_band_power(powers, freqs, band_freqs):
    idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
    return np.mean(powers[idx, :], axis=0)


def plot_modalities_power_spectrums_with_graph(
        subject, modalities, window_fname, figure_name='', percentiles=[5, 95], inverse_method='dSPM', ylims=[-18, 6],
        file_type='jpg', cb_ticks = [], cb_ticks_font_size=12, figure_fol=''):

    evoked = mne.read_evokeds(window_fname)[0]
    times = evoked.times if len(evoked.times) % 2 == 0 else evoked.times[:-1]
    times = utils.downsample(times, 2)
    min_t, max_t = round(times[0]), round(times[-1])
    freqs = np.concatenate([np.arange(1, 30), np.arange(31, 60, 3), np.arange(60, 125, 5)])
    # bands = dict(delta=[1, 4], theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    bands = dict(delta=[1, 4], high_gamma=[65, 120])
    if figure_fol == '':
        figure_fol = op.join(MMVT_DIR, subject, 'epilepsy-figures', 'power-spectrum')

    fig, ax = plt.subplots(3, len(modalities), figsize=(20, 10))
    for ind, modality in enumerate(modalities):
        powers_ax = ax[0, ind]
        powers_ax.set_title(modality.upper(), fontdict={'fontsize': 18})
        positive_powers_ax = ax[1, ind]
        negative_powers_ax = ax[2, ind]
        root_dir = op.join(EEG_DIR if modality == 'eeg' else MEG_DIR, subject)
        output_fname = op.join(root_dir, '{}-epilepsy-{}-{}-{}-induced_norm_power.npz'.format(
            subject, inverse_method, modality, '{window}'))
        window = utils.namebase(window_fname)
        window_output_fname = output_fname.format(window=window)
        d = np.load(window_output_fname)
        powers_negative, powers_positive = epi_utils.calc_masked_negative_and_positive_powers(d['min'], d['max'], percentiles)

        im1 = _plot_powers(powers_negative, powers_ax, times, freqs)
        im2 = _plot_powers(powers_positive, powers_ax, times, freqs)
        if ind == 0:
            powers_ax.set_ylabel('Frequency (Hz)')
        else:
            powers_ax.set_yticks([])
            add_colorbar(powers_ax, im2, cb_ticks, cb_ticks_font_size)

        # for powers, axis, positive in zip([powers_positive, powers_negative], ax[1:2, ind], [True, False]):
        #     for band_name, band_freqs in bands.items():
        #         idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
        #         band_power = np.mean(powers[idx, :], axis=0)
        #         band_power[abs(band_power) < 2] = 0
        #         axis.plot(times, band_power.T, label=band_name)
        #     axis.set_xlim([min_t, max_t])
        #     axis.set_ylim([2, ylims[1]] if positive else [ylims[0], -2])
        #     if ind == 0:
        #         axis.set_ylabel('Positive Z-Scores' if positive else 'Negative Z-Scores')
        #     else:
        #         axis.set_yticks([])
        #         axis.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        # negative_powers_ax.set_xlabel('Time (s)')

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_negative[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            positive_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        positive_powers_ax.set_xlim([min_t, max_t])
        positive_powers_ax.set_ylim([2, ylims[1]])
        if ind == 0:
            positive_powers_ax.set_ylabel('Positive Z-Scores')
        else:
            positive_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            add_legend(positive_powers_ax)

        for band_name, band_freqs in bands.items():
            idx = [k for k, f in enumerate(freqs) if band_freqs[0] <= f <= band_freqs[1]]
            band_power = np.mean(powers_positive[idx, :], axis=0)
            band_power[abs(band_power) < 2] = 0
            negative_powers_ax.plot(times, band_power.T, label=band_name.replace('_', ' '))
        negative_powers_ax.set_xlim([min_t, max_t])
        negative_powers_ax.set_ylim([ylims[0], -2])
        if ind == 0:
            negative_powers_ax.set_ylabel('Negative Z-Scores')
        else:
            negative_powers_ax.set_yticks([])
        if ind == len(modalities) - 1:
            add_legend(negative_powers_ax)
        negative_powers_ax.set_xlabel('Time (s)')

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=0.92, top=None, wspace=None, hspace=None)
    if figure_name != '':
        plt.savefig(op.join(figure_fol, '{}.{}'.format(figure_name, file_type)), dpi=300)
        plt.close()
    else:
        plt.show()


def add_colorbar(powers_ax, im, cb_ticks=[], cb_ticks_font_size=12):
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
    axins = inset_axes(powers_ax, width="5%", height="100%", loc=5,
                       bbox_to_anchor=(1.15, 0, 1, 1), bbox_transform=powers_ax.transAxes)
    cb = plt.colorbar(im, cax=axins)
    if cb_ticks != []:
        cb.set_ticks(cb_ticks)
    cb.ax.tick_params(labelsize=cb_ticks_font_size)
    cb.ax.set_ylabel('dBHZ Z-Score', color='black', fontsize=cb_ticks_font_size)


def add_legend(powers_ax):
    powers_ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))


def get_cm(vmin, vmax):
    from src.utils import color_maps_utils as cmu
    BuPu_YlOrRd_cm = cmu.create_BuPu_YlOrRd_cm()

    if vmin >= 0:
        cmap = matplotlib.cm.YlOrRd
    elif vmax <= 0:
        cmap = matplotlib.cm.BuPu
    else:
        cmap = BuPu_YlOrRd_cm
        maxmin = max(map(abs, [vmax, vmin]))
        vmin, vmax = -maxmin, maxmin
    return cmap, vmin, vmax


def extents(f):
    delta = f[1] - f[0]
    return [f[0] - delta / 2, f[-1] + delta / 2]

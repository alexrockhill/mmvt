

def plot_modalities(subject, windows, modalities, bands, inverse_method, max_t=0, overwrite=False, n_jobs=4):
    from itertools import product
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'modalities'))
    params = [(subject, modalities, inverse_method, figures_fol, band, window_fname, max_t, overwrite)
              for (band, window_fname) in product(bands, windows)]
    utils.run_parallel(_plot_modalities_parallel, params, n_jobs)


def _plot_modalities_parallel(p):
    subject, modalities, inverse_method, figures_fol, band, window_fname, max_t, overwrite = p
    window_name = utils.namebase(window_fname)
    plt.figure()
    for modality in modalities:
        modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
        fig_fname = op.join(figures_fol, '{}-{}.jpg'.format(window_name, band))
        if op.isfile(fig_fname) and not overwrite:
            print('{} already exist'.format(fig_fname))
            break
        stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
            subject, inverse_method, modality, window_name, band))
        if not op.isfile(stc_band_fname):
            print('Can\'t find {}!'.format(stc_band_fname))
            break
        print('Loading {} ({})'.format(stc_band_fname, utils.file_modification_time(stc_band_fname)))
        stc = mne.read_source_estimate(stc_band_fname)
        data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
        plt.plot(data.T)
    else:
        plt.title('{} {}'.format(window_name, band))
        plt.legend(modalities)
        print('Saving {} {}'.format(window_name, band))
        plt.savefig(fig_fname, dpi=300)
        plt.close()


def plot_activity_modalities(subject, windows, modalities, inverse_method, max_t=0, overwrite=False):
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'amplitude'))
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        fig_fname = op.join(figures_fol, '{}-amplitude.jpg'.format(window_name))
        # if op.isfile(fig_fname) and not overwrite:
        #     print('{} already exist'.format(fig_fname))
        #     break
        plt.figure()
        all_files_found = True
        for modality in modalities:
            modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
            stc_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_amplitude-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name))
            if not op.isfile(stc_fname):
                print('Can\'t find {}!'.format(stc_fname))
                all_files_found = False
                break
            stc = mne.read_source_estimate(stc_fname)
            data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
            plt.plot(data.T)
        if all_files_found:
            plt.title('{} amplitude'.format(window_name))
            plt.legend(modalities)
            print('Saving {} amplitude'.format(window_name))
            plt.savefig(fig_fname, dpi=300)
            plt.close()


def plot_freqs(subject, windows, modality, bands, inverse_method, max_t=0, subfol=''):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    for band in bands:
        plt.figure()
        for window_fname in windows:
            window_name = utils.namebase(window_fname)
            figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'runs_frontal_temporal'))
            if subfol != '':
                figures_fol = utils.make_dir(op.join(figures_fol, subfol))
            fig_fname = op.join(figures_fol, '{}-{}.jpg'.format(modality, band))
            if op.isfile(fig_fname):
                print('{} already exist'.format(fig_fname))
                break
            stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name, band))
            if not op.isfile(stc_band_fname):
                print('Can\'t find {}!'.format(stc_band_fname))
                break
            stc = mne.read_source_estimate(stc_band_fname)
            data = np.max(stc.data[:, :max_t], axis=0) if max_t > 0 else np.max(stc.data, axis=0)
            plt.plot(data.T)
        plt.title('{} {}'.format(modality, band))
        plt.legend([utils.namebase(w) for w in windows])
        print('Saving {} {}'.format(modality, band))
        plt.savefig(fig_fname, dpi=300)
        plt.close()


def plot_windows(subject, windows, modality, bands, inverse_method):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    for window_fname in windows:
        window_name = utils.namebase(window_fname)
        figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'epilepsy-per_window-figures'))
        stc_name = '{}-epilepsy-{}-{}-{}'.format(subject, inverse_method, modality, window_name)
        fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
        if op.isfile(fig_fname):
            print('{} already exist'.format(fig_fname))
            continue
        plt.figure()
        all_found = True
        for band in bands:
            stc_band_fname = op.join(modality_fol, '{}-epilepsy-{}-{}-{}_{}-zvals-lh.stc'.format(
                subject, inverse_method, modality, window_name, band))
            if not op.isfile(stc_band_fname):
                print('Can\'t find {}!'.format(stc_band_fname))
                all_found = False
                break
            stc = mne.read_source_estimate(stc_band_fname)
            data = np.max(stc.data, axis=0)
            plt.plot(data.T)
        if all_found:
            plt.title(window_name)
            plt.legend(bands)
            print('Saving {}'.format(window_name))
            plt.savefig(fig_fname, dpi=300)
            plt.close()


def plot_baseline(subject, baseline_name):
    stc_fnames = glob.glob(
        op.join(MMVT_DIR, subject, 'meg', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name))) + \
        glob.glob(op.join(MMVT_DIR, subject, 'eeg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name)))
    if len(stc_fnames) == 0:
        stc_fnames = glob.glob(
            op.join(MMVT_DIR, subject, 'meg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name))) + \
            glob.glob(op.join(MMVT_DIR, subject, 'eeg', 'non-zvals', '{}-epilepsy-*-{}_*.stc'.format(subject, baseline_name)))
    if len(stc_fnames) == 0:
        print('No baselines stc files were found!')
        return
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'baseline'))
    utils.run_parallel(_plot_stcs_files_parallel, [(stc_fname, figures_fol) for stc_fname in stc_fnames], n_jobs)


def plot_stcs_files(subject, modality, n_jobs=4):
    modality_fol = op.join(MMVT_DIR, subject, 'eeg' if modality == 'eeg' else 'meg')
    stc_files = [f for f in glob.glob(op.join(modality_fol, '*-lh.stc'))
                 if '-epilepsy-' in utils.namebase(f) and '-zvals-' in utils.namebase(f) and
                 '-{}-'.format(modality) in utils.namebase(f)]
    print('{} files for {}'.format(len(stc_files), modality))
    figures_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'epilepsy-figures', 'all_stcs', modality))
    utils.run_parallel(_plot_stcs_files_parallel, [(stc_fname, figures_fol) for stc_fname in stc_files], n_jobs)


def _plot_stcs_files_parallel(p):
    stc_fname, figures_fol = p
    plot_stc_file(stc_fname, figures_fol)


def plot_stc_file(stc_fname, figures_fol):
    stc_name = utils.namebase(stc_fname)[:-3]
    fig_fname = op.join(figures_fol, '{}.jpg'.format(stc_name))
    if not op.isfile(fig_fname):
        stc = mne.read_source_estimate(stc_fname)
        data = np.max(stc.data, axis=0)
        plt.figure()
        plt.plot(data.T)
        plt.title(utils.namebase(stc_fname)[:-3])
        print('Saving {}'.format(fig_fname))
        plt.savefig(fig_fname, dpi=300)
        plt.close()





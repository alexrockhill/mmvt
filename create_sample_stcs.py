import os.path as op
from src.preproc import meg, eeg
from src.utils import utils
import mne

LINKS_DIR = utils.get_links_dir()
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')

# new version can be downloaded from here
# https://github.com/mne-tools/mne-cpp/tree/master/bin/MNE-sample-data
MNE_ROOT = mne.datasets.sample.data_path()
# MNE_ROOT = '/autofs/space/thibault_001/users/npeled/datasets/MNE-sample-data/MEG/sample'


def calc_sample_meg_data():
    args = meg.read_cmd_args(dict(
        subject='sample',
        function='calc_epochs,calc_evokes,calc_stc',
        contrast='audvis', task='audvis',
        pick_meg=True, pick_eeg=False,
        fwd_usingMEG=True, fwd_usingEEG=False,
        fname_format='{subject}_audvis_meg-{ana_type}.{file_type}',
        fname_format_cond='{subject}_audvis_meg_{cond}-{ana_type}.{file_type}',
        trans_fname = op.join(MNE_ROOT, 'sample_audvis_raw-trans.fif'),
        events_fname = op.join(MNE_ROOT, 'sample_audvis_raw-eve.fif'),
        raw_fname = op.join(MNE_ROOT, 'sample_audvis_raw.fif'),
        inv_fname = op.join(MNE_ROOT, 'sample_audvis-meg-oct-6-meg-inv.fif'),
        fwd_fname = op.join(MNE_ROOT, 'sample_audvis-meg-oct-6-fwd.fif'),
        conditions=['LA', 'RA'],
        read_events_from_file=True,
        t_min=-0.2, t_max=0.5,
        overwrite_epochs=False,
        overwrite_evoked=False,
        overwrite_stc=True
    ))
    meg.call_main(args)


def calc_sample_eeg_data():
    args = eeg.read_cmd_args(dict(
        subject='sample',
        function='calc_epochs,calc_evokes,calc_stc',
        contrast='audvis', task='audvis',
        pick_meg=False, pick_eeg=True,
        fwd_usingMEG=False, fwd_usingEEG=True,
        fname_format='{subject}_audvis_eeg-{ana_type}.{file_type}',
        fname_format_cond='{subject}_audvis_eeg_{cond}-{ana_type}.{file_type}',
        trans_fname = op.join(MNE_ROOT, 'sample_audvis_raw-trans.fif'),
        events_fname = op.join(MNE_ROOT, 'sample_audvis_raw-eve.fif'),
        raw_fname = op.join(MNE_ROOT, 'sample_audvis_raw.fif'),
        inv_fname = op.join(MNE_ROOT, 'sample_audvis-eeg-oct-6-eeg-inv.fif'),
        fwd_fname = op.join(MNE_ROOT, 'sample_audvis-eeg-oct-6-fwd.fif'),
        conditions=['LA', 'RA'],
        read_events_from_file=True,
        t_min=-0.2, t_max=0.5,
        overwrite_epochs=False,
        overwrite_evoked=False,
        overwrite_stc=True
    ))
    eeg.call_main(args)


if __name__ == '__main__':
    calc_sample_eeg_data()
    calc_sample_meg_data()

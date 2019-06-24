import os.path as op
import glob
import re
from src.utils import utils

LINKS_DIR = utils.get_links_dir()
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
EEG_DIR = utils.get_link_dir(LINKS_DIR, 'eeg')
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects')


def find_room_noise(fol):
    files = glob.glob(op.join(fol, '*roomnoise_raw.fif'))
    if len(files) >= 1:
        return utils.select_one_file(files, 'room noise')
    files = glob.glob(op.join(fol, '*noiseroom_raw.fif'))
    if len(files) >= 1:
        return utils.select_one_file(files, 'room noise')
    file_name = input('Can\'t find the room noise file, please input its name: ')
    fname = op.join(fol, file_name)
    if not op.isfile(fname):
        raise Exception('No room noise!')
    else:
        return fname


def find_raw_fname(meg_fol, run):
    raw_fname = ''
    run_num = re.sub('\D', ',', run).split(',')[-1]
    raw_files = glob.glob(op.join(meg_fol, '*_{}_*raw*.fif'.format(str(run_num).zfill(2))))
    if not op.isfile(raw_fname):
        raw_run_files = [f for f in raw_files if 'annot' not in utils.namebase(f) and 'eve' not in utils.namebase(f)]
        if len(raw_run_files) == 0:
            return ''
        ssst_raw_files = [f for f in raw_run_files if 'ssst' in utils.namebase(f)]
        if len(ssst_raw_files) > 0:
            raw_fname = utils.select_one_file(ssst_raw_files, 'raw file for run {}'.format(run_num))
        else:
            raw_fname = utils.select_one_file(raw_run_files, 'raw file for run {}'.format(run_num))
    return raw_fname, run_num


def subject_nmr00857():
    subject = 'nmr00857'
    evokes_fol = op.join(MEG_DIR, subject, 'evoked')
    meg_fol = '/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123'
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG061,EEG02,EEG042,MEG0112,MEG0113'
    baseline_name = 'BaseLINE'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name


def subject_nmr01321():
    subject = 'nmr01321'
    evokes_fol = [d for d in [
        # '/autofs/space/frieda_001/users/valia/epilepsy/4272326_01321/MMVT_epochs',
        # '/homes/5/npeled/space1/MEG/nmr01321/evokeds',
        op.join(MMVT_DIR, subject, 'evoked')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        '/autofs/space/frieda_001/users/valia/epilepsy/5241495_00857/subj_5241495/190123',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG001,EEG003,EEG004,EEG005,EEG008,EEG034,EEG045,EEG051,EEG057,EEG058,EEG060,EEG061,EEG062,EEG074,MEG1422,MEG1532,MEG2012,MEG2022'
    baseline_name = 'Base_line'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name, True


def subject_nmr01325():
    subject = 'nmr01325'
    evokes_fol = [d for d in [
        # '/cluster/neuromind/valia/epilepsy/6645962_01325',
        op.join(MEG_DIR, subject, 'evokes')] if op.isdir(d)][0]
    meg_fol = [d for d in [
        '/cluster/neuromind/valia/epilepsy/6645962_01325/190523',
        op.join(MEG_DIR, subject)] if op.isdir(d)][0]
    empty_fname = find_room_noise(meg_fol)
    bad_channels = 'EEG020,EEG021,EEG050,EEG051'
    baseline_name = 'baseline_607' # '33_35secAWAKE' #'108_35secSLEEP' # 'baseline_607' # 'bl_502s' # 'bl_474s' #  #  '108_35secSLEEP' '33_35secAWAKE' '550_20sec'
    return subject, evokes_fol, meg_fol, empty_fname, bad_channels, baseline_name, True

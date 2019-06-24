from src.utils import trans_utils as tu
from src.utils import preproc_utils as pu
from src.utils import utils
import csv
import os.path as op
from tqdm import tqdm
import glob

SUBJECTS_DIR, MMVT_DIR, FREESURFER_HOME = pu.get_links()


def trans_tal_coords(files, template='colin27', overwrite=False):
    output_fol = utils.make_dir(op.join(MMVT_DIR, template, 'rois_peaks'))
    output_fname = op.join(output_fol, 'rois.pkl')
    if not op.isfile(output_fname) or overwrite:
        rois= get_tal_coordaintes(files)
        utils.save(rois, output_fname)
    else:
        rois = utils.load(output_fname)
    for roi in rois.keys():
        csv_fname = op.join(output_fol, '{}_mni.csv'.format(roi))
        csv_tal_fname = op.join(output_fol, '{}_tal.csv'.format(roi))
        with open(csv_fname, 'w') as csv_file_mni, open(csv_tal_fname, 'w') as csv_file_tal:
            csv_mni_writer = csv.writer(csv_file_mni, delimiter=',')
            csv_tal_writer = csv.writer(csv_file_tal, delimiter=',')
            for mni, tal in zip(rois[roi]['mni'], rois[roi]['tal']):
                csv_mni_writer.writerow(mni)
                csv_tal_writer.writerow(tal)


def get_tal_coordaintes(files):
    rois, errors = {}, {}
    driver = None
    for fname in tqdm(files):
        # subject = utils.namebase(utils.get_parent_fol(fname, 3))
        roi = utils.namebase(fname).split('.')[0]
        if roi not in rois:
            rois[roi] = {}
            rois[roi]['tal'] = []
            rois[roi]['mni'] = []
        lines = list(utils.csv_file_reader(fname, delimiter=' '))
        if len(lines) == 0:
            errors[fname] = '{} is empty!'.format(fname)
            continue
        elif len(lines) > 1:
            errors[fname] = 'More than one line in {}!'.format(fname)
            continue
        tal = [int(float(v)) for v in lines[0] if utils.is_float(v)]
        rois[roi]['tal'].append(tal)
        if driver is None:
            driver = tu.yale_get_driver()
        rois[roi]['mni'].append(tu.yale_tal2mni(tal, driver))
    if len(errors) > 0:
        print(errors)
    del driver
    return rois


if __name__ == '__main__':
    subjects_dir = '/autofs/space/lilli_004/users/DARPA-MSIT/msit/subjs/'
    files = glob.glob(op.join(subjects_dir, '**', '*3dExtrema.txt'), recursive=True)
    trans_tal_coords(files, 'colin27', False)
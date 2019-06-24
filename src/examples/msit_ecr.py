import os.path as op
import numpy as np
from itertools import product
import shutil
import os
import os
import time
import glob
from src.utils import utils
from src.utils import labels_utils as lu
from src.preproc import anatomy as anat
from src.preproc import meg as meg
from src.preproc import connectivity
from collections import defaultdict
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
from src.misc import meg_buddy as mb

LINKS_DIR = utils.get_links_dir()
SUBJECTS_DIR = utils.get_link_dir(LINKS_DIR, 'subjects', 'SUBJECTS_DIR')
MEG_DIR = utils.get_link_dir(LINKS_DIR, 'meg')
MMVT_DIR = utils.get_link_dir(LINKS_DIR, 'mmvt')


def prepare_files(args):
    # todo: should look in the dict for files locations
    ret = {}
    for subject in args.subject:
        ret[subject] = True
        for task in args.tasks:
            fol = utils.make_dir(op.join(MEG_DIR, task, subject))
            local_epo_fname = op.join(fol, args.epo_template.format(subject=subject, task=task))
            local_raw_fname = op.join(fol, '{}_{}-raw.fif'.format(subject, task))
            if not args.overwrite and (op.islink(local_epo_fname) or op.isfile(local_epo_fname)) and \
                    (op.islink(local_raw_fname) or op.isfile(local_raw_fname)):
                continue

            if op.islink(local_epo_fname) or op.isfile(local_epo_fname) and args.overwrite_local_files:
                os.remove(local_epo_fname)
            if not op.islink(local_epo_fname) and not op.isfile(local_epo_fname):
                remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
                if not op.isfile(remote_epo_fname):
                    print('{} does not exist!'.format(remote_epo_fname))
                    ret[subject] = False
                    continue
                print('Creating a link {} -> {}'.format(remote_epo_fname, local_epo_fname))
                utils.make_link(remote_epo_fname, local_epo_fname)

            if op.islink(local_raw_fname) or op.isfile(local_raw_fname) and args.overwrite_local_files:
                os.remove(local_raw_fname)
            if not op.islink(local_raw_fname) and not op.isfile(local_raw_fname):
                remote_raw_fname = op.join(
                    utils.get_parent_fol(args.meg_dir), 'raw_preprocessed', subject,
                    args.raw_template.format(subject=subject, task=task))
                if not op.isfile(remote_raw_fname):
                    print('{} does not exist!'.format(remote_raw_fname))
                    ret[subject] = False
                    continue
                print('Creating a link {} -> {}'.format(remote_raw_fname, local_raw_fname))
                utils.make_link(remote_raw_fname, local_raw_fname)
        ret[subject] = ret[subject] and (op.isfile(local_epo_fname) or op.islink(local_epo_fname)) and \
                       (op.isfile(local_raw_fname) or op.islink(local_raw_fname))
    print('Good subjects:')
    print([s for s, r in ret.items() if r])
    print('Bad subjects:')
    print([s for s, r in ret.items() if not r])


def anatomy_preproc(args, subject=''):
    args = anat.read_cmd_args(dict(
        subject=args.subject if subject == '' else subject,
        remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
        # high_level_atlas_name='darpa-atlas',
        # function='create_annotation,create_high_level_atlas',
        function='create_annotation',
        overwrite_fs_files=args.overwrite,
        atlas='laus125',
        ignore_missing=True
    ))
    anat.call_main(args)


def get_empty_fnames(subject, tasks, args, overwrite=False):
    utils.make_dir(op.join(MEG_DIR, subject))
    try:
        if op.islink(op.join(SUBJECTS_DIR, subject, 'bem')):
            os.remove(op.join(SUBJECTS_DIR, subject, 'bem'))
        utils.make_link(op.join(args.remote_subject_dir.format(subject=subject), 'bem'),
                        op.join(SUBJECTS_DIR, subject, 'bem'), overwrite=overwrite)
        utils.make_link(op.join(MEG_DIR, subject, 'bem'),
                        op.join(SUBJECTS_DIR, subject, 'bem'), overwrite=overwrite)
        for task in tasks:
            utils.make_dir(op.join(MEG_DIR, task, subject))
            utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'), overwrite=overwrite)
        utils.make_link(op.join(MEG_DIR, subject, 'bem'), op.join(SUBJECTS_DIR, subject, 'bem'), overwrite=overwrite)
    except:
        print('{}: Errors with making links'.format(subject))

    remote_meg_fol = op.join(args.remote_meg_dir, subject)
    csv_fname = op.join(remote_meg_fol, 'cfg.txt')
    empty_fnames, cors, days = '', '', ''

    if not op.isfile(csv_fname):
        print('No cfg file!')
        return '', '', ''
    days, empty_fnames, cors = {}, {}, {}
    for line in utils.csv_file_reader(csv_fname, ' '):
        for task in tasks:
            if line[4].lower() == task.lower():
                days[task] = line[2]
    # print(days)
    for line in utils.csv_file_reader(csv_fname, ' '):
        if line[4] == 'empty':
            for task in tasks:
                empty_fnames[task] = op.join(MEG_DIR, task, subject, '{}_empty_raw.fif'.format(subject))
                if op.isfile(empty_fnames[task]):
                    continue
                task_day = days[task]
                if line[2] == task_day:
                    empty_fname = op.join(remote_meg_fol, line[0].zfill(3), line[-1])
                    if not op.isfile(empty_fname):
                        raise Exception('empty file does not exist! {}'.format(empty_fname[task]))
                    utils.make_link(empty_fname, empty_fnames[task])
    cor_dir = op.join(args.remote_subject_dir.format(subject=subject), 'mri', 'T1-neuromag', 'sets')
    for task in tasks:
        if op.isfile(op.join(cor_dir, 'COR-{}-{}.fif'.format(subject, task.lower()))):
            cors[task] = op.join(cor_dir, 'COR-{}-{}.fif'.format('{subject}', task.lower()))
        elif op.isfile(op.join(cor_dir, 'COR-{}-day{}.fif'.format(subject, days[task]))):
            cors[task] = op.join(cor_dir, 'COR-{}-day{}.fif'.format('{subject}', days[task]))
    return empty_fnames, cors, days

#
# def calc_meg_epochs(args):
#     empty_fnames, cors, days = get_empty_fnames(args.subject[0], args.tasks, args)
#     times = (-2, 4)
#     for task in args.tasks:
#         args = meg.read_cmd_args(dict(
#             subject=args.subject, mri_subject=args.subject,
#             task=task,
#             remote_subject_dir='/autofs/space/lilli_001/users/DARPA-Recons/{subject}',
#             get_task_defaults=False,
#             fname_format='{}_{}_nTSSS-ica-raw'.format('{subject}', task.lower()),
#             empty_fname=empty_fnames[task],
#             function='calc_epochs,calc_evokes',
#             conditions=task.lower(),
#             data_per_task=True,
#             normalize_data=False,
#             t_min=times[0], t_max=times[1],
#             read_events_from_file=False, stim_channels='STI001',
#             use_empty_room_for_noise_cov=True,
#             n_jobs=args.n_jobs
#         ))
#         meg.call_main(args)


def calc_fwd_inv(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    subjects = get_good_subjects(args)
    args.subject = subjects
    prepare_files(args)
    good_subjects, bad_subjects = [], []

    for subject in subjects:
        args.subject = subject
        empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
        good_subject = True
        for task in args.tasks:
            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, atlas=atlas,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                empty_fname=empty_fnames[task] if empty_fnames != '' else '',
                function='make_forward_solution,calc_inverse_operator',
                conditions=task.lower(),
                fwd_fname=op.join(MEG_DIR, task, subject, '{}_{}_Onset-fwd.fif'.format(subject, task)),
                inv_fname=op.join(MEG_DIR, task, subject, '{}_{}_Onset-inv.fif'.format(subject, task)),
                cor_fname=cors[task].format(subject=subject) if cors != '' else '',
                average_per_event=False,
                data_per_task=True,
                use_empty_room_for_noise_cov=True,
                read_only_from_annot=False,
                check_for_channels_inconsistency=args.check_for_channels_inconsistency,
                overwrite_fwd=args.overwrite,
                overwrite_inv=args.overwrite,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)
            good_subject = good_subject and ret[subject]['make_forward_solution'] and \
                           ret[subject]['calc_inverse_operator']
        if good_subject:
            good_subjects.append(subject)
        else:
            bad_subjects.append(subject)
    print('Good subjects ({}):'.format(len(good_subjects)))
    print(good_subjects)
    print('Bad subjects ({}):'.format(len(bad_subjects)))
    print(bad_subjects)


def meg_preproc_evoked(args):
    inv_method, em, atlas= 'dSPM', 'mean_flip', args.atlas
    # bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    times = (-2, 4)
    subjects_with_error = []
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects
    prepare_files(args)

    for subject in good_subjects:
        args.subject = subject
        empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
        input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
        for task in args.tasks:

            # output_fname = op.join(
            #     MMVT_DIR, subject, 'meg', '{}_{}_{}_power_spectrum.npz'.format(task.lower(), inv_method, em))
            # if op.isfile(output_fname) and args.check_file_modification_time:
            #     file_mod_time = utils.file_modification_time_struct(output_fname)
            #     if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 9 and file_mod_time.tm_mday >= 21) or \
            #             (file_mod_time.tm_mon > 9):
            #         print('{} already exist!'.format(output_fname))
            #         continue

            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname) and not op.isfile(remote_epo_fname):
                print('Can\'t find {}!'.format(local_epo_fname))
                continue
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)

            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                empty_fname=empty_fnames[task] if empty_fnames != '' else '',
                function='make_forward_solution,calc_inverse_operator,calc_stc,calc_labels_avg_per_condition,calc_labels_min_max',
                conditions=task.lower(),
                cor_fname=cors[task].format(subject=subject) if cors != '' else '',
                average_per_event=False,
                data_per_task=True,
                pick_ori='normal', # very important for calculation of the power spectrum
                ica_overwrite_raw=False,
                normalize_data=False,
                t_min=times[0], t_max=times[1],
                read_events_from_file=False, stim_channels='STI001',
                use_empty_room_for_noise_cov=True,
                read_only_from_annot=False,
                # pick_ori='normal',
                check_for_channels_inconsistency=args.check_for_channels_inconsistency,
                overwrite_labels_power_spectrum=args.overwrite_labels_power_spectrum,
                overwrite_evoked=True,#args.overwrite,
                overwrite_fwd=args.overwrite,
                overwrite_inv=args.overwrite,
                overwrite_stc=True,#args.overwrite,
                overwrite_labels_data=True,#args.overwrite,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)
            # output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
            # join_res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr', subject))
            # for res_fname in glob.glob(op.join(output_fol, '{}_labels_{}_{}_*_power.npz'.format(
            #         task.lower(), inv_method, em))):
            #     utils.copy_file(res_fname, op.join(join_res_fol, utils.namebase_with_ext(res_fname)))
            if not ret:
                if args.throw:
                    raise Exception("errors!")
                else:
                    subjects_with_error.append(subject)


    good_subjects = [s for s in good_subjects if
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
           op.isfile(op.join(MMVT_DIR, subject, 'meg',
                             'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
    print('Good subjects:')
    print(good_subjects)
    print('subjects_with_error:')
    print(subjects_with_error)


def meg_sensors_psd(args):
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects
    prepare_files(args)
    for subject in good_subjects:
        args.subject = subject
        for task in args.tasks:
            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname) and not op.isfile(remote_epo_fname):
                print('Can\'t find {}!'.format(local_epo_fname))
                continue
            remote_raw_fname = op.join(args.remote_root_dir, 'raw_preprocessed', subject, args.raw_template.format(subject=subject, task=task))
            local_raw_fname = op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task))
            if not op.isfile(local_raw_fname) and not op.isfile(remote_raw_fname):
                print('Can\'t find {}!'.format(local_raw_fname))
                continue
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)
            if not op.isfile(local_raw_fname):
                utils.make_link(remote_raw_fname, local_raw_fname)

            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task,
                # meg_dir=args.meg_dir,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                function='calc_baseline_sensors_bands_psd,calc_epochs_psd',
                conditions=task.lower(),
                baseline_len=60000,
                average_per_event=False,
                data_per_task=True,
                max_epochs_num=args.max_epochs_num,
                ignore_missing=args.ignore_missing,
                check_for_channels_inconsistency=args.check_for_channels_inconsistency,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)


def meg_preproc_power_how_many(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    good_subjects, bad_subjects = [], []
    subjects = get_good_subjects(args)
    for subject in subjects:
        fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg'))
        good_subject = False
        for task in args.tasks:
            output_fname = op.join(fol, '{}_{}_{}_power_spectrum.npz'.format(task.lower(), inv_method, em))
            if op.isfile(output_fname):
                file_mod_time = utils.file_modification_time_struct(output_fname)
                if file_mod_time.tm_year < 2019 or file_mod_time.tm_mon < 6 or (file_mod_time.tm_mon == 6 and
                                                                                file_mod_time.tm_mday < 20):
                    break
                d = np.load(output_fname)
                if 'power_spectrum_basline' in d and d['power_spectrum_basline'] is not None:
                    good_subject = True
        if good_subject:
            good_subjects.append(subject)
        else:
            bad_subjects.append(subject)
    print('Good subjects ({}):'.format(len(good_subjects)))
    print(good_subjects)
    print('Bad subjects ({}):'.format(len(bad_subjects)))
    print(bad_subjects)


def meg_preproc_power(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    # bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
    baseline = (-2, 0)
    subjects_with_error = []
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects
    prepare_files(args)
    calc_power_spectrum = True

    # function = 'make_forward_solution,calc_inverse_operator'
    func_name = 'calc_source_power_spectrum' if calc_power_spectrum else 'calc_labels_induced_power'
    function = func_name # ',{}'.format(func_name)

    for subject in good_subjects:
        args.subject = subject
        fol = utils.make_dir(op.join(MMVT_DIR, subject, 'meg'))
        empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
        input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
        for task in args.tasks:
            vertices_data_fname = op.join(
                fol, '{}_{}_{}_vertices_power_spectrum.pkl'.format(task.lower(), inv_method, em))
            if op.isfile(vertices_data_fname):
                os.remove(vertices_data_fname)
            # output_fname = op.join(MMVT_DIR, subject, 'meg', '{}_{}_{}_power_spectrum.npz'.format(
            #     task.lower(), inv_method, em))
            # if op.isfile(output_fname) and args.check_file_modification_time:
            #     file_mod_time = utils.file_modification_time_struct(output_fname)
            #     if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 9 and file_mod_time.tm_mday >= 21) or \
            #             (file_mod_time.tm_mon > 9):
            #         print('{} already exist!'.format(output_fname))
            #         continue

            # if not args.overwrite_output_files:
            #     output_fnames = glob.glob(
            #         op.join(input_fol, '{}_*_{}_{}_{}_induced_power.npz'.format(task.lower(), atlas, inv_method, em)))
            #     overwrite = False
            #     for output_fname in output_fnames:
            #         file_mod_time = utils.file_modification_time_struct(output_fname)
            #         if file_mod_time.tm_year < 2018 or (file_mod_time.tm_mon == 10 and file_mod_time.tm_mday < 23) or \
            #                 (file_mod_time.tm_mon < 10):
            #             overwrite = True
            #
            #     if len(output_fnames) == 28:
            #         print('{} has already all the results for {}'.format(subject, task))
            #         continue
            if task not in empty_fnames:
                print('{} not in empty_fnames!'.format(task))
                continue
            if task not in cors:
                print('{} not in cors!'.format(task))
                continue
            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname) and not op.isfile(remote_epo_fname):
                print('Can\'t find {}!'.format(local_epo_fname))
                continue
            if not op.isfile(local_epo_fname) and not op.exists(local_epo_fname):
                if op.islink(local_epo_fname):
                    os.remove(local_epo_fname)
                utils.make_link(remote_epo_fname, local_epo_fname)

            meg_args = meg.read_cmd_args(dict(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
                # meg_dir=args.meg_dir,
                remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
                get_task_defaults=False,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                fwd_fname=op.join(MEG_DIR, task, subject, '{}_{}_Onset-fwd.fif'.format(subject, task)),
                inv_fname=op.join(MEG_DIR, task, subject, '{}_{}_Onset-inv.fif'.format(subject, task)),
                empty_fname=empty_fnames.get(task, '') if empty_fnames != '' else '',
                function=function,
                conditions=task.lower(),
                cor_fname=cors.get(task, '').format(subject=subject) if cors != '' else '',
                average_per_event=False,
                data_per_task=True,
                pick_ori='normal', # very important for calculation of the power spectrum
                # fmin=4, fmax=120, bandwidth=2.0,
                max_epochs_num=args.max_epochs_num,
                ica_overwrite_raw=False,
                normalize_data=False,
                fwd_recreate_source_space=True,
                baseline_min=baseline[0], baseline_max=baseline[1],
                read_events_from_file=False, stim_channels='STI001',
                use_empty_room_for_noise_cov=True,
                read_only_from_annot=False,
                average_over_label_indices=args.average_over_label_indices,
                ignore_missing=args.ignore_missing,
                save_tmp_files=False,
                check_for_channels_inconsistency=args.check_for_channels_inconsistency,
                # pick_ori='normal',
                overwrite_labels_induced_power=args.overwrite_output_files,
                overwrite_labels_power_spectrum=args.overwrite_output_files,
                overwrite_evoked=args.overwrite,
                overwrite_fwd=args.overwrite,
                overwrite_inv=args.overwrite,
                overwrite_stc=args.overwrite,
                overwrite_labels_data=args.overwrite,
                n_jobs=args.n_jobs
            ))
            ret = meg.call_main(meg_args)
            if subject in ret and func_name in ret[subject] and ret[subject][func_name]:
                good_subjects.append(subject)
            else:
                subjects_with_error.append(subject)
            output_fol = utils.make_dir(op.join(MMVT_DIR, subject, 'labels', 'labels_data'))
            join_res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr', subject))
            for res_fname in glob.glob(op.join(output_fol, '{}_labels_{}_{}_*_power.npz'.format(
                    task.lower(), inv_method, em))):
                utils.copy_file(res_fname, op.join(join_res_fol, utils.namebase_with_ext(res_fname)))
            # if not ret:
            #     if args.throw:
            #         raise Exception("errors!")
            #     else:
            #         subjects_with_error.append(subject)

    # good_subjects = [s for s in good_subjects if
    #        op.isfile(op.join(MMVT_DIR, subject, 'meg',
    #                          'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
    #        op.isfile(op.join(MMVT_DIR, subject, 'meg',
    #                          'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
    print('Good subjects:')
    print(good_subjects)
    print('subjects_with_error:')
    print(subjects_with_error)


# def calc_source_band_induced_power(args):
#     inv_method, em, atlas= 'MNE', 'mean_flip', 'darpa-atlas'
#     bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 200])
#     times = (-2, 4)
#     subjects_with_error = []
#     good_subjects = get_good_subjects(args)
#     args.subject = good_subjects
#     prepare_files(args)
#     done_subjects = []
#
#     for subject in good_subjects:
#         args.subject = subject
#         empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
#         for task in args.tasks:
#             fol = utils.make_dir(op.join(MEG_DIR, task, subject, 'induced_power'))
#             output_fnames = glob.glob(op.join(fol, '{}*induced_power*.stc'.format(task)))
#             # If another thread is working on this subject / task, continue to another subject / task
#             # if len(output_fnames) > 0:
#             #     done_subjects.append(subject)
#             #     continue
#             meg_args = meg.read_cmd_args(dict(
#                 subject=args.subject, mri_subject=args.subject,
#                 task=task, inverse_method=inv_method, extract_mode=em, atlas=atlas,
#                 # meg_dir=args.meg_dir,
#                 remote_subject_dir=args.remote_subject_dir, # Needed for finding COR
#                 get_task_defaults=False,
#                 fname_format='{}_{}_Onset'.format('{subject}', task),
#                 raw_fname=op.join(MEG_DIR, task, subject, '{}_{}-raw.fif'.format(subject, task)),
#                 epo_fname=op.join(MEG_DIR, task, subject, '{}_{}_meg_Onset-epo.fif'.format(subject, task)),
#                 function='calc_stc',
#                 calc_source_band_induced_power=True,
#                 calc_inducde_power_per_label=True,
#                 induced_power_normalize_proj=True,
#                 overwrite_stc=args.overwrite,
#                 conditions=task.lower(),
#                 cor_fname=cors[task].format(subject=subject),
#                 data_per_task=True,
#                 n_jobs=args.n_jobs
#             ))
#             ret = meg.call_main(meg_args)
#             if not ret:
#                 if args.throw:
#                     raise Exception("errors!")
#                 else:
#                     subjects_with_error.append(subject)
#
#     print('#done_subjects: {}'.format(len(set(done_subjects))))
#     good_subjects = [s for s in good_subjects if
#            op.isfile(op.join(MMVT_DIR, subject, 'meg',
#                              'labels_data_msit_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em))) and
#            op.isfile(op.join(MMVT_DIR, subject, 'meg',
#                              'labels_data_ecr_{}_{}_{}_minmax.npz'.format(atlas, inv_method, em)))]
#     print('Good subjects:')
#     print(good_subjects)
#     print('subjects_with_error:')
#     print(subjects_with_error)


# @utils.profileit(root_folder=op.join(MMVT_DIR, 'profileit'))
def post_meg_preproc(args):
    inv_method, em, atlas = 'dSPM', 'mean_flip', args.atlas
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    evoked_times = (500, 2500)
    baseline_times = (0, 500)
    do_plot = False

    subjects = args.subject
    res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr'))
    template_brain = 'colin27'
    subjects_results = {}
    bands_power_mmvt_all = []

    params = [(subject, atlas, bands, args.max_epochs_num, evoked_times, baseline_times, inv_method, em, args.tasks,
               template_brain, res_fol, args.n_jobs) for subject in args.subject]
    parallel_results = utils.run_parallel(_post_meg_preproc_parallel, params, args.n_jobs, print_time_to_go=True)
    for subject, results, bands_power_mmvt in parallel_results:
        subjects_results[subject] = results
        if all(results.values()):
            bands_power_mmvt_all.append(bands_power_mmvt)

    labels_data_template = op.join(MMVT_DIR, template_brain, 'meg', 'labels_data_power_{}_{}_{}_{}_{}.npz')  # task, atlas, extract_method, hemi
    labels = lu.read_labels(template_brain, SUBJECTS_DIR, atlas)
    hemi_labels_names = {hemi: [l.name for l in labels if l.hemi == hemi] for hemi in utils.HEMIS}
    for hemi in utils.HEMIS:
        for band_ind, band in enumerate(bands.keys()):
            power = np.array([x[hemi][band] for x in bands_power_mmvt_all]).mean(axis=0)
            labels_output_fname = meg.get_labels_data_fname(
                labels_data_template, inv_method, band, atlas, em, hemi)
            utils.make_dir(utils.get_parent_fol(labels_output_fname))
            np.savez(labels_output_fname, data=power, names=hemi_labels_names[hemi], conditions=args.tasks)

    have_all = len([subject for subject, results in subjects_results.items() if all(results.values())])
    print('{}/{} with all files'.format(have_all, len(subjects)))
    print(subjects_results)


def _post_meg_preproc_parallel(p):
    (subject, atlas, bands, epochs_max_num, evoked_times, baseline_times, inv_method, em, tasks, labels_template,
        res_fol, n_jobs) = p
    results = {}
    labels = lu.read_labels(labels_template, SUBJECTS_DIR, atlas)
    labels_names = [l.name for l in labels]
    hemi_labels_names = {hemi: [l.name for l in labels if l.hemi == hemi] for hemi in utils.HEMIS}
    labels_num = len(labels_names)
    labels_bands_avg = np.zeros((len(labels), len(bands)))
    labels_bands_avg_ind = np.zeros((len(labels), len(bands)))
    baseline = np.zeros((len(bands), len(tasks)))
    baseline_ind = np.zeros((len(bands), len(tasks)))
    input_fol = utils.make_dir(op.join(MEG_DIR, subject, 'labels_induced_power'))
    plots_fol = utils.make_dir(op.join(input_fol, 'plots'))
    bands_power_time = {'rh': {}, 'lh': {}}
    for task_ind, task in enumerate(tasks):
        task = task.lower()
        input_fnames = glob.glob(
            op.join(input_fol, '{}_*_{}_{}_{}_induced_power.npz'.format(task, atlas, inv_method, em)))
        print('{}, {}, {}'.format(subject, task, len(input_fnames)))
        if len(input_fnames) < labels_num:
            results[task] = False
            continue
        bands_power = np.empty((len(bands), labels_num, epochs_max_num))
        for input_fname in input_fnames if n_jobs > 1 else tqdm(input_fnames):
            if not results.get(task, True):
                break
            d = utils.Bag(np.load(input_fname))  # label_name, atlas, data
            label_power, label_name = d.data, d.label_name
            hemi = lu.get_hemi_from_name(label_name)
            if label_name in hemi_labels_names[hemi] and label_name in labels_names:
                hemi_label_ind = hemi_labels_names[hemi].index(label_name)
                label_ind = labels_names.index(label_name)
            else:
                print('label {} not in atlas!'.format(label_name))
                results[task] = False
                break
            for band_ind, band in enumerate(bands.keys()):
                baseline[band_ind, task_ind] += np.mean(label_power[band_ind][:, baseline_times[0]:baseline_times[1]])
                baseline_ind[band_ind, task_ind] += 1

                # Save for mmvt to plot the power over time
                if band not in bands_power_time[hemi]:
                    bands_power_time[hemi][band] = np.empty(
                        (len(hemi_labels_names[hemi]), label_power[band_ind].shape[1], len(tasks)))
                bands_power_time[hemi][band][hemi_label_ind, :, task_ind] = label_power[band_ind].mean(axis=0)

                # Save for statistics
                label_power_evoked = label_power[band_ind][:, evoked_times[0]:evoked_times[1]].mean(axis=1)[:epochs_max_num]
                labels_bands_avg[label_ind, band_ind] += np.mean(label_power[band_ind])
                labels_bands_avg_ind[label_ind, band_ind] += 1

                if len(label_power_evoked) != epochs_max_num:
                    # print('{} does have {} epochs!'.format(input_fname, len(label_power_evoked)))
                    continue
                bands_power[band_ind, label_ind] = label_power_evoked
            # fig_fname = op.join(plots_fol, 'power_{}_{}.jpg'.format(label_name, task))
            # if do_plot: # not op.isfile(fig_fname) and
            #     times = np.arange(0, label_power.shape[2]) if 'times' not in d else d.times
            #     plot_label_power(label_power, times, label_name, bands, task, fig_fname)
        if not results.get(task, True):
            break
        for band_ind, band in enumerate(bands.keys()):
            avg_baseline = baseline[band_ind, task_ind] / baseline_ind[band_ind, task_ind]
            band_power = np.array(bands_power[band_ind]) - avg_baseline
            power_fname = op.join(
                res_fol, subject, '{}_labels_{}_{}_{}_power.npz'.format(task.lower(), inv_method, em, band))
            np.savez(power_fname, data=band_power, names=labels_names)
            results[task] = True

            for hemi in utils.HEMIS:
                bands_power_time[hemi][band][:, :, task_ind] -= avg_baseline

    if all(results.values()):
        power_meta_fname = op.join(res_fol, subject, 'labels_{}_{}_meta_power.npz'.format(inv_method, em))
        np.savez(power_meta_fname, labels_bands_avg=labels_bands_avg, labels_bands_avg_ind=labels_bands_avg_ind,
                 baseline=baseline, baseline_ind=baseline_ind)
    else:
        print('{} does not have both tasks data!'.format(subject))

    print('unique(baseline_ind):', np.unique(baseline_ind))
    print('unique(labels_bands_avg_ind):', np.unique(labels_bands_avg_ind))
    return subject, results, bands_power_time


def plot_label_power(power, times, label, bands, task, fig_fname):
    # plt.figure()
    f, axs = plt.subplots(5, 1, sharex=True)
    for band_ind, (band_name, ax) in enumerate(zip(bands.keys(), axs)):
        power_mean = power[band_ind].mean(0)
        power_std = power[band_ind].std(0)
        ax.plot(times, power_mean)
        ax.fill_between(times, power_mean - power_std, power_mean + power_std, alpha=.5)
        ax.set_title(band_name)
    print('Saving {}'.format(fig_fname))
    plt.savefig(fig_fname)
    plt.close()


def calc_meg_connectivity(args):
    inv_method, em = 'dSPM', 'mean_flip'
    con_method, con_mode = 'coh', 'cwt_morlet'
    prepare_files(args)
    good_subjects = get_good_subjects(args)
    args.subject = good_subjects

    for subject in good_subjects:
        args.subject = subject
        for task in args.tasks:

            output_fname = op.join(
                MMVT_DIR, subject, 'connectivity', '{}_{}_{}_{}.npz'.format(task.lower(), em, con_method, con_mode))
            if op.isfile(output_fname):
                file_mod_time = utils.file_modification_time_struct(output_fname)
                if file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 11 and file_mod_time.tm_mday >= 6) or \
                        (file_mod_time.tm_mon > 11):
                    print('{} already exist!'.format(output_fname))
                    continue

            remote_epo_fname = op.join(args.meg_dir, subject, args.epo_template.format(subject=subject, task=task))
            local_epo_fname = op.join(MEG_DIR, task, subject, args.epo_template.format(subject=subject, task=task))
            if not op.isfile(local_epo_fname):
                utils.make_link(remote_epo_fname, local_epo_fname)

            con_args = meg.read_cmd_args(utils.Bag(
                subject=args.subject, mri_subject=args.subject,
                task=task, inverse_method=inv_method, extract_mode=em, atlas=args.atlas,
                # meg_dir=args.meg_dir,
                remote_subject_dir=args.remote_subject_dir,  # Needed for finding COR
                get_task_defaults=False,
                data_per_task=True,
                fname_format=args.epo_template.format(subject=subject, task=task)[:-len('-epo.fif')],
                raw_fname=op.join(MEG_DIR, task, subject, args.raw_template.format(subject=subject, task=task)),
                epo_fname=local_epo_fname,
                # empty_fname=empty_fnames[task],
                function='calc_labels_connectivity',
                conditions=task.lower(),
                max_epochs_num=args.max_epochs_num,
                overwrite_connectivity=True,#args.overwrite_connectivity,
                # cor_fname=cors[task].format(subject=subject),
                # ica_overwrite_raw=False,
                # normalize_data=False,
                # t_min=times[0], t_max=times[1],
                # read_events_from_file=False, stim_channels='STI001',
                # use_empty_room_for_noise_cov=True,
                # read_only_from_annot=False,
                # pick_ori='normal',
                # overwrite_evoked=args.overwrite,
                # overwrite_inv=args.overwrite,
                # overwrite_stc=args.overwrite,
                # overwrite_labels_data=args.overwrite,
                n_jobs=args.n_jobs
            ))
            meg.call_main(con_args)


def sensors_ttest(args):
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    alpha = 0.05
    good_subjects = 0
    mean_psd_task = {}
    for task in args.tasks:
        mean_psd_task[task] = defaultdict(list)
    now = time.time()
    for ind, subject in enumerate(args.subject):
        # utils.time_to_go(now, ind, len(args.subject), 1)
        fol = op.join(MMVT_DIR, subject, 'meg')
        band_data_exist = 0
        for band_ind, band in enumerate(bands.keys()):
            psds = {task: None for task in args.tasks}
            for task in args.tasks:
                # Loading baseline
                baseline_fname = op.join(
                    MMVT_DIR, subject, 'meg', 'baseline_{}_sensors_{}_psd.npz'.format(task.lower(), band))
                if not op.isfile(baseline_fname):
                    continue
                baseline_dict = utils.Bag(np.load(baseline_fname))
                # Setting nan as mean
                baseline_dict.data[np.where(np.isnan(baseline_dict.data))] = np.mean(baseline_dict.data)
                input_fname = op.join(fol, '{}_sensors_{}_psd.npz'.format(task.lower(), band))
                if not op.isfile(input_fname):
                    continue
                d = utils.Bag(np.load(input_fname))
                # normalize the psd with the baseline (for each sensors)
                psds[task] = (d.data.T / baseline_dict.data).T
                mean_psd_task[task][band].append(np.nanmean(psds[task]))
            if all([psds[task] is not None for task in args.tasks]):
                band_data_exist += 1
                x = [psds[args.tasks[0]], psds[args.tasks[1]]]
                # for sens_ind in range(x[0].shape[0]):
                #     title = '{} {} sens {}'.format(subject, band, sens_ind)
                #     sig, pval, = ttest(x[0][sens_ind], x[1][sens_ind],
                #                        args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=False)
                title = '{} {}'.format(subject, band)
                sig, pval, = ttest(x[0].mean(axis=0), x[1].mean(axis=0),
                                   args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=False)
        if band_data_exist == len(bands):
            good_subjects += 1
    print('{}/{} subjects with sensors psd'.format(good_subjects, len(args.subject)))

    for band in bands.keys():
        x = [np.array(mean_psd_task[task][band]) for task in args.tasks]
        # x = clean_power(x, band, percentile, high_limit_power)
        if x is None:
            print('band {} is None!'.format(band))
            continue
        title='{} vs {} band {}'.format(args.tasks[0], args.tasks[1], band)
        sig, pval, = ttest(x[0], x[1], args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=True)


def post_analysis(args):
    import matplotlib.pyplot as plt
    from collections import defaultdict
    from operator import mul

    def check_labels_bands_avg_ind(meta, label_ind, band_ind):
        if meta.labels_bands_avg_ind[label_ind, band_ind] != 2:
            print('{}: label {} band {} has wonrg labels_bands_avg! ({})'.format(
                subject, label, band, meta.labels_bands_avg_ind[label_ind, band_ind]))
            return False
        else:
            return True

    def check_baseline_ind(meta, band_ind, task_ind):
        if meta.baseline_ind[band_ind, task_ind] != 223:
            print('{}: band {} has wrong baseline! ({})'.format(subject, band, meta.baseline_ind[band_ind, task_ind]))
            return False
        else:
            return True

    inv_method, em = 'dSPM', 'mean_flip'
    res_fol = utils.make_dir(op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr'))
    plot_fol = utils.make_dir(op.join(res_fol, 'plots'))
    bands = dict(theta=[4, 8], alpha=[8, 15], beta=[15, 30], gamma=[30, 55], high_gamma=[65, 120])
    data_dic = np.load(op.join(res_fol, 'data_dictionary.npz'))
    meta_data = data_dic['noam_dict'].tolist()
    # brain_overall_res_fname = op.join(res_fol, 'brain_overall_res.npz')
    msit_subjects = set(meta_data[0]['MSIT'].keys())
    ecr_subjects = set(meta_data[1]['ECR'].keys())
    subjects_with_data = defaultdict(list)
    mean_evo = {group_id:defaultdict(list) for group_id in range(2)}
    mean_power_power_emotion_reactivit = {group_id: {} for group_id in range(2)}
    power_emotion_reactivit = {group_id: {} for group_id in range(2)}

    do_plot = False
    percentile = 90
    alpha = 0.05
    high_limit_power = 1000

    mean_power_power_task = {}
    power_task = {}
    no_norm_subjects = 0
    for task_ind, task in enumerate(args.tasks):
        mean_power_power_task[task] = defaultdict(list)
        power_task[task] = {band: None for band in bands.keys()}
    norm_dict = defaultdict(dict)
    bad_subjects = ['hc006', 'hc011', 'pp001', 'ep011']
    good_subjects = set(args.subject) - set(bad_subjects)
    now = time.time()
    for ind, subject in enumerate(good_subjects):
        # utils.time_to_go(now, ind, len(args.subject), 1)
        for band_ind, band in enumerate(bands.keys()):
            ecpohs_power = {}
            for task in args.tasks:
                if power_task[task][band] is None:
                    power_task[task][band] = defaultdict(list)
                if band not in norm_dict[task]:
                    norm_dict[task][band] = []
                power_fname = op.join(
                    MMVT_DIR, subject, 'labels', 'labels_data', '{}_labels_{}_{}_{}_power.npz'.format(
                        task.lower(), inv_method, em, band))
                file_mod_time = utils.file_modification_time_struct(power_fname)
                if not (file_mod_time.tm_year >= 2018 and (file_mod_time.tm_mon == 9 and file_mod_time.tm_mday >= 21) or \
                        (file_mod_time.tm_mon > 9)):
                    print('Old file! {}'.format(power_fname))
                    continue
                if op.isfile(power_fname):
                    try:
                        d = utils.Bag(np.load(power_fname))
                        if d.data.mean() > 1000:
                            continue
                        mean_power_power_task[task][band].append(d.data.mean())
                        ecpohs_power[task] = d.data
                        for label_ind, label in enumerate(d.names):
                            power_task[task][band][label].append(d.data[label_ind].mean())
                    except:
                        print('Can\'t open {}!'.format(power_fname))
            if all([t in ecpohs_power for t in args.tasks]):
                x = [ecpohs_power[args.tasks[0]], ecpohs_power[args.tasks[1]]]
                title = '{} {}'.format(subject, band)
                sig, pval, = ttest(x[0].mean(axis=0), x[1].mean(axis=0),
                                   args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=False)
                # for label_id, label in enumerate(d.names):
                #     title = '{} {} {}'.format(subject, band, label)
                #     sig, pval, = ttest(ecpohs_power[args.tasks[0]][label_id], ecpohs_power[args.tasks[1]][label_id],
                #                        args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=False)
            del  ecpohs_power
    # for task in args.tasks:
    #     for band_id, band in enumerate(bands.keys()):
    #         num = np.mean([len(power_task[task][band][label]) for label in d.names])
    #         print('power_task: {} {} {} items, norm: {}'.format(task, band, num, np.mean(norm_dict[task][band])))


    # for group_id in range(2):
    #     for task in args.tasks:
    #         mean_power_power_emotion_reactivit[group_id][task] = defaultdict(list)
    #         power_emotion_reactivit[group_id][task] = {band: None for band in bands.keys()}
    #     for subject in meta_data[group_id]['ECR'].keys():
    #         if not op.isdir(op.join(res_fol, subject)):
    #             print('No folder data for {}'.format(subject))
    #             continue
    #         for task in args.tasks:
    #             mean_fname = op.join(res_fol, subject, '{}_{}_mean.npz'.format(task.lower(), args.atlas))
    #             if op.isfile(mean_fname):
    #                 d = utils.Bag(np.load(mean_fname))
    #                 mean_evo[group_id][task].append(d.data.mean())
    #             for band in bands.keys():
    #                 if power_emotion_reactivit[group_id][task][band] is None:
    #                     power_emotion_reactivit[group_id][task][band] = defaultdict(list)
    #                 power_fname = op.join(
    #                     res_fol, subject, '{}_labels_{}_{}_{}_power.npz'.format(task.lower(), inv_method, em, band))
    #                 if op.isfile(power_fname):
    #                     d = utils.Bag(np.load(power_fname))
    #                     mean_power_power_emotion_reactivit[group_id][task][band].append(d.data.mean())
    #                     for label_id, label in enumerate(d.names):
    #                         power_emotion_reactivit[group_id][task][band][label].append(d.data[label_id].mean())

    for band in bands.keys():
        ttest_stats, ttest_labels, welch_stats, welch_labels = [], [], [], []
        x = [np.array(mean_power_power_task[task][band]) for task in args.tasks]
        x = clean_power(x, band, percentile, high_limit_power)
        if x is None:
            print('band {} is None!'.format(band))
            continue
        title='{} vs {} band {}'.format(args.tasks[0], args.tasks[1], band)
        sig, pval, = ttest(x[0], x[1], args.tasks[0], args.tasks[1], title=title, alpha=alpha, always_print=True)
        if do_plot: # or sig:
            f, (ax1, ax2) = plt.subplots(2, 1)
            ax1.hist(x[0], bins=80)
            ax1.set_title('{} {}'.format(band, args.tasks[0]))
            ax2.hist(x[1], bins=80)
            ax2.set_title('{} {}'.format(band, args.tasks[1]))
            # plt.title('{} mean power'.format(band))
            # plt.show()
            plt.savefig(op.join(plot_fol, '{}.jpg'.format(band)))

        for label_id, label in enumerate(d.names):
            x = [np.array(power_task[task][band][label]) for task in args.tasks]
            x = clean_power(x, band, percentile, high_limit_power)
            if x is None:
                # print('band: {} label: {} is None!'.format(band, label))
                continue
            title = '{} {}'.format(band, label)
            sig, pval = ttest(x[0], x[1], args.tasks[0], args.tasks[1], alpha=alpha, title=title, always_print=False)
            if sig:
                welch_stats.append(pval)
                welch_labels.append(label)
            if do_plot:
                f, (ax1, ax2) = plt.subplots(2, 1)
                ax1.hist(x[0], bins=80)
                ax2.hist(x[1], bins=80)
                plt.title('{} mean power'.format(band))
                # plt.show()
                plt.savefig(op.join(plot_fol, '{}_{}.jpg'.format(band, label)))

        labels_data_fol = utils.make_dir(op.join(MMVT_DIR, 'colin27', 'labels', 'labels_data'))
        np.savez(op.join(labels_data_fol, 'MSIT_ECR_{}.npz'.format(band)), names=np.array(welch_labels),
                 atlas=args.atlas, data=np.array(welch_stats), title='MSIT vs ECR',
                 data_min=0, data_max=0.05, cmap='YlOrRd')
        continue

        for group_id in range(2): #, ax in zip(range(2), [ax1, ax2]):
            # subjects_with_data[group_id] = np.array(subjects_with_data[group_id])
            # print()
            x = [np.array(mean_power_power_emotion_reactivit[group_id][task][band]) for task in args.tasks]
            # x = [_x[_x < np.percentile(_x, 90)] for _x in x]
            x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
            x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
            print('band {}, group {}, {} for {}, {} for {}'.format(
                band, group_id, len(x[0]), args.tasks[0], len(x[1]), args.tasks[1]))
            ttest(x[0], x[1], title='group {} band {}'.format(group_id, band), alpha=alpha)
            if do_plot:
                f, (ax1, ax2) = plt.subplots(2, 1)
                ax1.hist(x[0], bins=80)
                ax2.hist(x[1], bins=80)
                plt.title('{} mean power'.format(band))
                plt.savefig(op.join(plot_fol, '{}_group_{}.jpg'.format(band, group_id)))

            for label_id, label in enumerate(d.names):
                x = [np.array(power_emotion_reactivit[group_id][task][band][label]) for task in args.tasks]
                # x = [_x[_x < np.percentile(_x, 90)] for _x in x]
                x[0] = x[0][x[0] < np.percentile(x[0], percentile)]
                x[1] = x[1][x[1] < np.percentile(x[1], percentile)]
                ttest(x[0], x[1], alpha=alpha, title='group {} band {} label {}'.format(group_id, band, label))
                if do_plot:
                    f, (ax1, ax2) = plt.subplots(2, 1)
                    ax1.hist(x[0], bins=80)
                    ax2.hist(x[1], bins=80)
                    plt.title('{} {} power'.format(band, label))
                    plt.savefig(op.join(plot_fol, '{}_group_{}_label_{}.jpg'.format(band, group_id, label)))
        # ax.set_title('group {}'.format(group_id))
        # ax.bar(np.arange(2), [np.mean(mean_power[group_id][task][band]) for task in args.tasks])
        # ax.set_xticklabels(args.tasks, rotation=30)
        # fig.suptitle('{} mean_power'.format(band))
        # plt.show()


def clean_power(x, band, percentile, high_limit_power, do_print=True):
    for ind in range(2):
        x[ind] = x[ind][np.where(~np.isnan(x[ind]))]
        x[ind] = x[ind][np.where(~np.isinf(x[ind]))]
        x[ind] = x[ind][np.where((x[ind] >= 0))[0]]
        x[ind] = x[ind][np.where((x[ind] < high_limit_power))[0]]
        if len(x[ind]) > 0:
            x[ind] = x[ind][x[ind] < np.percentile(x[ind], percentile)]
        else:
            x = None
            break
    # if x is not None and do_print:
    #     print('{} {} ({}): {}-{}, {} ({}): {}-{}'.format(
    #         band, args.tasks[0], len(x[0]), np.min(x[0]), np.max(x[0]),
    #         args.tasks[1], len(x[1]), np.min(x[1]), np.max(x[1])))
    return x


def ttest(x1, x2, x1_name, x2_name, two_tailed_test=True, alpha=0.05, is_greater=True, title='',
          calc_welch=True, long_print=True, always_print=False):
    import scipy.stats
    t, pval = scipy.stats.ttest_ind(x1, x2, equal_var=not calc_welch)
    sig = is_significant(pval, t, two_tailed_test, alpha, is_greater)
    if sig or always_print:
        long_str = '#{} {:.4f}+-{:.4f}, #{} {:.4f}+-{:.4f}'.format(
            len(x1), np.mean(x1), np.std(x1), len(x2), np.mean(x2), np.std(x2)) if long_print else ''
        print('{}: {} {} {} ({:.6f}) {}'.format(title, x1_name, '>' if t > 0 else '<', x2_name, pval, long_str))

    return sig, pval


def is_significant(pval, t, two_tailed_test, alpha=0.05, is_greater=True):
    if two_tailed_test:
        return pval < alpha
    else:
        if is_greater:
            return pval / 2 < alpha and t > 0
        else:
            return pval / 2 < alpha and t < 0

    # for subject, task, band in product(args.subject, tasks, bands.keys()):
    #     mean_power_fol = op.join(MMVT_DIR, subject, 'labels', 'labels_data')
    #     d = utils.Bag(np.load(op.join(mean_power_fol, '{}_mean_power_{}.npz'.format(task.lower(), band))))
    #     for label_name, label_data in zip(d.names, d.data):
    #         hemi = lu.get_label_hemi(label_name)
    #         plt.figure()
    #         plt.axhline(label_data * 1e5, color='r', linestyle='--')
    #         plt.show()
    #         plt.close()
    #         print('asdf')
    #


def get_good_subjects(args, check_dict=False):
    if check_dict:
        data_dict_fname = op.join(args.remote_root_dir, 'data_dictionary.npz')
        if not op.isfile(data_dict_fname):
            ret = input('No data dict, do you want to continue? (y/n)')
            if not au.is_true(ret):
                return
            msit_ecr_subjects = args.subject
        else:
            data_dic = np.load(op.join(args.remote_root_dir, 'data_dictionary.npz'))
            meta_data = data_dic['noam_dict'].tolist()
            msit_subjects = set(meta_data[0]['MSIT'].keys()) | set(meta_data[1]['MSIT'].keys())
            ecr_subjects = set(meta_data[0]['ECR'].keys()) | set(meta_data[1]['ECR'].keys())
            msit_ecr_subjects = msit_subjects.intersection(ecr_subjects)
    else:
        msit_ecr_subjects = set()

    if not args.check_files:
        good_subjects = args.subject
    else:
        good_subjects = []
        for subject in args.subject:
            # if subject == 'pp009':
            #     continue
            if check_dict and subject not in msit_ecr_subjects:
                print('*** {} not in the meta data!'.format(subject))
                continue
            if not op.isdir(args.remote_subject_dir.format(subject=subject)) and not op.isdir(op.join(SUBJECTS_DIR, subject)):
                print('*** {}: No recon-all files!'.format(subject))
                continue
            if args.anatomy_preproc:
                anatomy_preproc(args, subject)
            if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', args.atlas))):
                anatomy_preproc(args, subject)
            if not utils.both_hemi_files_exist(op.join(SUBJECTS_DIR, subject, 'label', '{}.{}.annot'.format('{hemi}', args.atlas))):
                print('*** Can\'t find the atlas {}!'.format(args.atlas))
                continue
            empty_fnames, cors, days = get_empty_fnames(subject, args.tasks, args)
            if empty_fnames == '' or cors == '' or days == '':
                print('{}: Error with get_empty_fnames!'.format(subject))
            if any([task not in cors for task in args.tasks]) and  args.check_cor:
                print('*** {}: one of the tasks does not have a cor transformation matrix!'.format(subject))
                print(cors)
                continue
            files_exist = \
                op.isfile(op.join(args.meg_dir.format(task='ECR'), subject, args.epo_template.format(subject=subject, task='ECR'))) and \
                op.isfile(op.join(args.meg_dir.format(task='MSIT'), subject, args.epo_template.format(subject=subject, task='MSIT')))
            if not files_exist and args.check_for_both_files:
                print('**** {} doesn\'t have both MSIT and ECR files!'.format(subject))
                continue
            # for task in args.tasks:
            #     print('{}: empty: {}, cor: {}'.format(subject, empty_fnames[task], cors[task].format(subject=subject)))
            good_subjects.append(subject)
        print('Good subjects: ({}):'.format(len(good_subjects)))
        print(good_subjects)
        bad_subjects = set(args.subject) - set(good_subjects)
        print('Bad subjects: ({}):'.format(len(bad_subjects)))
        print(bad_subjects)
    return good_subjects


if __name__ == '__main__':
    import argparse
    from src.utils import args_utils as au
    parser = argparse.ArgumentParser(description='MMVT')
    parser.add_argument('-s', '--subject', help='subject name', required=True, type=au.str_arr_type)
    parser.add_argument('-f', '--function', help='function name', required=False, default='meg_preproc_power')
    parser.add_argument('-a', '--atlas', help='atlas name', required=False, default='laus125') #darpa-atlas')
    parser.add_argument('-t', '--tasks', help='tasks', required=False, default='MSIT,ECR', type=au.str_arr_type)
    parser.add_argument('-i', '--inverse_method', help='inverse_method', required=False, default='dSPM',
                        type=au.str_arr_type)
    parser.add_argument('--overwrite', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_output_files', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_local_files', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_connectivity', required=False, default=False, type=au.is_true)
    parser.add_argument('--overwrite_labels_power_spectrum', required=False, default=False, type=au.is_true)
    parser.add_argument('--throw', required=False, default=False, type=au.is_true)
    parser.add_argument('--anatomy_preproc', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_files', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_file_modification_time', required=False, default=False, type=au.is_true)
    parser.add_argument('--check_cor', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_for_both_files', required=False, default=True, type=au.is_true)
    parser.add_argument('--check_for_channels_inconsistency', required=False, default=0, type=au.is_true)
    parser.add_argument('--average_over_label_indices', help='', required=False, default=1, type=au.is_true)
    parser.add_argument('--ignore_missing', help='ignore missing files', required=False, default=0, type=au.is_true)
    parser.add_argument('--max_epochs_num', help='', required=False, default=50, type=int)

    parser.add_argument('--remote_root_dir', required=False,
                        default='/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/')
    meg_dirs = ['/home/npeled/meg/{task}',
                '/autofs/space/cassia_004/users/MSIT_ECR_Preprocesing_for_Noam/epochs']
    meg_dir = [d for d in meg_dirs if op.isdir(d.format(task='MSIT'))][0]
    parser.add_argument('--meg_dir', required=False, default=meg_dir)
                        # default='/autofs/space/karima_001/users/alex/MSIT_ECR_Preprocesing_for_Noam/raw_preprocessed')
    remote_subject_dirs = ['/autofs/space/lilli_001/users/DARPA-Recons/',
                           '/home/npeled/subjects']
    remote_subject_dir = [op.join(d, '{subject}') for d in remote_subject_dirs if op.isdir(d)][0]
    parser.add_argument('--remote_subject_dir', required=False, default=remote_subject_dir)
    parser.add_argument('--remote_meg_dir', required=False,
                        default='/autofs/space/lilli_003/users/DARPA-TRANSFER/meg')
    parser.add_argument('--epo_template', required=False, default='{subject}_{task}_meg_Onset_ar-epo.fif')
    parser.add_argument('--raw_template', required=False, default='{subject}_{task}_meg_ica-raw.fif')
    parser.add_argument('--n_jobs', help='cpu num', required=False, default=-1)
    args = utils.Bag(au.parse_parser(parser))
    args.n_jobs = utils.get_n_jobs(args.n_jobs)

    inv_method, em = 'dSPM', 'mean_flip'
    bad_subjects = ['ep006', 'ep005', 'sp004', 'ep007', 'hc004', 'ep004', 'ep016'] #, 'ep006', 'ep005', 'sp004', 'ep007', 'hc004']
    if args.subject[0] == 'all':
        if args.function == 'post_analysis':
            res_fol = op.join(utils.get_parent_fol(MMVT_DIR), 'msit-ecr')
            args.subject = utils.shuffle(
                [utils.namebase(d) for d in glob.glob(op.join(res_fol, '*')) if op.isdir(d) and
                 op.isfile(op.join(d, 'ecr_labels_dSPM_mean_flip_alpha_power.npz')) and
                 op.isfile(op.join(d, 'msit_labels_dSPM_mean_flip_alpha_power.npz'))])
        elif args.function == 'post_meg_preproc':
            args.subject = utils.shuffle(
                [utils.namebase(d) for d in glob.glob(op.join(MEG_DIR, '*')) if op.isdir(d) and
                 len(glob.glob(op.join(d, 'labels_induced_power',
                                       'ecr_*_{}_{}_{}_induced_power.npz'.format(args.atlas, inv_method, em)))) > 0 and
                 len(glob.glob(op.join(d, 'labels_induced_power',
                                       'msit_*_{}_{}_{}_induced_power.npz'.format(args.atlas, inv_method, em)))) > 0])
        else:
            args.subject = utils.shuffle(
                [utils.namebase(d) for d in glob.glob(op.join(args.meg_dir, '*')) if op.isdir(d) and
                 op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='ECR'))) and
                 op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='MSIT')))])
        print('{} subjects were found with both tasks!'.format(len(args.subject)))
        print(sorted(args.subject))
        for bad_subject in bad_subjects:
            if bad_subject in args.subject:
                args.subject.remove(bad_subject)
    elif '*' in args.subject[0]:
        args.subject = utils.shuffle(
            [utils.namebase(d) for d in glob.glob(op.join(args.meg_dir, args.subject[0])) if op.isdir(d) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='ECR'))) and
             op.isfile(op.join(d, args.epo_template.format(subject=utils.namebase(d), task='MSIT')))])
        print('{} subjects were found with both tasks:'.format(len(args.subject)))
        print(sorted(args.subject))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        locals()[args.function](args)



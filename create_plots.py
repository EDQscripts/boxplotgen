__author__ = 'Sol'
__version__ = 'RZ'

import numpy as np
import matplotlib.pylab as plt
import glob
import os
import re

BINOC_INCLUDE_TRACKERS = ()
RIGHT_INCLUDE_TRACKERS = ('dpi', 'eyefollower', 'eyelink', 'eyetribe', 'hispeed1250', 'hispeed240',
              'red250', 'red500', 'redm', 't60xl', 'tx300', 'x2')
RIGHT_INCLUDE_TRACKERS = ('hispeed1250')

INCLUDE_TRACKERS = RIGHT_INCLUDE_TRACKERS

PLOT_IV = ['window_onset','min_error','max_error','average_error', 'median_error', 'stdev_error']
COMBINE_XY = True
COMBINE_LEFT_RIGHT = False

binoc_file_dtype = [('subject_id', 'u1'), ('display_refresh_rate', '<f4'), ('eyetracker_model', 'S32'), ('dot_deg_sz', '<f4'), ('eyetracker_sampling_rate', '<f4'), ('eyetracker_mode', 'S16'), ('fix_stim_center_size_pix', 'u1'), ('operator', 'S8'), ('eyetracker_id', 'u1'), ('display_width_pix', '<f4'), ('display_height_pix', '<f4'), ('screen_width', '<f4'), ('screen_height', '<f4'), ('eye_distance', '<f4'), ('SESSION_ID', 'u1'), ('trial_id', '<u2'), ('TRIAL_START', '<f4'), ('TRIAL_END', '<f4'), ('posx', '<f4'), ('posy', '<f4'), ('dt', '<f4'), ('ROW_INDEX', 'u1'), ('session_id', 'u1'), ('device_time', '<f4'), ('time', '<f4'), ('left_gaze_x', '<f4'), ('left_gaze_y', '<f4'), ('left_pupil_measure1', '<f4'), ('right_gaze_x', '<f4'), ('right_gaze_y', '<f4'), ('right_pupil_measure1', '<f4'), ('status', 'u1'), ('target_angle_x', '<f4'), ('target_angle_y', '<f4'), ('left_angle_x', '<f4'), ('left_angle_y', '<f4'), ('right_angle_x', '<f4'), ('right_angle_y', '<f4'), ('left_fiona_100', 'u1'), ('left_dixon1_100', 'u1'), ('left_dixon2_100', 'u1'), ('left_dixon3_100', 'u1'), ('left_jeff_100', 'u1'), ('right_fiona_100', 'u1'), ('right_dixon1_100', 'u1'), ('right_dixon2_100', 'u1'), ('right_dixon3_100', 'u1'), ('right_jeff_100', 'u1'), ('left_fiona_175', 'u1'), ('left_dixon1_175', 'u1'), ('left_dixon2_175', 'u1'), ('left_dixon3_175', 'u1'), ('left_jeff_175', 'u1'), ('right_fiona_175', 'u1'), ('right_dixon1_175', 'u1'), ('right_dixon2_175', 'u1'), ('right_dixon3_175', 'u1'), ('right_jeff_175', 'u1')]

binoc_file_dtype = [('subject_id', 'u1'), ('display_refresh_rate', '<f4'), ('eyetracker_model', 'S32'), ('dot_deg_sz', '<f4'), ('eyetracker_sampling_rate', '<f4'), ('eyetracker_mode', 'S16'), ('fix_stim_center_size_pix', 'u1'), ('operator', 'S8'), ('eyetracker_id', 'u1'), ('display_width_pix', '<f4'), ('display_height_pix', '<f4'), ('screen_width', '<f4'), ('screen_height', '<f4'), ('eye_distance', '<f4'), ('SESSION_ID', 'u1'), ('trial_id', '<u2'), ('TRIAL_START', '<f4'), ('TRIAL_END', '<f4'), ('posx', '<f4'), ('posy', '<f4'), ('dt', '<f4'), ('ROW_INDEX', 'u1'), ('BLOCK', 'S6'), ('session_id', 'u1'), ('device_time', '<f4'), ('time', '<f4'), ('left_gaze_x', '<f4'), ('left_gaze_y', '<f4'), ('left_pupil_measure1', '<f4'), ('right_gaze_x', '<f4'), ('right_gaze_y', '<f4'), ('right_pupil_measure1', '<f4'), ('status', 'u1'), ('target_angle_x', '<f4'), ('target_angle_y', '<f4'), ('left_angle_x', '<f4'), ('left_angle_y', '<f4'), ('right_angle_x', '<f4'), ('right_angle_y', '<f4'), ('left_fiona_100', 'u1'), ('left_dixon1_100', 'u1'), ('left_dixon2_100', 'u1'), ('left_dixon3_100', 'u1'), ('left_jeff_100', 'u1'), ('right_fiona_100', 'u1'), ('right_dixon1_100', 'u1'), ('right_dixon2_100', 'u1'), ('right_dixon3_100', 'u1'), ('right_jeff_100', 'u1'), ('left_fiona_175', 'u1'), ('left_dixon1_175', 'u1'), ('left_dixon2_175', 'u1'), ('left_dixon3_175', 'u1'), ('left_jeff_175', 'u1'), ('right_fiona_175', 'u1'), ('right_dixon1_175', 'u1'), ('right_dixon2_175', 'u1'), ('right_dixon3_175', 'u1'), ('right_jeff_175', 'u1')]
binoc_file_dtype = [('subject_id', 'u1'), ('display_refresh_rate', '<f4'), ('eyetracker_model', 'S32'), ('dot_deg_sz', '<f4'), ('eyetracker_sampling_rate', '<f4'), ('eyetracker_mode', 'S16'), ('fix_stim_center_size_pix', 'u1'), ('operator', 'S8'), ('eyetracker_id', 'u1'), ('display_width_pix', '<f4'), ('display_height_pix', '<f4'), ('screen_width', '<f4'), ('screen_height', '<f4'), ('eye_distance', '<f4'), ('SESSION_ID', 'u1'), ('trial_id', '<u2'), ('TRIAL_START', '<f4'), ('TRIAL_END', '<f4'), ('posx', '<f4'), ('posy', '<f4'), ('dt', '<f4'), ('ROW_INDEX', 'u1'), ('session_id', 'u1'), ('device_time', '<f4'), ('time', '<f4'), ('left_gaze_x', '<f4'), ('left_gaze_y', '<f4'), ('left_pupil_measure1', '<f4'), ('right_gaze_x', '<f4'), ('right_gaze_y', '<f4'), ('right_pupil_measure1', '<f4'), ('status', 'u1'), ('target_angle_x', '<f4'), ('target_angle_y', '<f4'), ('left_angle_x', '<f4'), ('left_angle_y', '<f4'), ('right_angle_x', '<f4'), ('right_angle_y', '<f4'), ('left_fiona_100', 'u1'), ('left_dixon1_100', 'u1'), ('left_dixon2_100', 'u1'), ('left_dixon3_100', 'u1'), ('left_jeff_100', 'u1'), ('right_fiona_100', 'u1'), ('right_dixon1_100', 'u1'), ('right_dixon2_100', 'u1'), ('right_dixon3_100', 'u1'), ('right_jeff_100', 'u1'), ('left_fiona_175', 'u1'), ('left_dixon1_175', 'u1'), ('left_dixon2_175', 'u1'), ('left_dixon3_175', 'u1'), ('left_jeff_175', 'u1'), ('right_fiona_175', 'u1'), ('right_dixon1_175', 'u1'), ('right_dixon2_175', 'u1'), ('right_dixon3_175', 'u1'), ('right_jeff_175', 'u1')]




right_file_dtype = [('subject_id', 'u1'), ('display_refresh_rate', '<f4'), ('eyetracker_model', 'S32'), ('dot_deg_sz', '<f4'), ('eyetracker_sampling_rate', '<f4'), ('eyetracker_mode', 'S16'), ('fix_stim_center_size_pix', 'u1'), ('operator', 'S8'), ('eyetracker_id', 'u1'), ('display_width_pix', '<f4'), ('display_height_pix', '<f4'), ('screen_width', '<f4'), ('screen_height', '<f4'), ('eye_distance', '<f4'), ('SESSION_ID', 'u1'), ('trial_id', '<u2'), ('TRIAL_START', '<f4'), ('TRIAL_END', '<f4'), ('posx', '<f4'), ('posy', '<f4'), ('dt', '<f4'), ('ROW_INDEX', 'u1'), ('session_id', 'u1'), ('device_time', '<f4'), ('time', '<f4'), ('left_gaze_x', '<f4'), ('left_gaze_y', '<f4'), ('left_pupil_measure1', '<f4'), ('right_gaze_x', '<f4'), ('right_gaze_y', '<f4'), ('right_pupil_measure1', '<f4'), ('status', 'u1'), ('target_angle_x', '<f4'), ('target_angle_y', '<f4'), ('left_angle_x', '<f4'), ('left_angle_y', '<f4'), ('right_angle_x', '<f4'), ('right_angle_y', '<f4'), ('right_fiona_100', 'u1'), ('right_dixon1_100', 'u1'), ('right_dixon2_100', 'u1'), ('right_dixon3_100', 'u1'), ('right_jeff_100', 'u1'),('right_fiona_175', 'u1'), ('right_dixon1_175', 'u1'), ('right_dixon2_175', 'u1'), ('right_dixon3_175', 'u1'), ('right_jeff_175', 'u1')]

input_file_dtype = binoc_file_dtype
INPUT_FILE_ROOT = r"/media/Data/EDQ/data_npy/"

binoc_type_cols = ['left_fiona_100','right_fiona_100',
                    'left_dixon1_100','right_dixon1_100',
                    'left_dixon2_100','right_dixon2_100',
                    'left_dixon3_100','right_dixon3_100',
                    'left_jeff_100', 'right_jeff_100',
                    'left_fiona_175', 'right_fiona_175',
                    'left_dixon1_175','right_dixon1_175',
                    'left_dixon2_175','right_dixon2_175',
                    'left_dixon3_175','right_dixon3_175',
                    'left_jeff_175', 'right_jeff_175',
                    ]

right_type_cols = ['right_fiona_100',
                   'right_dixon1_100',
                    'right_dixon2_100',
                   'right_dixon3_100',
                     'right_jeff_100',
                     'right_fiona_175',
                    'right_dixon1_175',
                    'right_dixon2_175',
                    'right_dixon3_175',
                     'right_jeff_175',
                    ]

window_type_cols = right_type_cols


def nabs(file_path):
    """

    :param file_path:
    :return:
    """
    return os.path.normcase(os.path.normpath(os.path.abspath(file_path)))

def analyseit(fpath):
    """

    :param fpath:
    :return:
    """
    tracker_type, sub = getInfoFromPath(fpath)
    return (tracker_type in INCLUDE_TRACKERS)

def getInfoFromPath(fpath):
    """

    :param fpath:
    :return:
    """
    _, fname = os.path.split(fpath)
    tracker_name = fname.split('_')[0]
    subject = fname.split('_')[-2]
    return tracker_name, subject


def plot(et_name, iv_measure, single_eyetracker_results):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon


    # Generate some data from five different probability distributions,
    # each with different characteristics. We want to play with how an IID
    # bootstrap resample of the data preserves the distributional
    # properties of the original sample, and a boxplot is one visual tool
    # to make this assessment

    numDists = len(single_eyetracker_results.keys())
    distNames = single_eyetracker_results.keys()

    print 'distNames:',distNames
    data = single_eyetracker_results.values()

    ###########################################
    fig = plt.figure(figsize=(10,6))
    fig.canvas.set_window_title(et_name+' : '+iv_measure)
    ax1 = fig.add_subplot(111)
    plt.subplots_adjust(left=0.075, right=0.95, top=0.9, bottom=0.25)

    bp = plt.boxplot(data, notch=0, sym='', vert=1, whis=1.5)
    plt.setp(bp['boxes'], color='black')
    plt.setp(bp['whiskers'], color='blue')
    plt.setp(bp['fliers'], color='red', marker='+')

    # Add a horizontal grid to the plot, but make it very light in color
    # so we can use it for reading data values but not be distracting
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                  alpha=0.5)

    # Hide these grid behind plot objects
    ax1.set_axisbelow(True)
    ax1.set_title('Comparison of Sample Selection Window Algorithms\n'+et_name+' : '+iv_measure)
    ax1.set_xlabel('Window Type')
    ax1.set_ylabel(iv_measure)

    # Now fill the boxes with desired colors
    boxColors = ['darkkhaki']#,'royalblue']
    numBoxes = numDists#*2
    medians = range(numBoxes)
    for i in range(numBoxes):
      med = bp['medians'][i]
      medianX = []
      medianY = []
      for j in range(2):
          medianX.append(med.get_xdata()[j])
          medianY.append(med.get_ydata()[j])
          plt.plot(medianX, medianY, 'k')
          medians[i] = medianY[0]
      plt.plot([np.average(med.get_xdata())], [np.average(data[i])],
               color='g', marker='*', markeredgecolor='k')

    # Set the axes ranges and axes labels
    ax1.set_xlim(0.5, numBoxes+0.5)
    bottom, top = ax1.get_ylim()
    top = top+0.1
    top = min(top, 10.0)
    bottom = bottom-0.1
    ax1.set_ylim(bottom, top)
    xtickNames = plt.setp(ax1, xticklabels=distNames)
    plt.setp(xtickNames, rotation=45, fontsize=8)

    # Due to the Y-axis scale being different across samples, it can be
    # hard to compare differences in medians across the samples. Add upper
    # X-axis tick labels with the sample medians to aid in comparison
    # (just use two decimal places of precision)
    pos = np.arange(numBoxes)+1
    upperLabels = [str(np.round(s, 3)) for s in medians]
    weights = ['bold', 'semibold']
    for tick,label in zip(range(numBoxes),ax1.get_xticklabels()):
       k = 0#tick % 2
       ax1.text(pos[tick], top-(top*0.05), upperLabels[tick],
            horizontalalignment='center', size='x-small', weight=weights[k],
            color=boxColors[k])

    plt.savefig('%s_%s.png'%(et_name,iv_measure), bbox_inches='tight')
    plt.close()
    
def saveWinTrackerTypeStats(results):
    OUTPUT = file("trial_alg_stats.txt",'w')

    try:
        for track_name, track_results in results.items():
            print
            print '=================='
            print 'EyeTracker:',track_name
            for alg_name, alg_results in track_results.items():
                print '\tAlgorithm:',alg_name
                for stat_name, stat_value in alg_results.items():
                    ra = np.asarray(stat_value,dtype=np.float64)
                    try:
                        rr = '** Error Calculating **'
                        rr = ra.mean()
                    except:
                        pass
                    print '\t\t',stat_name,'(%s):'%(str(ra.shape)),rr
                    # each stat value will be an array of the values for that stat (num_trials*num_sessions)
                    #OUTPUT.write('%s\t%s\t%s\n'%(track_name,alg_name,'\t'.join([str(v) for v in alg_results.values()])))#track_name,'\t',alg_name,'\t','\t '.join([str(v) for v in alg_results.values()])
    except Exception, e:
        OUTPUT.close()
        print "Error Occurred:", e
        import traceback
        traceback.print_exc()
        print 'stat_value:',type(stat_value),stat_value



if __name__ == '__main__':
    
    GLOB_PATH_PATTERN = nabs('{root}/*/*_win_select/*.npy'.format(root=os.path.join(INPUT_FILE_ROOT)))

    paths=[fpath for fpath in glob.glob(GLOB_PATH_PATTERN) if analyseit(fpath)]
    
    from collections import OrderedDict
    results = OrderedDict()


    for fpath in paths:
        try:
#            datafile = np.loadtxt(fpath, dtype=input_file_dtype, comments='#', delimiter='\t',skiprows=1)#, converters=None, skiprows=0, usecols=None, unpack=False, ndmin=0)[source]
            datafile=np.load(fpath)
        except:
            print 'Failed to Load:',fpath
            continue
        
        subject_ids = np.unique(datafile['subject_id'])
        eye_tracker_models = np.unique(datafile['eyetracker_model'])
        session_ids = np.unique(datafile['SESSION_ID'])
        trial_ids = np.unique(datafile['trial_id'])
        assert len(eye_tracker_models) == 1
        assert len(subject_ids) == 1
        et_model_results = results.setdefault(eye_tracker_models[0], OrderedDict())
        for sid in session_ids:
            for tid in trial_ids:
                smask = datafile['SESSION_ID']==sid
                tmask = datafile['trial_id']==tid
                mask = np.bitwise_and(smask,tmask)

                trial_data =  datafile[mask]

                target_x = trial_data['target_angle_x'][0]
                target_y = trial_data['target_angle_y'][0]
                target_start = trial_data['TRIAL_START'][0]
                target_end = trial_data['TRIAL_END'][0]
                for wintype in window_type_cols:
                    et_model_wintype_results = et_model_results.setdefault(wintype, OrderedDict())

                    win_sample_rows = trial_data[trial_data[wintype]==1]
                    #print sid, tid, eye_tracker_models, subject_ids, session_ids, wintype, win_sample_rows.shape
                    #print 'wintype:',wintype
                    if win_sample_rows.shape[0]>0:
                        #print "win_sample_rows['time'][0]:",win_sample_rows['time'][0]
                        #print "target_start:",target_start
                        #print 'target_end:',target_end
                        window_onset = win_sample_rows['time'][0]-target_start
                        if window_onset <0.2:
                            print trial_data['subject_id'][0], window_onset, wintype, tid
#                            STOP
#                            print win_sample_rows['time']
                            
                        sample_count = win_sample_rows.shape[0]
                        #print "window_onset:",window_onset
                        #print "window_offset:",window_offset
                        #print 'sample_count:',sample_count
                        #print '---------------------'


                        if wintype.find('right')>=0:
                            eye_pos_x = win_sample_rows['right_angle_x']
                            eye_pos_y = win_sample_rows['right_angle_y']
                            x_errors = win_sample_rows['right_angle_x'] -  target_x
                            y_errors = win_sample_rows['right_angle_y'] -  target_y
                        elif wintype.find('left')>=0:
                            eye_pos_x = win_sample_rows['left_angle_x']
                            eye_pos_y = win_sample_rows['left_angle_y']
                            x_errors = win_sample_rows['left_angle_x'] -  target_x
                            y_errors = win_sample_rows['left_angle_y'] -  target_y
                        else:
                            print "UNHANDLED WINDOW TYPE:",wintype
                        x_nan_mask = np.isnan(eye_pos_x)
                        y_nan_mask = np.isnan(eye_pos_y)
                        x_nan_count = eye_pos_x[x_nan_mask].shape[0]
                        y_nan_count = eye_pos_y[y_nan_mask].shape[0]
                        if x_nan_count > 0 or y_nan_count > 0:
                            c=et_model_wintype_results.setdefault('nan_trial_count',0)
                            et_model_wintype_results['nan_trial_count']+=1
                            break

                        abs_x_errors = np.abs(x_errors)
                        abs_y_errors = np.abs(y_errors)

                        et_model_wintype_results.setdefault('sample_count',[]).append(sample_count)

                        et_model_wintype_results.setdefault("window_onset",[]).append(window_onset)
                        #et_model_wintype_results.setdefault("max_window_onset",[]).append(window_onset.max())
                        #et_model_wintype_results.setdefault("average_window_onset",[]).append(window_onset.mean())
                        #et_model_wintype_results.setdefault("median_window_onset",[]).append(np.median(window_onset))
                        #et_model_wintype_results.setdefault("stdev_window_onset",[]).append(window_onset.std())

                        #et_model_wintype_results.setdefault("window_offset",[]).append(window_offset)
                        #et_model_wintype_results.setdefault("max_window_offset",[]).append(window_offset.max())
                        #et_model_wintype_results.setdefault("average_window_offset",[]).append(window_offset.mean())
                        #et_model_wintype_results.setdefault("median_window_offset",[]).append(np.median(window_offset))
                        #et_model_wintype_results.setdefault("stdev_window_offset",[]).append(window_offset.std())

                        #et_model_wintype_results.setdefault("min_x",[]).append(eye_pos_x.min())
                        #et_model_wintype_results.setdefault("min_y",[]).append(eye_pos_y.min())
                        #et_model_wintype_results.setdefault("max_x",[]).append(eye_pos_x.max())
                        #et_model_wintype_results.setdefault("max_y",[]).append(eye_pos_y.max())
                        #et_model_wintype_results.setdefault("average_x",[]).append(eye_pos_x.mean())
                        #et_model_wintype_results.setdefault("average_y",[]).append(eye_pos_y.mean())
                        #et_model_wintype_results.setdefault("median_x",[]).append(np.median(eye_pos_x))
                        #et_model_wintype_results.setdefault("median_y",[]).append(np.median(eye_pos_y))
                        #et_model_wintype_results.setdefault("stdev_x",[]).append(eye_pos_x.std())
                        #et_model_wintype_results.setdefault("stdev_y",[]).append(eye_pos_y.std())
                        #et_model_wintype_results.setdefault("rms_x",[]).append(np.sqrt(np.mean(np.power(eye_pos_x, 2.0))) / sample_count)
                        #et_model_wintype_results.setdefault("rms_y",[]).append(np.sqrt(np.mean(np.power(eye_pos_y, 2.0))) / sample_count)

                        et_model_wintype_results.setdefault("min_error_x",[]).append(abs_x_errors.min())
                        et_model_wintype_results.setdefault("min_error_y",[]).append(abs_y_errors.min())
                        et_model_wintype_results.setdefault("max_error_x",[]).append(abs_x_errors.max())
                        et_model_wintype_results.setdefault("max_error_y",[]).append(abs_y_errors.max())
                        et_model_wintype_results.setdefault("average_error_x",[]).append(abs_x_errors.mean())
                        et_model_wintype_results.setdefault("average_error_y",[]).append(abs_y_errors.mean())
                        et_model_wintype_results.setdefault("median_error_x",[]).append(np.median(abs_x_errors))
                        et_model_wintype_results.setdefault("median_error_y",[]).append(np.median(abs_y_errors))
                        et_model_wintype_results.setdefault("stdev_error_x",[]).append(abs_x_errors.std())
                        et_model_wintype_results.setdefault("stdev_error_y",[]).append(abs_y_errors.std())

        del datafile
        datafile = None

    #saveWinTrackerTypeStats(results)

    print "PLOT_IV:",PLOT_IV
    print
    for plot_iv in PLOT_IV:
        data=OrderedDict()
        print "plot_iv:"
        for et_type in results.keys():
            print "\tet_type:",et_type
            for wintype in window_type_cols:
                if COMBINE_XY:
                    if results[et_type][wintype].get("%s_x"%(plot_iv)) is not None:
                        datx = np.asarray(results[et_type][wintype]["%s_x"%(plot_iv)])
                        daty = np.asarray(results[et_type][wintype]["%s_y"%(plot_iv)])
                        data[wintype]=np.hypot(datx, daty)
                    else:
                        data[wintype]=np.asarray(results[et_type][wintype][plot_iv])
                else:
                    print "** NOT COMBINING X & Y IS NOT IMPLEMENTED **"

            if COMBINE_LEFT_RIGHT:
                lrdata = OrderedDict()
                for wintype in window_type_cols:
                    base_win_type = wintype.split('_',1)[-1]
                    r = lrdata.get(base_win_type)
                    if r is not None:
                        tl = r.tolist()
                        tl.extend(data[wintype].tolist())
                        lrdata[base_win_type]= np.asarray(tl)
                    else:
                        lrdata[base_win_type]=data[wintype]
                data = lrdata

            plot(et_type,plot_iv,data)

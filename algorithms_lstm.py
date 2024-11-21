import cv2
import logging
import base64
import time
import numpy as np
import matplotlib.pyplot as plt
from vis.visual import write_on_image, visualise, activity_dict, visualise_tracking
from vis.processor import Processor
from helpers import pop_and_add, last_ip, dist, move_figure, get_hist
from default_params import *
from vis.inv_pendulum import *
import re
import pandas as pd
from scipy.signal import savgol_filter, lfilter
from model.model import LSTMModel
import torch
import math

from statsmodels.tsa.arima.model import ARIMA

from collections import Counter


import cv2
import logging
import pandas as pd
import re

def get_source(args):
    tagged_df = None
    # Set the video path directly to your file
    video_path = '/Users/aditya/Desktop/Projects/gaitAnalysis/HumanFallDetection-master/50waystofall.mp4'
    
    # Use the specified video file instead of checking for args.video
    logging.debug(f'Video source: {video_path}')
    cam = cv2.VideoCapture(video_path)

    # Process the video file as before (if applicable)
    if isinstance(video_path, str):
        vid = [int(s) for s in re.findall(r'\d+', video_path)]
        if len(vid) == 5:
            tagged_df = pd.read_csv("dataset/CompleteDataSet.csv", usecols=[
                "TimeStamps", "Subject", "Activity", "Trial", "Tag"], skipinitialspace=True)
            tagged_df = tagged_df.query(
                f'Subject == {vid[1]} & Activity == {vid[0]} & Trial == {vid[2]}')
    
    # Capture the first frame of the video
    img = cam.read()[1]
    logging.debug('Image shape:', img.shape)
    
    return cam, tagged_df


def resize(img, resize, resolution):
    # Resize the video
    if resize is None:
        height, width = img.shape[:2]
    else:
        width, height = [int(dim) for dim in resize.split('x')]
    width_height = (int(width * resolution // 16) * 16,
                    int(height * resolution // 16) * 16)
    return width, height, width_height


def extract_keypoints_parallel(queue, args, self_counter, other_counter, consecutive_frames, event):
    try:
        cam, tagged_df = get_source(args)
        ret_val, img = cam.read()
    except Exception as e:
        queue.put(None)
        event.set()
        print('Exception occurred:', e)
        print('Most likely that the video/camera doesn\'t exist')
        return

    width, height, width_height = resize(img, args.resize, args.resolution)
    logging.debug(f'Target width and height = {width_height}')
    processor_singleton = Processor(width_height, args)

    output_video = None

    frame = 0
    fps = 0
    t0 = time.time()
    while not event.is_set():
        # print(args.video,self_counter.value,other_counter.value,sep=" ")
        if args.num_cams == 2 and (self_counter.value > other_counter.value):
            continue

        ret_val, img = cam.read()
        frame += 1
        self_counter.value += 1
        if tagged_df is None:
            curr_time = time.time()
        else:
            curr_time = tagged_df.iloc[frame-1]['TimeStamps'][11:]
            curr_time = sum(x * float(t) for x, t in zip([3600, 60, 1], curr_time.split(":")))

        if curr_time is None:
            print('no more images captured')
            print(args.video, curr_time, sep=" ")
            if not event.is_set():
                event.set()
                break

        img = cv2.resize(img, (width, height))
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        keypoint_sets, bb_list, width_height = processor_singleton.single_image(img)
        assert bb_list is None or (type(bb_list) == list)
        if bb_list:
            assert type(bb_list[0]) == tuple
            assert type(bb_list[0][0]) == tuple
        # assume bb_list is a of the form [(x1,y1),(x2,y2)),etc.]

        if args.coco_points:
            keypoint_sets = [keypoints.tolist() for keypoints in keypoint_sets]
        else:
            anns = [get_kp(keypoints.tolist()) for keypoints in keypoint_sets]
            ubboxes = [(np.asarray([width, height])*np.asarray(ann[1])).astype('int32')

                       for ann in anns]
            lbboxes = [(np.asarray([width, height])*np.asarray(ann[2])).astype('int32')
                       for ann in anns]
            bbox_list = [(np.asarray([width, height])*np.asarray(box)).astype('int32') for box in bb_list]
            uhist_list = [get_hist(hsv_img, bbox) for bbox in ubboxes]
            lhist_list = [get_hist(img, bbox) for bbox in lbboxes]
            keypoint_sets = [{"keypoints": keyp[0], "up_hist":uh, "lo_hist":lh, "time":curr_time, "box":box}
                             for keyp, uh, lh, box in zip(anns, uhist_list, lhist_list, bbox_list)]

            cv2.polylines(img, ubboxes, True, (255, 0, 0), 2)
            cv2.polylines(img, lbboxes, True, (0, 255, 0), 2)
            for box in bbox_list:
                cv2.rectangle(img, tuple(box[0]), tuple(box[1]), ((0, 0, 255)), 2)

        dict_vis = {"img": img, "keypoint_sets": keypoint_sets, "width": width, "height": height, "vis_keypoints": args.joints,
                    "vis_skeleton": args.skeleton, "CocoPointsOn": args.coco_points,
                    "tagged_df": {"text": f"Avg FPS: {frame//(time.time()-t0)}, Frame: {frame}", "color": [0, 0, 0]}}
        queue.put(dict_vis)

    queue.put(None)
    return


###################################################### Post human estimation ###########################################################


def show_tracked_img(img_dict, ip_set, num_matched, output_video, args):
    img = img_dict["img"]
    tagged_df = img_dict["tagged_df"]
    keypoints_frame = [person[-1] for person in ip_set]
    img = visualise_tracking(img=img, keypoint_sets=keypoints_frame, width=img_dict["width"], height=img_dict["height"],
                             num_matched=num_matched, vis_keypoints=img_dict["vis_keypoints"], vis_skeleton=img_dict["vis_skeleton"],
                             CocoPointsOn=False)

    img = write_on_image(img=img, text=tagged_df["text"],
                         color=tagged_df["color"])

    if output_video is None:
        if args.save_output:
            if isinstance(args.video, int):
                vidname = [str(args.video)+'.avi']
            else:
                vidname = args.video.split('/')
            filename = '/'.join(vidname[:-1])
            if filename:
                filename += '/'
            filename += 'out' + vidname[-1][:-3] + 'avi'
            output_video = cv2.VideoWriter(filename=filename, fourcc=cv2.VideoWriter_fourcc(*'MP42'),
                                           fps=args.fps, frameSize=img.shape[:2][::-1])
            logging.debug(
                f'Saving the output video at {filename} with {args.fps} frames per seconds')
        else:
            output_video = None
            logging.debug(f'Not saving the output video')
    else:
        output_video.write(img)
    return img, output_video


def remove_wrongly_matched(matched_1, matched_2):

    unmatched_idxs = []
    i = 0

    for ip1, ip2 in zip(matched_1, matched_2):
        # each of these is a set of the last t framses of each matched person
        correlation = cv2.compareHist(last_valid_hist(ip1)["up_hist"], last_valid_hist(ip2)["up_hist"], cv2.HISTCMP_CORREL)
        if correlation < 0.5*HIST_THRESH:
            unmatched_idxs.append(i)
        i += 1

    return unmatched_idxs


def match_unmatched(unmatched_1, unmatched_2, arima_sets_1, arima_sets_2, num_matched):
    # Placeholder for ARIMA sets as we are no longer using LSTM
    new_matched_1 = []
    new_matched_2 = []
    new_arima1 = []
    new_arima2 = []
    final_pairs = [[], []]

    if not unmatched_1 or not unmatched_2:
        return final_pairs, new_matched_1, new_matched_2, new_arima1, new_arima2

    new_matched = 0
    correlation_matrix = - np.ones((len(unmatched_1), len(unmatched_2)))
    dist_matrix = np.zeros((len(unmatched_1), len(unmatched_2)))
    
    for i in range(len(unmatched_1)):
        for j in range(len(unmatched_2)):
            correlation_matrix[i][j] = cv2.compareHist(last_valid_hist(unmatched_1[i])["up_hist"],
                                                       last_valid_hist(unmatched_2[j])["up_hist"], cv2.HISTCMP_CORREL)
            dist_matrix[i][j] = np.sum(np.absolute(last_valid_hist(unmatched_1[i])["up_hist"] - last_valid_hist(unmatched_2[j])["up_hist"]))

    freelist_1 = [i for i in range(len(unmatched_1))]
    pair_21 = [-1]*len(unmatched_2)
    unmatched_1_preferences = np.argsort(-correlation_matrix, axis=1)

    unmatched_indexes1 = [0]*len(unmatched_1)
    finish_array = [False]*len(unmatched_1)
    while freelist_1:
        um1_idx = freelist_1[-1]
        if finish_array[um1_idx] == True:
            freelist_1.pop()
            continue
        next_unasked_2 = unmatched_1_preferences[um1_idx][unmatched_indexes1[um1_idx]]
        if pair_21[next_unasked_2] == -1:
            pair_21[next_unasked_2] = um1_idx
            freelist_1.pop()
        else:
            curr_paired_2 = pair_21[next_unasked_2]
            if correlation_matrix[curr_paired_2][next_unasked_2] < correlation_matrix[um1_idx][next_unasked_2]:
                pair_21[next_unasked_2] = um1_idx
                freelist_1.pop()
                if not finish_array[curr_paired_2]:
                    freelist_1.append(curr_paired_2)

        unmatched_indexes1[um1_idx] += 1
        if unmatched_indexes1[um1_idx] == len(unmatched_2):
            finish_array[um1_idx] = True

    for j, i in enumerate(pair_21):
        if correlation_matrix[i][j] > HIST_THRESH:
            final_pairs[0].append(i + num_matched)
            final_pairs[1].append(j + num_matched)
            new_matched_1.append(unmatched_1[i])
            new_matched_2.append(unmatched_2[j])
            new_arima1.append(arima_sets_1[i])
            new_arima2.append(arima_sets_2[j])

    return final_pairs, new_matched_1, new_matched_2, new_arima1, new_arima2




from statsmodels.tsa.arima.model import ARIMA
import numpy as np

# ARIMA Configuration
p, d, q = 2, 1, 2  # You can adjust these parameters based on your needs

def get_all_features(ip_set, arima_sets, model):
    valid_idxs = []
    invalid_idxs = []
    predictions = [15] * len(ip_set)  # 15 is the tag for None

    for i, ips in enumerate(ip_set):
        # ip set for a particular person
        last1 = None
        last2 = None
        for j in range(-2, -1 * DEFAULT_CONSEC_FRAMES - 1, -1):
            if ips[j] is not None:
                if last1 is None:
                    last1 = j
                elif last2 is None:
                    last2 = j
        if ips[-1] is None:
            invalid_idxs.append(i)
        else:
            ips[-1]["features"] = {}
            # Get features (height_bbox, ratio_bbox, etc.)
            ips[-1]["features"]["height_bbox"] = get_height_bbox(ips[-1])
            ips[-1]["features"]["ratio_bbox"] = FEATURE_SCALAR["ratio_bbox"] * get_ratio_bbox(ips[-1])

            body_vector = ips[-1]["keypoints"]["N"] - ips[-1]["keypoints"]["B"]
            ips[-1]["features"]["angle_vertical"] = FEATURE_SCALAR["angle_vertical"] * get_angle_vertical(body_vector)
            ips[-1]["features"]["log_angle"] = FEATURE_SCALAR["log_angle"] * np.log(1 + np.abs(ips[-1]["features"]["angle_vertical"]))

            if last1 is None:
                invalid_idxs.append(i)
            else:
                ips[-1]["features"]["re"] = FEATURE_SCALAR["re"] * get_rot_energy(ips[last1], ips[-1])
                ips[-1]["features"]["ratio_derivative"] = FEATURE_SCALAR["ratio_derivative"] * get_ratio_derivative(ips[last1], ips[-1])
                if last2 is None:
                    invalid_idxs.append(i)
                else:
                    ips[-1]["features"]["gf"] = get_gf(ips[last2], ips[last1], ips[-1])
                    valid_idxs.append(i)

        # Prepare time series data
        xdata = []
        if ips[-1] is None:
            if last1 is None:
                xdata = [0] * len(FEATURE_LIST)
            else:
                for feat in FEATURE_LIST[:FRAME_FEATURES]:
                    xdata.append(ips[last1]["features"][feat])
                xdata += [0] * (len(FEATURE_LIST) - FRAME_FEATURES)
        else:
            for feat in FEATURE_LIST:
                if feat in ips[-1]["features"]:
                    xdata.append(ips[-1]["features"][feat])
                else:
                    xdata.append(0)

        # Use ARIMA model to forecast
        try:
            series = np.array(xdata)  # Convert to numpy array for ARIMA input
            model_arima = ARIMA(series, order=(p, d, q))
            arima_fit = model_arima.fit()
            prediction = arima_fit.forecast(steps=1)[0]
            # Convert the ARIMA prediction to the corresponding class if necessary
            # Here, we are just using the first element of the prediction
            predictions[i] = int(prediction)
        except Exception as e:
            print(f"ARIMA model fitting error for person {i}: {e}")
            predictions[i] = 15  # Default "None" if prediction fails

    return valid_idxs, predictions[0] if len(predictions) > 0 else 15


def get_frame_features(ip_set, new_frame, re_matrix, gf_matrix, num_matched, max_length_mat=DEFAULT_CONSEC_FRAMES):
    # Now, ARIMA-based feature extraction and matching
    match_ip(ip_set, new_frame, re_matrix, gf_matrix, max_length_mat)
    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            if ip_set[i][-2] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(ip_set[i][-2], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-3] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(ip_set[i][-3], ip_set[i][-1]), max_length_mat)
            elif ip_set[i][-4] is not None:
                pop_and_add(re_matrix[i], get_rot_energy(ip_set[i][-4], ip_set[i][-1]), max_length_mat)
            else:
                pop_and_add(re_matrix[i], 0, max_length_mat)
        else:
            pop_and_add(re_matrix[i], 0, max_length_mat)

    for i in range(len(ip_set)):
        if ip_set[i][-1] is not None:
            last1 = None
            last2 = None
            for j in [-2, -3, -4, -5]:
                if ip_set[i][j] is not None:
                    if last1 is None:
                        last1 = j
                    elif last2 is None:
                        last2 = j

            if last2 is None:
                pop_and_add(gf_matrix[i], 0, max_length_mat)
                continue

            pop_and_add(gf_matrix[i], get_gf(ip_set[i][last2], ip_set[i][last1], ip_set[i][-1]), max_length_mat)

        else:
            pop_and_add(gf_matrix[i], 0, max_length_mat)

    return


def process_arima_predictions(arima_predictions):
    """
    Process the ARIMA predictions and return the final mapped activity prediction.

    :param arima_predictions: A list of ARIMA model predictions (e.g., activity type IDs).
    :return: Mapped activity prediction.
    """
    # Here, we assume the activity_dict is a dictionary mapping prediction IDs to activity names
    # Modify the dictionary as needed to fit your use case
    activity_dict = {
        0: 'Walking',
        1: 'Running',
        2: 'Sitting',
        3: 'Standing',
        4: 'Falling',
        # add more as necessary
    }

    # Check if the arima_predictions list is empty
    if not arima_predictions:
        print("Warning: No ARIMA predictions available!")
        return 'Unknown'  # Return 'Unknown' if no predictions are available

    # Example: Taking the most frequent prediction as the final prediction
    # You can customize this logic based on your requirements
    prediction = Counter(arima_predictions).most_common(1)[0][0]

    # Return the mapped activity based on the prediction
    return activity_dict.get(prediction, 'Unknown')  # Default to 'Unknown' if no match

def alg2_sequential(queues, argss, consecutive_frames, event):
    # ARIMA model configuration
    p, d, q = 2, 1, 2  # Example values; these should be determined by analyzing the data
    output_videos = [None for _ in range(argss[0].num_cams)]
    t0 = time.time()
    feature_plotters = [[[], [], [], [], []] for _ in range(argss[0].num_cams)]
    ip_sets = [[] for _ in range(argss[0].num_cams)]
    arima_sets = [[] for _ in range(argss[0].num_cams)]  # Renamed to arima_sets
    max_length_mat = 300
    num_matched = 0
    if not argss[0].plot_graph:
        max_length_mat = consecutive_frames
    else:
        f, ax = plt.subplots()
        move_figure(f, 800, 100)
    
    window_names = [args.video if isinstance(args.video, str) else 'Cam '+str(args.video) for args in argss]
    [cv2.namedWindow(window_name) for window_name in window_names]

    while True:
        if not any(q.empty() for q in queues):
            dict_frames = [q.get() for q in queues]

            if any([(dict_frame is None) for dict_frame in dict_frames]):
                if not event.is_set():
                    event.set()
                break

            if cv2.waitKey(1) == 27 or any(cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1 for window_name in window_names):
                if not event.is_set():
                    event.set()

            kp_frames = [dict_frame["keypoint_sets"] for dict_frame in dict_frames]
            if argss[0].num_cams == 1:
                num_matched, new_num, indxs_unmatched = match_ip(ip_sets[0], kp_frames[0], arima_sets[0], num_matched, max_length_mat)

                # Extract time series for ARIMA forecasting
                time_series_data = []
                for data in ip_sets[0]:
                    if isinstance(data, dict) and 'time_series_feature' in data:
                        time_series_data.append(data['time_series_feature'])
                    else:
                        # Handle the case where data is not a dictionary or missing 'time_series_feature'
                        print("Skipping data (not a dictionary or missing 'time_series_feature'):", data)

                # Fit ARIMA model and generate prediction
                arima_predictions = []
                for series in time_series_data:
                    arima_model = ARIMA(series, order=(p, d, q))
                    fitted_model = arima_model.fit()
                    arima_predictions.append(fitted_model.forecast(steps=1)[0])

                # Ensure that arima_predictions is not empty
                if not arima_predictions:
                    print("No ARIMA predictions generated!")
                    continue  # Skip this frame if no predictions were generated

                # Apply prediction mapping if needed
                prediction = process_arima_predictions(arima_predictions)  # Map the predictions if using activity_dict
                dict_frames[0]["tagged_df"]["text"] += f" Pred: {prediction}"
                img, output_videos[0] = show_tracked_img(dict_frames[0], ip_sets[0], num_matched, output_videos[0], argss[0])

                # Display the frame
                if img is not None:
                    print("Displaying frame...")
                    cv2.imshow(window_names[0], img)
                else:
                    print("Failed to capture frame")

            elif argss[0].num_cams == 2:
                # Similar changes for multi-camera matching with ARIMA
                num_matched, new_num, indxs_unmatched1 = match_ip(ip_sets[0], kp_frames[0], arima_sets[0], num_matched, max_length_mat)
                assert(new_num == len(ip_sets[0]))
                
                # Handle other camera with ARIMA predictions in a similar way
                # Matching and ARIMA prediction handling here...
            
            # Handle the prediction and feature plotting as needed.
    
    # Ensure model is deleted only if it's assigned
    try:
        del model
    except NameError:
        print("Model not defined, cannot delete.")
    
    cv2.destroyAllWindows()
    return

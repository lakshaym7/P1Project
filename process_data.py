if __name__ == "__main__":
    import fall_detector
    import sys
    import csv
    import os
    import joblib
    import time
    from algorithms import initialize_arima_models, get_all_features  # Import ARIMA initialization and prediction functions

    sub_start = 15
    sub_end = 18
    orig_sys_argv = sys.argv
    
    # Loop over activity, subject, trial, and camera IDs
    for act_id in range(1, 12):
        for sub_id in range(sub_start, sub_end):
            dl = []
            t0 = time.time()
            
            # Initialize ARIMA models (assumed that ip_set is available here as the data structure holding input points)
            ip_set = []  # You need to initialize the ip_set with data for feature extraction.
            arima_set = initialize_arima_models(ip_set)  # Initialize ARIMA models
            
            # Iterate over trials and camera IDs
            for trial_id in range(1, 4):
                for cam_id in range(1, 3):
                    video_path = f'dataset/Activity{act_id}/Subject{sub_id}/Trial{trial_id}Cam{cam_id}.mp4'
                    
                    # Check if the video file exists before processing
                    if os.path.exists(video_path):
                        # Argument setup for fall detection
                        args = ['--coco_points', f'--video={video_path}']
                        sys.argv = [orig_sys_argv[0]] + args
                        
                        # Initialize FallDetector (assumed to extract features and keypoints)
                        f = fall_detector.FallDetector()
                        q1 = f.begin_mixed()
                        ip_set.append(q1)  # Assuming `q1` contains keypoints or feature data.

            # After processing the frames, get all features and make predictions using ARIMA
            valid_idxs, predictions = get_all_features(ip_set, arima_set)
            
            # Save the processed keypoints and predictions to a file using joblib
            joblib.dump({
                'keypoints': ip_set,
                'predictions': predictions
            }, f'dataset/Activity{act_id}/Subject{sub_id}/coco.kps', True)
            
            print(f"Processing time for Subject {sub_id}, Activity {act_id}: {time.time() - t0} seconds")

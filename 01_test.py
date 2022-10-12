########################################################################
# import default libraries
########################################################################
import os
import csv
import sys
import gc
########################################################################


########################################################################
# import additional libraries
########################################################################
import numpy as np
import scipy.stats
# from import
from tqdm import tqdm
from sklearn import metrics
from numpy.lib.stride_tricks import as_strided
try:
    from sklearn.externals import joblib
except:
    import joblib
# original lib
import common as com
import keras_model
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import librosa
########################################################################


########################################################################
# load parameter.yaml
########################################################################
param = com.yaml_load()

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
#######################################################################


########################################################################
# output csv file
########################################################################
def save_csv(save_file_path,
             save_data):
    with open(save_file_path, "w", newline="") as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(save_data)

def pool1d(A, kernel_size, stride, padding=0, pool_mode='max'):
    '''
    1D Pooling

    Parameters:
        A: input 1D array
        kernel_size: int, the size of the window over which we take pool
        stride: int, the stride of the window
        padding: int, implicit zero paddings on both sides of the input
        pool_mode: string, 'max' or 'avg'
    '''
    # Padding
    A = np.pad(A, padding, mode='constant')
    # Window view of A
    
    output_shape = ((A.shape[0] - kernel_size) // stride + 1,
                    (A.shape[1] - kernel_size) // stride + 1)

    shape_w = (A.shape[0],output_shape[1], kernel_size)
    strides_w = (A.strides[0],stride*A.strides[1], A.strides[1])
    
    A_w = as_strided(A, shape_w, strides_w)
    
    # Return the result of pooling
    if pool_mode == 'max':
        return A_w.max(axis=2)
    elif pool_mode == 'avg':
        return A_w.mean(axis=2)
        # b = np.array(param["weight"])
        # weight = np.ones_like(A_w)
        # for j in range(309):
        #     for n in range(5):
        #         weight[j][64*n:64*n+64][:] = b
        # return np.average(A_w, axis=2, weights=weight)
    
def masking(data):
      for idx in range(len(data)):
        vectors_masked = com.spec_augment(data[idx,:,:])
        # data[idx,:,:] = librosa.power_to_db(data[idx,:,:])
        # vectors_masked[idx,:,:] = librosa.power_to_db(vectors_masked[idx,:,:])
        if idx == 0:
            data_masked = np.zeros((len(data), vectors_masked.shape[0], vectors_masked.shape[1]), float)
        data_masked[idx, :, :] = vectors_masked
      return np.swapaxes(data_masked, 1, 2), np.swapaxes(data, 2, 1)

########################################################################


########################################################################
# main 01_test.py
########################################################################
if __name__ == "__main__":
    # check mode
    # "development": mode == True
    # "evaluation": mode == False
    mode = com.command_line_chk()
    if mode is None:
        sys.exit(-1)

    # make output result directory
    os.makedirs(param["result_directory"], exist_ok=True)

    # load base directory
    dirs = com.select_dirs(param=param, mode=mode)

    # initialize lines in csv for AUC and pAUC
    csv_lines = []

    if mode:
        performance_over_all = []

    # loop of the base directory
    for idx, target_dir in enumerate(dirs):
        print("\n===========================")
        print("[{idx}/{total}] {target_dir}".format(target_dir=target_dir, idx=idx+1, total=len(dirs)))
        machine_type = os.path.split(target_dir)[1]

        print("============== MODEL LOAD ==============")
        # load model file
        model_file = "{model}/model_{machine_type}.hdf5".format(model=param["model_directory"],
                                                                machine_type=machine_type)
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            sys.exit(-1)
        model = keras_model.load_model(model_file)
        model.summary()

        if mode:
            # results for each machine type
            csv_lines.append([machine_type])
            csv_lines.append(["section", "domain", "AUC", "pAUC",])
            performance = []

        dir_names = ["source_test"]
        
        for dir_name in dir_names:

            #list machine id
            section_names = com.get_section_names(target_dir, dir_name=dir_name)

            for section_name in section_names:
                # load test file
                files, y_true = com.file_list_generator(target_dir=target_dir,
                                                        section_name=section_name,
                                                        dir_name=dir_name,
                                                        mode=mode)

                # setup anomaly score file path
                anomaly_score_csv = "{result}/anomaly_score_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                 machine_type=machine_type,
                                                                                                                 section_name=section_name,
                                                                                                                 dir_name=dir_name)
                anomaly_score_list = []
                
                # setup anomaly score time tagging file path
                
                anomaly_score_time_csv = "{result}/anomaly_score_time_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                    machine_type=machine_type,
                                                                                                                    section_name=section_name,
                                                                                                                    dir_name=dir_name)
                anomaly_score_time_list = []
                x = np.arange(0, 512, 1)
                x = x*431/22050
                x[511] = 10
                d = []
                d.extend(['name','type'])
                d.extend(x)
                anomaly_score_time_list.append(d)

                # setup decision result file path
                decision_result_csv = "{result}/decision_result_{machine_type}_{section_name}_{dir_name}.csv".format(result=param["result_directory"],
                                                                                                                     machine_type=machine_type,
                                                                                                                     section_name=section_name,
                                                                                                                     dir_name=dir_name)
                decision_result_list = []

                print("\n============== BEGIN TEST FOR A SECTION ==============")
                y_pred_new = [0. for k in files]
                y_pred_freq_new = [0. for k in files]
                y_pred_time_new = [0. for k in files]
                
                
                # New
                y_pred = [0. for k in range(512)]
                y_pred_freq = [0. for k in range(512)]
                y_pred_time = [0. for k in range(512)]

                for file_idx, file_path in tqdm(enumerate(files), total=len(files)):
                    try:
                        data = com.file_to_vectors(file_path,
                                                        n_mels=param["feature"]["n_mels"],
                                                        n_frames=param["feature"]["n_frames"],
                                                        n_fft=param["feature"]["n_fft"],
                                                        hop_length=param["feature"]["hop_length"],
                                                        power=param["feature"]["power"])
                        vectors = librosa.power_to_db(data)
                        vectors_masked = com.spec_augment_freq(vectors)
                        data = np.zeros((1, vectors.shape[1], vectors.shape[0]),float)
                        data_masked = np.zeros((1, vectors_masked.shape[1], vectors_masked.shape[0]), float)
                        data[0, :, :] = vectors.T
                        data_masked[0, :, :] = vectors_masked.T
                        data = data[:, :, :, np.newaxis]
                        data_masked = data_masked[:, :, :, np.newaxis]
                    except:
                        com.logger.error("File broken!!: {}".format(file_path))
                    # # OLD BETTER
                    
                    # reconstruction = model.predict(data)
                    # # Freq avg-pool
                    # data_freq = pool1d(data, kernel_size=2, stride=2, padding=0, pool_mode='avg')
                    # predict_freq = pool1d(reconstruction, kernel_size=2, stride=2, padding=0, pool_mode='avg')              
                    # y_pred_freq_new[file_idx] = np.mean(np.square(data_freq - predict_freq))
                    # # Time max-pool
                    # data_time = pool1d(data.T, kernel_size=2, stride=2, padding=0, pool_mode='max')
                    # predict_time = pool1d(reconstruction.T, kernel_size=2, stride=2, padding=0, pool_mode='max')            
                    # y_pred_time_new[file_idx] = np.mean(np.square(data_time - predict_time))
                    # # Anomaly score csv
                    # anomaly_score_list.append([os.path.basename(file_path), y_pred_freq_new[file_idx], 'freq avg-pooling'])
                    # anomaly_score_list.append([os.path.basename(file_path), y_pred_time_new[file_idx], 'time max-pooling'])                    
                    
                    # y_pred_new[file_idx] = np.mean(np.square(data - reconstruction))
                    
                    # # store anomaly scores
                    # anomaly_score_list.append([os.path.basename(file_path), y_pred_new[file_idx]])
                    
                    #New
                    reconstruction = model.predict(data_masked)
                    # hop = 15
                    # win_length = 62
                    # b=[]
                    # Freq avg-pool
                    data = np.squeeze(data, axis=3)
                    reconstruction = np.squeeze(reconstruction , axis=3)
                    # print(data[:,:].shape, reconstruction[:,:].shape)
                    data_freq = pool1d(data[0,:,:], kernel_size=2, stride=2, padding=0, pool_mode='avg')
                    predict_freq = pool1d(reconstruction[0,:,:], kernel_size=2, stride=2, padding=0, pool_mode='avg')
                    for n in range(len(data_freq[0])):              
                        y_pred_freq[n] = np.mean(np.square(data_freq[n,:] - predict_freq[n,:]))
                        # y_pred_freq[n] = np.mean(np.square(data_freq[43:79,n] - predict_freq[43:79,n]))
                    # Time max-pool
                    data_time = pool1d(data[0,:,:].T, kernel_size=2, stride=2, padding=0, pool_mode='max')
                    predict_time = pool1d(reconstruction[0,:,:].T, kernel_size=2, stride=2, padding=0, pool_mode='max')    
                    for n in range(len(data_time[1])):
                        y_pred_time[n] = np.mean(np.square(data_time[24:44, n] - predict_time[24:44, n]))
                        # y_pred_time[n] = np.mean(np.square(data_time[n,43:79] - predict_time[n,43:79]))
                    # Anomaly score csv
                    y_pred_freq_new[file_idx] = np.mean(y_pred_freq)
                    y_pred_time_new[file_idx] = np.mean(y_pred_time)
                    anomaly_score_list.append([os.path.basename(file_path), y_pred_freq_new[file_idx] , 'freq avg-pooling'])
                    anomaly_score_list.append([os.path.basename(file_path), y_pred_time_new[file_idx], 'time max-pooling'])
                    d = []                
                    d.extend([os.path.basename(file_path),'time max-pooling'])
                    d.extend(y_pred_time)
                    anomaly_score_time_list.append(d)
                    d = []                
                    d.extend([os.path.basename(file_path),'freq avg-pooling'])
                    d.extend(y_pred_freq)
                    anomaly_score_time_list.append(d)
                    for n in range(len(data[0])):
                        y_pred[n] = np.mean(np.square(data[0,n,24:44] - reconstruction[0,n,24:44]))
                        # y_pred[n] = np.mean(np.square(data[0,n,43:79] - reconstruction[0,n,43:79]))
                    y_pred_new[file_idx] = np.mean(y_pred)
                    # store anomaly scores
                    anomaly_score_list.append([os.path.basename(file_path), y_pred_new[file_idx]]) 
                    d = []                
                    d.extend([os.path.basename(file_path),'No pooling'])
                    d.extend(y_pred)
                    anomaly_score_time_list.append(d)



                # # output anomaly time scores
                save_csv(save_file_path=anomaly_score_time_csv, save_data=anomaly_score_time_list)
                com.logger.info("anomaly score time result ->  {}".format(anomaly_score_time_csv))
                
                # output anomaly scores
                save_csv(save_file_path=anomaly_score_csv, save_data=anomaly_score_list)
                com.logger.info("anomaly score result ->  {}".format(anomaly_score_csv))

                # output decision results
                save_csv(save_file_path=decision_result_csv, save_data=decision_result_list)
                com.logger.info("decision result ->  {}".format(decision_result_csv))

                if mode:
                    # append AUC and pAUC to lists
                    auc = metrics.roc_auc_score(y_true, y_pred_new)
                    p_auc = metrics.roc_auc_score(y_true, y_pred_new, max_fpr=param["max_fpr"])
                    
                    auc_time = metrics.roc_auc_score(y_true, y_pred_time_new)
                    p_auc_time = metrics.roc_auc_score(y_true, y_pred_time_new, max_fpr=param["max_fpr"])
                    
                    auc_freq = metrics.roc_auc_score(y_true, y_pred_freq_new)
                    p_auc_freq = metrics.roc_auc_score(y_true, y_pred_freq_new, max_fpr=param["max_fpr"])
                    
                    csv_lines.append([section_name.split("_", 1)[1], dir_name.split("_", 1)[0], auc, p_auc])
                    csv_lines.append([section_name.split("_", 1)[1], dir_name.split("_", 1)[0], auc_freq, p_auc_freq, "freq avg-pooling"])
                    csv_lines.append([section_name.split("_", 1)[1], dir_name.split("_", 1)[0], auc_time, p_auc_time,  "time max-pooling"])
                    
                    performance.append([auc, p_auc])
                    performance.append([auc_freq, p_auc_freq])
                    performance.append([auc_time, p_auc_time])
                    
                    performance_over_all.append([auc, p_auc])
                    com.logger.info("AUC : {}".format(auc))
                    com.logger.info("pAUC : {}".format(p_auc))


                            
                    plt.ioff()
                    # Visualize error distribution
                    fig = plt.figure(figsize=(12,8))
                    data = np.column_stack((range(len(y_pred_new)), y_pred_new))
                    data2 = np.column_stack((range(len(y_pred_freq_new)), y_pred_freq_new))
                    data3 = np.column_stack((range(len(y_pred_time_new)), y_pred_time_new))
                    bin_width = 0.75
                    bins = np.arange(min(y_pred_new), max(y_pred_new) + bin_width, bin_width)
                    bins2 = np.arange(min(y_pred_freq_new), max(y_pred_freq_new) + bin_width, bin_width)
                    bins3 = np.arange(min(y_pred_time_new), max(y_pred_time_new) + bin_width, bin_width)
                    plt.hist(data[y_true == 0][:, 1], bins=bins, color=colors[1], alpha=0.6, label='Normal Signals', edgecolor='#FFFFFF')
                    plt.hist(data[y_true == 1][:, 1], bins=bins, color=colors[5], alpha=0.6, label='Abnormal Signals', edgecolor='#FFFFFF')
                    plt.xlabel("Testing Reconstruction Error")
                    plt.ylabel("# Samples")
                    plt.title('Reconstruction Error Distribution on the Test Set')
                    plt.legend()
                    plt.savefig(f'picture/1_{machine_type}.png')
                    plt.close(fig)
                    
                    fig2 = plt.figure(figsize=(12,8))
                    plt.hist(data2[y_true == 0][:, 1], bins=bins2, color=colors[1], alpha=0.6, label='Normal Signals', edgecolor='#FFFFFF')
                    plt.hist(data2[y_true == 1][:, 1], bins=bins2, color=colors[5], alpha=0.6, label='Abnormal Signals', edgecolor='#FFFFFF')
                    plt.xlabel("Testing Reconstruction Error")
                    plt.ylabel("# Samples")
                    plt.title('Reconstruction Error Distribution on the Test Set (Freq Avg-pooling)')
                    plt.legend()
                    plt.savefig(f'picture/2_{machine_type}.png')
                    plt.close(fig2)
                    
                    fig3 = plt.figure(figsize=(12,8))
                    plt.hist(data3[y_true == 0][:, 1], bins=bins3, color=colors[1], alpha=0.6, label='Normal Signals', edgecolor='#FFFFFF')
                    plt.hist(data3[y_true == 1][:, 1], bins=bins3, color=colors[5], alpha=0.6, label='Abnormal Signals', edgecolor='#FFFFFF')
                    plt.xlabel("Testing Reconstruction Error")
                    plt.ylabel("# Samples")
                    plt.title('Reconstruction Error Distribution on the Test Set (Time Max-pooling)')
                    plt.legend()
                    plt.savefig(f'picture/3_{machine_type}.png')
                    plt.close(fig3)


                    
                    
                    #Threshold visualizer
                    threshold_min = 10.0
                    threshold_max = 60.0
                    threshold_step = 0.50

                    normal_x, normal_y = data[y_true==0][:,0], data[y_true==0][:,1]
                    abnormal_x, abnormal_y = data[y_true==1][:,0], data[y_true==1][:,1]
                    x = np.concatenate((normal_x, abnormal_x))

                    fig, ax = plt.subplots(figsize=(12,8))
                    plt.scatter(normal_x, normal_y, s=15, color='tab:green', alpha=0.3, label='Normal Signals')
                    plt.scatter(abnormal_x, abnormal_y, s=15, color='tab:red', alpha=0.3, label='Abnormal Signals')
                    plt.fill_between(x, threshold_min, threshold_max, alpha=0.1, color='tab:orange', label='Threshold Range')
                    plt.hlines([threshold_min, threshold_max], x.min(), x.max(), linewidth=0.5, alpha=0.8, color='tab:orange')
                    plt.legend(loc='upper left')
                    plt.title('Threshold Range Exploration', fontsize=16)
                    plt.xlabel('Samples')
                    plt.ylabel('Reconstruction Error')
                    plt.savefig(f'D:/Nextcloud/dcase/dcase2021_task2_baseline_ae/r_{machine_type}.png')
                    plt.close(fig)
                    
                print("\n============ END OF TEST FOR A SECTION ============")

        if mode:
            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            csv_lines.append(["arithmetic mean", ""] + [amean_performance])
            hmean_performance = scipy.stats.hmean(np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            csv_lines.append(["harmonic mean", ""] + [hmean_performance])
            csv_lines.append([])

        del data
        del model
        gc.collect()

    if mode:
        # output results
        result_path = "{result}/{file_name}".format(result=param["result_directory"], file_name=param["result_file"])
        com.logger.info("results -> {}".format(result_path))
        save_csv(save_file_path=result_path, save_data=csv_lines)

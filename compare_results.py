import numpy as np
from matplotlib import pyplot as plt

for path in ['fno_all_next_step',
             'pretrain_fno_GCL_all_next_step',
             'pretrain_fno_wnxent_all_next_step',
             'pretrain_fno_physics_informed_all_next_step']:
    fig, ax = plt.subplots()
    test_l2s = []
    done = 0
    for i in range(5):
        try:
            ax.plot(np.load("./{}/train_l2s_{}.npy".format(path, i)))
            ax.plot(np.load("./{}/val_l2s_{}.npy".format(path, i)))
            train_vals = np.load("./{}/train_l2s_{}.npy".format(path, i))
            val_vals = np.load("./{}/val_l2s_{}.npy".format(path, i))
            test_vals = np.load("./{}/test_vals_{}.npy".format(path, i))
            test_l2s.append(test_vals)
            #print(val_vals.shape)
            #print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
            #raise
            done += 1
        except FileNotFoundError:
            #print("WORKING ON: {}".format(i))
            pass
    
    try:
        model = 'FNO'
        model += ' GCL' if('GCL' in path) else ' WNXENT' if('wnxent' in path) else ''
        if('GCL' in path):
            print("GCL TEST MSE:\t\t{0:.6f} \t {1:.6f} \t {2}/{3} Complete".format(
                   np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1], done, i+1))
            np.save("./gcl_test_l2s.npy", test_l2s)
        elif('wnxent' in path):
            print("WNXENT TEST MSE:\t{0:.6f} \t {1:.6f} \t {2}/{3} Complete".format(
                   np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1], done, i+1))
            np.save("./wnxent_test_l2s.npy", test_l2s)
        elif('physics_informed' in path):
            print("PI WNXENT TEST MSE:\t{0:.6f} \t {1:.6f} \t {2}/{3} Complete".format(
                   np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1], done, i+1))
            np.save("./piwnxent_test_l2s.npy", test_l2s)
        else:
            print("TEST MSE:\t\t{0:.6f} \t {1:.6f} \t {2}/{3} Complete".format(
                   np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1], done, i+1))
            np.save("./test_l2s.npy", test_l2s)
    except IndexError:
        print("No completed runs.")
    plt.show()

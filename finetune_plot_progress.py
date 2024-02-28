import numpy as np
from matplotlib import pyplot as plt

fig, ax = plt.subplots()
test_l2s = []
print("Combined:")
for i in range(5):
    try:
        ax.plot(np.load("./train_l2s_{}.npy".format(i)))
        ax.plot(np.load("./val_l2s_{}.npy".format(i)))
        train_vals = np.load("./train_l2s_{}.npy".format(i))
        val_vals = np.load("./val_l2s_{}.npy".format(i))
        test_vals = np.load("./test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass
try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

test_l2s = []
print("\nHeat:")
for i in range(5):
    try:
        ax.plot(np.load("./Heat_train_l2s_{}.npy".format(i)))
        ax.plot(np.load("./Heat_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./Heat_train_l2s_{}.npy".format(i))
        val_vals = np.load("./Heat_val_l2s_{}.npy".format(i))
        test_vals = np.load("./Heat_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass
try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

test_l2s = []
print("\nBurgers:")
for i in range(5):
    try:
        ax.plot(np.load("./Burgers_train_l2s_{}.npy".format(i)))
        ax.plot(np.load("./Burgers_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./Burgers_train_l2s_{}.npy".format(i))
        val_vals = np.load("./Burgers_val_l2s_{}.npy".format(i))
        test_vals = np.load("./Burgers_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass
try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")

test_l2s = []
print("\nAdvection:")
for i in range(5):
    try:
        ax.plot(np.load("./Advection_train_l2s_{}.npy".format(i)))
        ax.plot(np.load("./Advection_val_l2s_{}.npy".format(i)))
        train_vals = np.load("./Advection_train_l2s_{}.npy".format(i))
        val_vals = np.load("./Advection_val_l2s_{}.npy".format(i))
        test_vals = np.load("./Advection_test_vals_{}.npy".format(i))
        test_l2s.append(test_vals)
        #print(val_vals.shape)
        print("{0:.6f}\t{1:.6f}\t{2:.6f}".format(np.min(train_vals), np.min(val_vals), test_vals[1]))
        #raise
    except FileNotFoundError:
        print("WORKING ON: {}".format(i))
        pass
try:
    print("TEST MSE: {0:.6f} \t {1:.6f}".format(np.mean(test_l2s, axis=0)[1], np.std(test_l2s, axis=0)[1]))
except IndexError:
    print("No completed runs.")
    
plt.show()

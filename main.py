# ----------------------- ----------------------- -----------------------
# This is the following code to the ML Sys Q
# -----------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import Aer, execute
from math import pi,log
from qiskit import *
import time
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import tensorflow as tf
import numpy as np
from qiskit import Aer, IBMQ
import os
from datetime import datetime

# Network saving and loading
epoch = 25
runtime_name = datetime.now().strftime("Date-%d%m%y--Hours-%H%M")
runtime_name += "-epoch-{}".format(epoch)
os.mkdir(runtime_name)
backend = Aer.get_backend('qasm_simulator')

# Subsample to SUBSAMPLE datapoints. This is due to computational cost.
# Chance SUBSAMPLE to what best suits your computer, to make a reasonable training time.
test_images,test_labels = tf.keras.datasets.mnist.load_data()
train_images = test_images[0].reshape(60000,784)
train_labels = test_images[1]
labels = test_images[1]
train_images = train_images/255
k=4
pca = PCA(n_components=k)
pca.fit(train_images)
# Computational cost is high for 60,000 data points. Change 6000 to what your system can handle
SUBSAMPLE = 1000
pca_data = pca.transform(train_images)[:SUBSAMPLE]
train_labels = train_labels[:SUBSAMPLE]
t_pca_data = pca_data.copy()
pca_descaler = [[] for _ in range(k)]
# Data Transformation Section
for i in range(k):
    if pca_data[:,i].min() < 0:
        pca_descaler[i].append(pca_data[:,i].min())
        pca_data[:,i] += np.abs(pca_data[:,i].min())
    else:
        pca_descaler[i].append(pca_data[:,i].min())
        pca_data[:,i] -= pca_data[:,i].min()
    pca_descaler[i].append(pca_data[:,i].max())
    pca_data[:,i] /= pca_data[:,i].max()
pca_data_rot= 2*np.arcsin(np.sqrt(pca_data))
valid_labels = None
valid_labels = train_labels==3
valid_labels += train_labels == 6

for col in range(pca_data.shape[1]):
    t_data_mean = pca_data[:,col].mean()
    t_data_std = pca_data[:,col].std()
    valid_upper_bound = pca_data[:,col] < t_data_mean+t_data_std
    valid_lower_bound = pca_data[:,col] > t_data_mean-t_data_std
    valid = np.logical_and(valid_upper_bound,valid_lower_bound)
    pca_data = pca_data[valid]
pca_data_rot3 = pca_data_rot[train_labels==3]
pca_data_rot6 = pca_data_rot[train_labels==6]


# Checkpointing code
def save_variables(var_dict, epoch, number_class):
    with open(f"{runtime_name}/Epoch-{epoch}-Variables-numbers-{number_class}", 'w') as file:
        file.write(str(var_dict))



# Ran_ang returns a random angle
def ran_ang():
    return np.random.rand() * 2 * np.pi


def single_qubit_unitary(circ_ident, qubit_index, values):
    circ_ident.ry(values[0], qubit_index)
    circ_ident.rz(values[1], qubit_index)


def dual_qubit_unitary(circ_ident, qubit_1, qubit_2, values):
    circ_ident.ryy(values[0], qubit_1, qubit_2)
    circ_ident.rzz(values[1], qubit_1, qubit_2)


def controlled_dual_qubit_unitary(circ_ident, control_qubit, act_qubit, values):
    circ_ident.cry(values[0], control_qubit, act_qubit)
    circ_ident.crz(values[1], control_qubit, act_qubit)


def traditional_learning_layer(circ_ident, num_qubits, values, style="Dual", qubit_start=1, qubit_end=5):
    if style == "Dual":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])
        for qub in np.arange(qubit_start, qubit_end - 1):
            dual_qubit_unitary(circ_ident, qub, qub + 1, values[str(qub) + "," + str(qub + 1)])
    elif style == "Single":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])
    elif style == "Controlled-Dual":
        for qub in np.arange(qubit_start, qubit_end):
            single_qubit_unitary(circ_ident, qub, values[str(qub)])
        for qub in np.arange(qubit_start, qubit_end - 1):
            dual_qubit_unitary(circ_ident, qub, qub + 1, values[str(qub) + "," + str(qub + 1)])
        for qub in np.arange(qubit_start, qubit_end - 1):
            controlled_dual_qubit_unitary(circ_ident, qub, qub + 1, values[str(qub) + "--" + str(qub + 1)])


def data_loading_circuit(circ_ident, num_qubits, values, qubit_start=1, qubit_end=5):
    k = 0
    for qub in np.arange(qubit_start, qubit_end):
        circ_ident.ry(values[k], qub)
        try:
            circ_ident.rz(values[k + 1], qub)
        except:
            circ_ident.rz(0, qub)
        k += 2


def swap_test(circ_ident, num_qubits):
    num_swap = num_qubits // 2
    for i in range(num_swap):
        circ_ident.cswap(0, i + 1, i + num_swap + 1)
    circ_ident.h(0)
    circ_ident.measure(0, 0)


def init_random_variables(q, style):
    trainable_variables = {}
    if style == "Single":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [ran_ang(), ran_ang()]
    elif style == "Dual":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [ran_ang(), ran_ang()]
            if i != q:
                trainable_variables[str(i) + "," + str(i + 1)] = [ran_ang(), ran_ang()]
    elif style == "Controlled-Dual":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [ran_ang(), ran_ang()]
            if i != q:
                trainable_variables[str(i) + "," + str(i + 1)] = [ran_ang(), ran_ang()]
                trainable_variables[str(i) + "--" + str(i + 1)] = [ran_ang(), ran_ang()]
    return trainable_variables


def init_gradient_variables(q, style):
    trainable_variables = {}
    if style == "Single":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [[], []]
    elif style == "Dual":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [[], []]
            if i != q:
                trainable_variables[str(i) + "," + str(i + 1)] = [[], []]
    elif style == "Controlled-Dual":
        for i in np.arange(1, q + 1):
            trainable_variables[str(i)] = [0, 0]
            if i != q:
                trainable_variables[str(i) + "," + str(i + 1)] = [[], []]
                trainable_variables[str(i) + "--" + str(i + 1)] = [[], []]
    return trainable_variables


def get_probabilities(circ, count=10000, inducing=False):
    if inducing == True:
        count *= 10
    job = execute(circ, backend, shots=count)
    results = job.result().get_counts(circ)
    try:
        prob = results['0'] / (results['1'] + results['0'])
        prob = (prob - 0.5)
        if prob <= 0:
            prob = 1e-16
        else:
            prob = prob * 2
    except:
        prob = 1
    return prob


# ------------------------------------------------------------------------------------
# We treat the first n qubits are the discriminators state. n is always defined as the
# integer division floor of the qubit count.
# This is due to the fact that a state will always be k qubits, therefore the
# number of total qubits must be 2k+1. 2k as we need k for the disc, and k to represent
# either the other learned quantum state, or k to represent a data point
# then +1 to perform the SWAP test. Therefore, we know that we will always end up
# with an odd number of qubits. We take the floor to solve for k. 1st k represents
# disc, 2nd k represents the "loaded" state be it gen or real data
# ------------------------------------------------------------------------------------
# Use different function calls to represent training a GENERATOR or training a DISCRIMINATOR
# ------------------------------------------------------------------------------------
def disc_real_training_circuit(training_variables, data, key=None, key_value=None, diff=False, fwd_diff=False):
    circ = QuantumCircuit(q, c)
    circ.h(0)
    if diff == True and fwd_diff == True:
        training_variables[key][key_value] += par_shift
    if diff == True and fwd_diff == False:
        training_variables[key][key_value] -= par_shift
    traditional_learning_layer(circ, q, training_variables, style=layer_style, qubit_start=1, qubit_end=q // 2 + 1)
    data_loading_circuit(circ, q, data, qubit_start=q // 2 + 1, qubit_end=q)
    swap_test(circ, q)
    if diff == True and fwd_diff == True:
        training_variables[key][key_value] -= par_shift
    if diff == True and fwd_diff == False:
        training_variables[key][key_value] += par_shift
    return circ


# Define loss function. SWAP Test returns probability, so minmax probability is logical
def cost_function(p, yreal):
    if yreal == 1:
        return -np.log(p)
    else:
        return -np.log(1 - p)


def update_weights(init_value, lr, grad):
    while lr * grad > 2 * np.pi:
        lr /= 10
        print("Warning - Gradient taking steps that are very large. Drop learning rate")
    weight_update = lr * grad
    new_value = init_value
    if new_value - weight_update > 2 * np.pi:
        new_value = (new_value - weight_update) - 2 * np.pi
    elif new_value - weight_update < 0:
        new_value = (new_value - weight_update) + 2 * np.pi
    else:
        new_value = new_value - weight_update
    return new_value


# ------------------------------------------------------------------------------------------
# THIS SECTION WE DO THE TUNING FOR WHAT WE KNOW WE WANT TO BE CHANGING!
# ------------------------------------------------------------------------------------------

q = 5  # Number of qubits = Dimensionality of data = round up to even number = num qubits
c = 1
circ = QuantumCircuit(q, c)
circ.h(0)
layer_style = "Dual"
train_var_0 = init_random_variables(q // 2, layer_style)
train_var_1 = init_random_variables(q // 2, layer_style)
train_var_2 = init_random_variables(q // 2, layer_style)

tracked_d_loss = []
tracked_d_loss1 = []
tracked_d_loss2 = []
gradients = []
learning_rate = 0.01
train_iter = 25
corr = 0
wrong = 0
loss_d_to_real = 0
print('Starting Training')
print('-' * 20)
print("train_var_0 training")
for epoch in range(train_iter):
    start = time.time()
    loss = [0, 0]
    par_shift = 0.5 * np.pi / ((1 + epoch) ** 0.5)
    for index, point in enumerate(pca_data_rot3):
        for key, value in train_var_0.items():
            if str(q // 2 + 1) in key:
                break
            for key_value in range(len(value)):
                forward_diff = -np.log(get_probabilities(
                    disc_real_training_circuit(train_var_0, point, key, key_value, diff=True, fwd_diff=True)))
                backward_diff = -np.log(get_probabilities(
                    disc_real_training_circuit(train_var_0, point, key, key_value, diff=True, fwd_diff=False)))
                df = 0.5 * (forward_diff - backward_diff)
                train_var_0[key][key_value] -= df * learning_rate
    print('Time for Epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    print("-" * 20)
    save_variables(train_var_0, epoch, 3)

for epoch in range(train_iter):
    start = time.time()
    loss = [0, 0]
    par_shift = 0.5 * np.pi / ((1 + epoch) ** 0.5)
    for index, point in enumerate(pca_data_rot6):
        for key, value in train_var_1.items():
            if str(q // 2 + 1) in key:
                break
            for key_value in range(len(value)):
                forward_diff = -np.log(get_probabilities(
                    disc_real_training_circuit(train_var_1, point, key, key_value, diff=True, fwd_diff=True)))
                backward_diff = -np.log(get_probabilities(
                    disc_real_training_circuit(train_var_1, point, key, key_value, diff=True, fwd_diff=False)))
                df = 0.5 * (forward_diff - backward_diff)
                train_var_1[key][key_value] -= df * learning_rate
    print('Time for Epoch {} is {} sec'.format(epoch + 1, time.time() - start))
    print("-" * 20)
    save_variables(train_var_1, epoch, 6)

#CHANGE PCA_DATA_ROT VARS HERE
pca_data = []
[pca_data.append(x) for x in pca_data_rot3]
[pca_data.append(x) for x in pca_data_rot6]
labels = []
[labels.append(0) for _ in range(len(pca_data_rot3))]
[labels.append(1) for _ in range(len(pca_data_rot6))]

correct = 0
wrong = 0
ones = []
zeros = []
layer_style = 'Dual'
for index, x in enumerate(pca_data):
    p0 = get_probabilities(disc_real_training_circuit(train_var_0, x, None, None, diff=False, fwd_diff=False))
    p1 = get_probabilities(disc_real_training_circuit(train_var_1, x, None, None, diff=False, fwd_diff=False))
    tp = p0 + p1
    p0 = p0 / tp
    p1 = p1 / tp
    probs = np.array([p0, p1])
    if np.argmax(probs) == labels[index]:
        correct += 1
    else:
        wrong += 1

print(f"Accuracy {correct / (correct + wrong)}")


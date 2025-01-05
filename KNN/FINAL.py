import pandas as pd
import numpy as np
from util.utilfuncs import *
import numpy

# fix random seed for reproducibility
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("data/pima-indians-diabetes.data.csv", delimiter=",")
numpy.random.shuffle(dataset)
splitratio = 0.8

# Split dataset
X_train = dataset[:int(len(dataset)*splitratio),0:8]
X_test = dataset[int(len(dataset)*splitratio):,0:8]
Y_train = dataset[:int(len(dataset)*splitratio),8]
Y_test = dataset[int(len(dataset)*splitratio):,8]
print(X_train)
print(Y_train)


nitr = 150
kst = 3
k = range(kst,nitr)
Accuracy = np.zeros(len(k))
Recall = np.zeros(len(k))
Precision = np.zeros(len(k))
F1 = np.zeros(len(k))
TP_array = np.zeros(len(k))
TN_array = np.zeros(len(k))
FP_array = np.zeros(len(k))
FN_array = np.zeros(len(k))
for K in k:
    #print(K)
    #print(K-kst)
    yh = np.zeros((len(X_test)))
    for j in range(len(X_test)):
        # print(sum(Y_train[mindist(X_test[j,:],X_train,K)]))
        array=Y_train[mindist(X_test[j,:],X_train,K)]
        if np.count_nonzero(array == 1)>np.count_nonzero(array == 0):
        #if sum(Y_train[mindist(X_test[j,:],X_train,K)])>K/2:
            yh[j] = 1
    print(K)
    Accuracy[K-kst], Recall[K-kst], Precision[K-kst], F1[K-kst]=eval(yh,Y_test)
    TP, TN, FP, FN = eval(yh, Y_test, return_confusion_matrix=True)
    TP_array[K-kst], TN_array[K-kst], FP_array[K-kst], FN_array[K-kst] = TP, TN, FP, FN

import plotly.graph_objects as go
from plotly.subplots import make_subplots

k_values=list(k)
# Create a Plotly figure with two subplots
fig = make_subplots(
    rows=2, cols=1,
    subplot_titles=("Evaluation Metrics vs. K", "Confusion Matrix Values vs. K"),
    shared_xaxes=True
)

# Add traces for evaluation metrics to the first subplot
fig.add_trace(go.Scatter(x=k_values, y=Accuracy, mode='lines', name='Accuracy', line=dict(color='blue')), row=1, col=1)
fig.add_trace(go.Scatter(x=k_values, y=Recall, mode='lines', name='Recall', line=dict(color='green')), row=1, col=1)
fig.add_trace(go.Scatter(x=k_values, y=Precision, mode='lines', name='Precision', line=dict(color='red')), row=1, col=1)
fig.add_trace(go.Scatter(x=k_values, y=F1, mode='lines', name='F1-Score', line=dict(color='purple')), row=1, col=1)

# Add traces for confusion matrix values to the second subplot
fig.add_trace(go.Scatter(x=k_values, y=TP_array, mode='lines', name='True Positives (TP)', line=dict(color='orange')), row=2, col=1)
fig.add_trace(go.Scatter(x=k_values, y=TN_array, mode='lines', name='True Negatives (TN)', line=dict(color='blue')), row=2, col=1)
fig.add_trace(go.Scatter(x=k_values, y=FP_array, mode='lines', name='False Positives (FP)', line=dict(color='pink')), row=2, col=1)
fig.add_trace(go.Scatter(x=k_values, y=FN_array, mode='lines', name='False Negatives (FN)', line=dict(color='brown')), row=2, col=1)

# Update layout for titles and labels
fig.update_layout(
    title='Evaluation Metrics and Confusion Matrix Values vs. K',
    xaxis_title='K',
    yaxis_title='Metric Value',
    legend_title='Metrics & Confusion Matrix',
    height=800
)

# Show the plot
fig.show()
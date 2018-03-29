import numpy as np
import pandas as pd
import mxnet as mx
import sys

def getType(a, b, c):
    Max = a
    result = 1
    if b > Max:
        Max = b
        result = 2
    if c > Max:
        Max = c
        result = 3
        if b > c:
            Max = b
            result = 2
    return result
#################################################################################
# pre trained model path
model_path_class_1 = "model/class1model"
model_path_class_2 = "model/class2model"
model_path_class_3 = "model/class3model"

# args i/o
input_test = sys.argv[1]
output_file = sys.argv[2]

# iteration
num_round = 1
#################################################################################
# load testing dataset
data = np.loadtxt(input_test, ndmin = 2, delimiter=",")
# data = pd.read_csv(input_test).reshape((-1, 1, 20, 20))
X1 = data[:,0:400].reshape(len(data),1,20,20)
# X1 = data.iloc[:,-1].astype('float32')
#################################################################################
# load model 1
model_1 = mx.model.FeedForward.load(prefix=model_path_class_1, epoch=num_round)

# predict model 1
pred_1 = model_1.predict(X1,num_round)
pred_1 = round(pred_1[0][1], 4)
print("class1 : {}".format(pred_1*100))

#################################################################################
# load model 2
model_2 = mx.model.FeedForward.load(prefix=model_path_class_2, epoch=num_round)

# predict model 2
pred_2 = model_2.predict(X1,num_round)
pred_2 = round(pred_2[0][1], 4)
print("class2 : {}".format(pred_2*100))

#################################################################################
# load model 3
model_3 = mx.model.FeedForward.load(prefix=model_path_class_3, epoch=num_round)

# predict model 3
pred_3 = model_3.predict(X1,num_round)
pred_3 = round(pred_3[0][1], 4)
print("class3 : {}".format(pred_3*100))
#################################################################################
family = getType(pred_1,pred_2,pred_3)
# print("result : \n family class a : {} \n family class b : {} \n family class c : {} \n class : {}".format(pred_1[0]*100, pred_2[0]*100, pred_3[0]*100,family))
f = open(output_file,'w')
f.write(str(pred_1*100) + ',' + str(pred_2*100) + ',' + str(pred_3*100) + ',' + str(family))
f.close()
print("Result saved.")

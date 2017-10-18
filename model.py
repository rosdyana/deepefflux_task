import numpy
import mxnet as mx
import sys

# pre trained model path
model_path = sys.argv[1]
model_path_class_1 = ''.join(model_path,"model_1")
model_path_class_2 = ''.join(model_path,"model_2")
model_path_class_3 = ''.join(model_path,"model_3")

# args i/o
input_test = sys.argv[2]
output_file = sys.argv[3]

# iteration
num_round = 1

# load testing dataset
batch = numpy.loadtxt(input_test, ndmin = 2, delimiter=","))

# load model 1
model_1 = mx.model.FeedFoward.load(prefix=model_path_class_1, iteration=num_round)

# predict model 1
pred_1 = model_1.predict(batch)

# load model 2
model_2 = mx.model.FeedFoward.load(prefix=model_path_class_1, iteration=num_round)

# predict model 2
pred_2 = model_2.predict(batch)

# load model 3
model_3 = mx.model.FeedFoward.load(prefix=model_path_class_1, iteration=num_round)

# predict model 3
pred_3 = model_3.predict(batch)

# compare the highest acc.

output = np_utils.categorical_probas_to_classes(predictions)
f = open(output_file,'w')
for x in output:
    f.write(str(x) + '\n')
f.close()
print("Result saved.")

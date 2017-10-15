import numpy
from keras.utils import np_utils
from keras.models import model_from_json
import sys

#define params
json_model = sys.argv[1]
h5_model = sys.argv[2]
test_file = sys.argv[3]
file_out = sys.argv[4]

# load testing dataset
ds = numpy.loadtxt(test_file, ndmin = 2, delimiter=",")
X1 = ds[:,0:400].reshape(len(ds),1,20,20)

# load json and create model
json_file = open(json_model, 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(h5_model)
print("Loaded model from disk.")

predictions = loaded_model.predict(X1)
output = np_utils.categorical_probas_to_classes(predictions)
f = open(file_out,'w')
for x in output:
    f.write(str(x) + '\n')
f.close()
print("Result saved.")

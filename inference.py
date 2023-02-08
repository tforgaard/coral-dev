import os
import pathlib
from pycoral.utils import edgetpu
from pycoral.utils import dataset
from pycoral.adapters import common
from pycoral.adapters import classify

from tflite_runtime.interpreter import Interpreter
from tflite_runtime.interpreter import load_delegate

from PIL import Image
from pathlib import Path
import numpy as np

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
# model_file = os.path.join(script_dir, 'quadrotor-tiny-two-sensors-no-rnn-final-model_quantized_edgetpu.tflite')

model_file = os.path.join(script_dir, 'quadrotor-tiny-two-sensors-no-rnn-final-model_unquantized.tflite')

#label_file = os.path.join(script_dir, 'imagenet_labels.txt')
#image_file = os.path.join(script_dir, 'parrot.jpg')


# run_dir = "runs/quadrotor-tiny-two-sensors-final-test-seed-42-terrain-dev"
run_dir = "../"
obsv_data_entry = None
hidden_state_entry = None
run_dir = Path(run_dir)
obsv_data_dir = run_dir / 'obsv_logs'

if obsv_data_entry is None:
    #find latest one
    obsv_data_entries = sorted(list(obsv_data_dir.glob("obsv_log_*.npz")))
    obsv_data_entry = obsv_data_entries[-1].name


if hidden_state_entry is None:
    #find latest one
    hidden_state_entries = sorted(list(obsv_data_dir.glob("hs_log_*.npz")))
    hidden_state_entry = hidden_state_entries[-1].name


obsv_data = np.load(obsv_data_dir / obsv_data_entry, mmap_mode='r')
hidden_data = np.load(obsv_data_dir / hidden_state_entry, mmap_mode='r')

print(obsv_data.shape)
print(hidden_data.shape)

def rep_dataset():
    """Generator function to produce representative dataset for post-training quantization."""

    # Use a few samples from the training set
    N = 30
    obsv_data_ind = np.random.choice(obsv_data.shape[0],N,replace=False)
    print(obsv_data_ind.shape)
    
    for i in range(N):
        X = np.expand_dims(obsv_data[obsv_data_ind[i]],axis=1)
        hs = np.expand_dims(hidden_data[obsv_data_ind[i]],axis=2)
        #print(X.shape)
        #print(hs.shape)
        #print(hs[0].shape)
        #print(hs[1].shape)
        yield {'input': X[0]} #, 'h0': hs[0,0], 'c0': hs[1,0]}

# Initialize the TF interpreter
# interpreter = edgetpu.make_interpreter(model_file)
interpreter = Interpreter(model_file)
interpreter.allocate_tensors()

# Resize the image
# size = common.input_size(interpreter)
# image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

# Run an inference
dataset = rep_dataset()

X = next(dataset)['input']

print(X.shape)

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# common.set_input(interpreter, X)
input_shape = input_details[0]['shape']

print(input_details)

print(output_details)

interpreter.set_tensor(input_details[0]['index'], X)
interpreter.invoke()
pose = common.output_tensor(interpreter, 0).copy() # .reshape(_NUM_KEYPOINTS, 3)
# classes = classify.get_classes(interpreter, top_k=1)
print(pose)
# Print the result
# labels = dataset.read_label_file(label_file)
# for c in classes:
#   print('%s: %.5f' % (labels.get(c.id, c.id), c.score))
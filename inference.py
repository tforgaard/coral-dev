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
import time

# Specify the TensorFlow model, labels, and image
script_dir = pathlib.Path(__file__).parent.absolute()
# model_file = os.path.join(script_dir, "models", 'quadrotor-tiny-two-sensors-no-rnn-final-model_quantized_edgetpu.tflite')
model_file = os.path.join(script_dir, "models", 'quadrotor-tiny-two-sensors-relu-no-rnn-model', 'tflite', 'model_int8_edgetpu.tflite')


# run_dir = "runs/quadrotor-tiny-two-sensors-final-test-seed-42-terrain-dev"
run_dir = "./"
obsv_data_entry = None
hidden_state_entry = None
run_dir = Path(run_dir)
obsv_data_dir = run_dir / 'data'

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


print(f"data std: {np.std(obsv_data)}")
print(f"data mean: {np.mean(obsv_data)}")
print(obsv_data.shape)
print(hidden_data.shape)

def rep_dataset():
    """Generator function to produce representative dataset for post-training quantization."""

    # Use a few samples from the training set
    N = 30
    # obsv_data_ind = np.random.choice(obsv_data.shape[0],N,replace=False)
    obsv_data_ind = np.array(list(range(N)))
    print(obsv_data_ind.shape)
    
    for i in range(N):
        X = np.expand_dims(obsv_data[obsv_data_ind[i]],axis=1)
        hs = np.expand_dims(hidden_data[obsv_data_ind[i]],axis=2)
        yield {'input': X[0]} #, 'h0': hs[0,0], 'c0': hs[1,0]}

# Initialize the TF interpreter
print("creating interpreter")
interpreter = edgetpu.make_interpreter(model_file)
# interpreter = Interpreter(model_file)
print("allocating tensors")
interpreter.allocate_tensors()

# Resize the image
# size = common.input_size(interpreter)
# image = Image.open(image_file).convert('RGB').resize(size, Image.ANTIALIAS)

print("creating dataset")
dataset = rep_dataset()

# Model must be uint8 quantized
if common.input_details(interpreter, 'dtype') != np.uint8:
    raise ValueError('Only support uint8 input type.')


# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# print(input_details)
# print(output_details)



# Image data must go through two transforms before running inference:
# 1. normalization: f = (input - mean) / std
# 2. quantization: q = f / scale + zero_point
# The following code combines the two steps as such:
# q = (input - mean) / (std * scale) + zero_point
# However, if std * scale equals 1, and mean - zero_point equals 0, the input
# does not need any preprocessing (but in practice, even if the results are
# very close to 1 and 0, it is probably okay to skip preprocessing for better
# efficiency; we use 1e-5 below instead of absolute zero).
params = common.input_details(interpreter, 'quantization_parameters')
scale = params['scales']
zero_point = params['zero_points']
mean = 0.94974434 # args.input_mean
std = 0.26410374 # args.input_std

print(abs(scale * std - 1))
print(abs(mean - zero_point))

def preprocess_input(X):
    if abs(scale * std - 1) < 1e-5 and abs(mean - zero_point) < 1e-5:
        # Input data does not require preprocessing.
        # common.set_input(interpreter, image)
        interpreter.set_tensor(input_details[0]['index'], X.astype(np.uint8))
    else:
        print("preprocessing data")
        # Input data requires preprocessing
        normalized_input = (np.asarray(X) - mean) / (std * scale) + zero_point
        np.clip(normalized_input, 0, 255, out=normalized_input)
        # common.set_input(interpreter, normalized_input.astype(np.uint8))

        interpreter.set_tensor(input_details[0]['index'], normalized_input.astype(np.uint8))

# output_params = common.output_details(interpreter, 'quantization_parameters')
output_params = output_details[0]['quantization_parameters']

output_scale = output_params['scales']
output_zero_point = output_params['zero_points']

def process_output(y):
    if abs(output_scale * std - 1) < 1e-5 and abs(mean - output_zero_point) < 1e-5:
        # Output data does not require preprocessing.
        return y.astype(np.float32)
    else:
        print("postprocessing data")
        # Output data requires preprocessing
        scaled_output = (y.astype(np.float32)  - zero_point) * (std * scale) + mean
        # normalized_input = (np.asarray(X) - mean) / (std * scale) + zero_point
        # np.clip(normalized_input, 0, 255, out=normalized_input)

        return scaled_output

count = 10 
# Run inference
print('----INFERENCE TIME----')
print('Note: The first inference on Edge TPU is slow because it includes',
'loading the model into Edge TPU memory.')
for _ in range(count):
    
    X = next(dataset)['input']
    preprocess_input(X)
    start = time.perf_counter()
    interpreter.invoke()
    inference_time = time.perf_counter() - start
    # classes = classify.get_classes(interpreter, args.top_k, args.threshold)
    
    pose = common.output_tensor(interpreter, 0).copy() # .reshape(_NUM_KEYPOINTS, 3)
    print('%.1fms' % (inference_time * 1000))
    print(pose)
    print(process_output(pose))



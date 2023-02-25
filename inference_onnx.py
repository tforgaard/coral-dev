
import time
import numpy as np
import onnxruntime as ort
#import onnx

from utils.utils import load_rep_dataset

def infer_onnxruntime(model_path):
    print('Onnxruntime Inference')
    print('==========================')
    print()

    #onnx_model = onnx.load(model_path)
    #onnx.checker.check_model(onnx_model)



    so = ort.SessionOptions()
    so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    exproviders = ['CPUExecutionProvider']


    #ort_session = onnxruntime.InferenceSession(model_onnx_path, so, providers=exproviders)
    ort_session = ort.InferenceSession(model_path, so, providers=exproviders)


    ort_session_input_info = ort_session.get_inputs()

    for input_info in ort_session_input_info:
        print(input_info)

    # TODO: use input_info...
    include_hidden = False
    dummy_model_input = {'input': np.zeros((1, 145)).astype(np.float32)}

    if len(ort_session_input_info) == 3:
        include_hidden = True
        dummy_model_input = {**dummy_model_input, 
                            'h0': np.zeros((1,1, 256)).astype(np.float32),
                            'c0': np.zeros((1,1, 256)).astype(np.float32)}
        

    dataset = load_rep_dataset(include_hidden=include_hidden,debug=True)


    #warm up run
    ort_outs = ort_session.run(None,dummy_model_input)
    # ort_outs = io_binding.copy_outputs_to_cpu()


    latency = []

    total_samples = 10000

    #for i in range(total_samples):
    for data in dataset():
        t0 = time.time()
        # ort_session.run_with_iobinding(io_binding)
        ort_outs = ort_session.run(None,data)
        latency.append(time.time() - t0)
        # print(ort_outs)
        # ort_outs = io_binding.copy_outputs_to_cpu()

    print('Number of runs:', len(latency))
    print("Average onnxruntime Inference time = {} ms".format(format(sum(latency) * 1000 / len(latency), '.4f'))) 


if __name__ == "__main__":
    infer_onnxruntime("./models/quadrotor-tiny-two-sensors-final/onnx/quadrotor-tiny-two-sensors-final.onnx")
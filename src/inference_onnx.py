
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from argparse import ArgumentParser
#import onnx

from pathlib import Path

from utils.utils import load_rep_dataset, create_dummy_input, print_session_options

def infer_onnxruntime(model_path, intra_op_num_threads=1, inter_op_num_threads=1, execution_mode="sequential", quantize_dyn=False, debug=False):
    print('Onnxruntime Inference')
    print('==========================')
    print()

    #onnx_model = onnx.load(model_path)
    #onnx.checker.check_model(onnx_model)


    if quantize_dyn:

        print("using dymamic quantized model")
        
        model_fp32 = model_path
        model_p = Path(model_path)
        model_quant = str(model_p.parent / f'{model_p.stem}_quant.onnx')

        quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

        model_path = model_quant


    so = ort.SessionOptions()
    if execution_mode == "sequential":
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    elif execution_mode == "parallel":
        so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
    else:
        raise KeyError(f"execution mode: {execution_mode} not found!")
    
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    if debug:
        so.enable_profiling = True 

    so.intra_op_num_threads = intra_op_num_threads
    so.inter_op_num_threads = inter_op_num_threads


    print_session_options(so)
    exproviders = ['CPUExecutionProvider']
    ort_session = ort.InferenceSession(model_path, so, providers=exproviders)


    ort_session_input_info = ort_session.get_inputs()

    if debug:
        for input_info in ort_session_input_info:
            print(input_info)

    # TODO: use input_info...
    
    include_hidden = True if len(ort_session_input_info) == 3 else False
    
    dummy_model_input = create_dummy_input(include_hidden=include_hidden, tensor=False)

        
    dataset = load_rep_dataset(include_hidden=include_hidden,debug=debug)

    #warm up run
    ort_outs = ort_session.run(None,next(dummy_model_input))
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


    ave_inference_time = sum(latency) * 1000 / len(latency)
    print('Number of runs:', len(latency))
    print("Average onnxruntime Inference time = {} ms".format(format(ave_inference_time, '.4f'))) 


    return ave_inference_time

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--debug", action="store_true")
    
    parser.add_argument("--intra_op_num_threads", default=1, type=int)
    parser.add_argument("--inter_op_num_threads", default=1, type=int)
    parser.add_argument("--execution_mode", default="sequential") 
    parser.add_argument("--quantize_dyn", action="store_true")
    args = parser.parse_args()
    infer_onnxruntime(**vars(args))
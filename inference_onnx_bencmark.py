
import time
import numpy as np
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from argparse import ArgumentParser
#import onnx

from pathlib import Path

from utils.utils import load_rep_dataset, create_dummy_input, print_session_options


import inference_onnx





if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")

    args = parser.parse_args()

    best_inf_time = 1000

    for intra_op_num_threads in range(1,5):
        for inter_op_num_threads in range(1,5):
            for execution_mode in ['sequential', 'parallel']:
                for quantize_dyn in [True, False]:
                    inf_time = inference_onnx.infer_onnxruntime(args.model_path,    intra_op_num_threads=intra_op_num_threads, 
                                                                                    inter_op_num_threads=inter_op_num_threads,
                                                                                    execution_mode=execution_mode,
                                                                                    quantize_dyn=quantize_dyn
                                                                                )
                    if inf_time < best_inf_time:
                        best_inf_time = inf_time

    print(f"best inference time: {best_inf_time}")
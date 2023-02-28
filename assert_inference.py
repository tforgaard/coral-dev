
import time
import numpy as np
import torch
from argparse import ArgumentParser
from inspect import signature
from pathlib import Path

import onnxruntime as ort

from utils.utils import load_rep_dataset, create_dummy_input, DEFAULT_INPUT_NAMES

def infer_torch(model_dir, interpreters, include_hidden=False, torch_type=None, onnx_quantize_dyn=False):
    print('Inference Assertion')
    print('==========================')
    print()

    print(f"loading models from directory: {model_dir}")

    model_dir = Path(model_dir)

    inference_dict = {}

    if "torch" in interpreters:
        model_path_torch = str(model_dir / "torchscript" / f"{model_dir.name}.pt")
        print(f"loading torch model: {model_path_torch}")
        model_torch = torch.jit.load(model_path_torch)

        if torch_type == "freezed":
            print("freezing torch model")
            model_torch = torch.jit.freeze(model_torch)
        
        elif torch_type == "mobile":
            print("using torch mobile optimization")
            from torch.utils.mobile_optimizer import optimize_for_mobile
            model_torch = optimize_for_mobile(model_torch)

        elif torch_type == "inference":
            print("using torch inference optimization")
            model_torch = torch.jit.optimize_for_inference(model_torch)

        elif torch_type == "onednn":
            print("enable torch onednn fusion")
            torch.jit.enable_onednn_fusion(True)
        
        elif torch_type is not None:
            raise KeyError(f"torch optimization type: {torch_type} not found!")
        

        torch_input_names = ('input', 'inp', 'inp0')
        
        dataset_torch = load_rep_dataset(include_hidden=include_hidden,debug=False, tensor=True, input_names=torch_input_names)

        inference_dict['torch'] = {'dataset': dataset_torch(), 
                                   'model': model_torch,
                                   'run_model': lambda x : model_torch(**x), 
                                   'input_names': torch_input_names}

    if "onnx" in interpreters:

        model_path_onnx = str(model_dir / "onnx" / f"{model_dir.name}.onnx")

        if onnx_quantize_dyn:
            from onnxruntime.quantization import quantize_dynamic, QuantType

            print("using dymamic quantized model")
            
            model_fp32 = model_path_onnx
            model_p = Path(model_path_onnx)
            model_quant = str(model_p.parent / f'{model_p.stem}_quant.onnx')

            quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)

            model_path_onnx = model_quant


        so = ort.SessionOptions()
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        exproviders = ['CPUExecutionProvider']

        model_onnx = ort.InferenceSession(model_path_onnx, so, providers=exproviders)

        dataset_onnx = load_rep_dataset(include_hidden=include_hidden,debug=False)

        inference_dict['onnx'] = {'dataset': dataset_onnx(), 
                                  'model': model_onnx,
                                  'run_model':  lambda x : model_onnx.run(None,x),
                                  'input_names': DEFAULT_INPUT_NAMES}

    rtol = 1e-10
    atol = 1e-10

    finished = False
    prev_outputs = {}
    while not finished:
        model_outs = []        
        for model_name, model_dict in inference_dict.items():
            data = next(model_dict['dataset'],None)
            if data is None:
                print("finished!")
                finished = True
                break

            # use prev hidden states                
            if include_hidden and prev_outputs.get(model_name) is not None:

                data = {model_dict['input_names'][0]: data[model_dict['input_names'][0]], 
                        model_dict['input_names'][1]: prev_outputs[model_name][-2],
                        model_dict['input_names'][2]: prev_outputs[model_name][-1]}

            model_out = model_dict['run_model'](data)
            prev_outputs[model_name] = model_out # with correct datatype
            if isinstance(model_out[0],torch.Tensor):
                model_out = [out.detach().numpy() for out in model_out]
            model_outs.append(model_out)

            #TODO: compare prev hidden states with true hidden states?

        # check outputs
        check_tolerance = True
        while check_tolerance:
            check_tolerance = False
            for j in range(len(model_outs)):
                for i in range(j+1,len(model_outs)):
                    for out_j, out_i in zip(model_outs[j],model_outs[i]):
                        try:
                            np.testing.assert_allclose(out_j, out_i, rtol=rtol, atol=atol)
                        except AssertionError as e:
                            print(e)
                            print(f"tolerance test failed with rtol: {rtol} and atol: {atol}")
                            print("decreasing tolerance")
                            rtol *= 10
                            atol *= 10
                            print("=====================================================")
                            check_tolerance = True

    print(f"tolerance test passed with rtol: {rtol}, atol: {atol}")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_dir")
    parser.add_argument("--interpreters", default=[],nargs="+")
    parser.add_argument("--include_hidden",  action="store_true")
    parser.add_argument("--torch_type", default=None)
    parser.add_argument("--onnx_quantize_dyn", action="store_true")
    args = parser.parse_args()
    infer_torch(**vars(args))
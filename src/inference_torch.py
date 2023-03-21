
import time
import numpy as np
import torch
from argparse import ArgumentParser
from inspect import signature
from pathlib import Path

from utils.utils import load_rep_dataset, create_dummy_input

def infer_torch(model_path, include_hidden=False):
    print('Torchscript Inference')
    print('==========================')
    print()

    print(model_path)
    net = torch.jit.load(model_path)
    # net.eval()


    model_p = Path(model_path)
    net_pre_freezed = torch.jit.load(str(model_p.parent / f"{model_p.stem}_freezed.pt"))

    net_post_freezed = torch.jit.freeze(net)

    
    #torch.jit.enable_onednn_fusion(True)

    from torch.utils.mobile_optimizer import optimize_for_mobile

    net_opt_inf = torch.jit.optimize_for_inference(net)

    net_opt_mobile = optimize_for_mobile(net)

    net_sig = signature(net)
    
    print(net_sig)
    print(net_sig.parameters)


    input_names =('input', 'inp', 'inp0')

    for model, model_name in zip((net, net_pre_freezed, net_post_freezed, net_opt_inf, net_opt_mobile), ("traced", "pre_freezed", "post_freezed", "opt inf", "opt mobile")):

        dummy_model_input = create_dummy_input(include_hidden=include_hidden, tensor=True, input_names=input_names)
        
        dataset = load_rep_dataset(include_hidden=include_hidden,debug=True, tensor=True, input_names=input_names)


        #warm up run
        model(**next(dummy_model_input))

        latency = []

        total_samples = 10000

        #for i in range(total_samples):
        for data in dataset():
            t0 = time.time()
            net_outs = model(**data)
            latency.append(time.time() - t0)
            #print(net_outs)

        print('Number of runs:', len(latency))
        print(f"Average torchscript {model_name} Inference time = {sum(latency) * 1000 / len(latency):.4f} ms") 


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_path")
    parser.add_argument("--include_hidden", action="store_true")
    args = parser.parse_args()
    infer_torch(**vars(args))

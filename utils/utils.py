

from pathlib import Path
import numpy as np
import onnxruntime as ort

DEFAULT_INPUT_NAMES = ('input', 'h0', 'c0')

def print_session_options(so):
    print(f"onnx session options")
    print('==========================')   
    print(f"execution mode: {so.execution_mode}")
    print(f"intra op num threads: {so.intra_op_num_threads}")
    print(f"inter op num threads: {so.inter_op_num_threads}")
    print(f"graph optimization level: {so.graph_optimization_level}")

    print()

def create_dummy_input(size=1, include_hidden=False, tensor=False, input_names=None, shapes=None, seed=None):

    # shapes: tuple of tuples

    np.random.seed(seed)

    if tensor:
        import torch

    if input_names is None:
        input_names = DEFAULT_INPUT_NAMES


    for i in range(size):

        dummy_model_input = {input_names[0]: np.random.rand(1, 145).astype(np.float32)}

        if include_hidden:
            dummy_model_input = {**dummy_model_input, 
                                input_names[1]: np.zeros((1,1, 256)).astype(np.float32),
                                input_names[2]: np.zeros((1,1, 256)).astype(np.float32)}

        if tensor:
            for k,v in dummy_model_input.items():
                dummy_model_input[k] = torch.from_numpy(v)

        yield dummy_model_input


def load_rep_dataset(dataset_dir = None, include_hidden=False, debug=False, tensor=False, input_names=None):

    if tensor:
        import torch

    if dataset_dir is None:
        #TODO fix this
        run_dir = "./"
        run_dir = Path(run_dir)
        dataset_dir = run_dir / 'data'
    
    obsv_data_entry = None
    hidden_state_entry = None


    if obsv_data_entry is None:
        #find latest one
        obsv_data_entries = sorted(list(dataset_dir.glob("obsv_log_*.npz")))
        obsv_data_entry = obsv_data_entries[-1].name


    if hidden_state_entry is None:
        #find latest one
        hidden_state_entries = sorted(list(dataset_dir.glob("hs_log_*.npz")))
        hidden_state_entry = hidden_state_entries[-1].name


    obsv_data = np.load(dataset_dir / obsv_data_entry, mmap_mode='r')
    hidden_data = np.load(dataset_dir / hidden_state_entry, mmap_mode='r')

    if debug:
        
        print("data info:")
        print(obsv_data.shape)
        print(f"data std: {np.std(obsv_data)}")
        print(f"data mean: {np.mean(obsv_data)}")
        print()

        print("hidden info:")
        print(hidden_data.shape)
        print(f"hidden std: {np.std(hidden_data)}") # is np.std unbiased?
        print(f"hidden mean: {np.mean(hidden_data)}")
        print()

    if input_names is None:
        input_names = DEFAULT_INPUT_NAMES

    def rep_dataset():
        """Generator function to produce representative dataset for post-training quantization."""

        # Use a few samples from the training set

        N = obsv_data.shape[0] # * obsv_data.shape[1]
        # obsv_data_ind = np.random.choice(obsv_data.shape[0],N,replace=False)
        obsv_data_ind = np.array(list(range(N)))
        if debug:
            print(obsv_data_ind.shape)
        
        for i in range(N):
            X = np.expand_dims(obsv_data[obsv_data_ind[i]],axis=1)
            hs = np.expand_dims(np.expand_dims(hidden_data[obsv_data_ind[i]],axis=2),axis=2)
            
            if tensor:
                X = torch.from_numpy(X)
                hs = torch.from_numpy(hs)

            if include_hidden:
                #TODO: standardize naming
                yield {input_names[0]: X[0], input_names[1]: hs[0,0], input_names[2]: hs[1,0]}
            else:
                yield {input_names[0]: X[0]}
        

    return rep_dataset
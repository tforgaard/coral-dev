

from pathlib import Path
import numpy as np

def load_rep_dataset(dataset_dir = None, include_hidden=True, debug=False):

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

    def rep_dataset():
        """Generator function to produce representative dataset for post-training quantization."""

        # Use a few samples from the training set

        N = obsv_data.shape[0] # * obsv_data.shape[1]
        # obsv_data_ind = np.random.choice(obsv_data.shape[0],N,replace=False)
        obsv_data_ind = np.array(list(range(N)))
        print(obsv_data_ind.shape)
        
        for i in range(N):
            X = np.expand_dims(obsv_data[obsv_data_ind[i]],axis=1)
            hs = np.expand_dims(np.expand_dims(hidden_data[obsv_data_ind[i]],axis=2),axis=2)
            if include_hidden:
                #TODO: standardize naming
                yield {'input': X[0], 'h0': hs[0,0], 'c0': hs[1,0]}
            else:
                yield {'input': X[0]}
        

    return rep_dataset
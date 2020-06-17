import torch
import os


def get_set_device_functions():
    dev = {"device": None}
    def get_device():
        if dev["device"] is None:
            raise ValueError("No device set. Call `set_device(device)` first.")
        return dev["device"]

    def set_device(device):
        if "cuda" in str(device):
            if not torch.cuda.is_available():
                raise ValueError("GPU device provided but `torch.cuda.is_available()` = False.")
        device = device if type(device) == torch.device else torch.device(device)
        dev["device"] = device

    def set_cpu_as_device():
        dev["device"] = torch.device('cpu')

    def set_first_gpu():
        assert torch.cuda.is_available()
        device = torch.device('cuda:0')
        if "cuda" in str(device):
            if not torch.cuda.is_available():
                raise ValueError("GPU device provided but `torch.cuda.is_available()` = False.")
        dev["device"] = device

    return get_device, set_device, set_cpu_as_device, set_first_gpu


get_device, set_device, set_cpu_as_device, set_first_gpu = get_set_device_functions()


def build_cache(cachedir=None):
    from joblib import Memory
    memory = Memory(os.getcwd() if cachedir is None else cachedir, verbose=0)
    return memory


memory = build_cache()

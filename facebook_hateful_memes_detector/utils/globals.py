import torch
import os


def get_set_device_functions():
    dev = {"device": None}

    def set_global(key, value):
        assert type(key) == str
        if key in dev:
            raise ValueError("Globals can be set only once. Global %s has already been set." % key)
        dev[key] = value

    def get_global(key):
        assert type(key) == str
        if key not in dev:
            raise ValueError("Global key = %s not set. Use `set_global(%s, <value>)` first" % (key, key))
        return dev[key]


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

    return get_device, set_device, set_cpu_as_device, set_first_gpu, set_global, get_global


get_device, set_device, set_cpu_as_device, set_first_gpu, set_global, get_global = get_set_device_functions()


def build_cache(cachedir=None):
    from joblib import Memory
    memory = Memory(os.getcwd() if cachedir is None else cachedir, verbose=0)
    # @memory.cache
    return memory


memory = build_cache()

# https://github.com/airsplay/lxmert

from .file_utils import cached_path
from .entry import LXRTEncoder


def model_name_to_url(name):
    assert type(name) == str
    if name == "full" or name == "model" or name == "20":
        return "https://nlp1.cs.unc.edu/data/model_LXRT.pth"
    elif name in ["0"+str(i) for i in range(1, 10)] + ["10", "11", "12"]:
        return "https://nlp1.cs.unc.edu/data/github_pretrain/lxmert/Epoch%s_LXRT.pth" % name
    elif name == "vqa":
        return "https://drive.google.com/uc?id=10I7rrxjR0-N9K-9NHVFImo4nh0qliqbQ" # "https://drive.google.com/file/d/10I7rrxjR0-N9K-9NHVFImo4nh0qliqbQ"
    elif name == "gqa":
        return "https://drive.google.com/uc?id=1L2TM9Y_KZsl4x28CJe9H82_P39gGyhTe" # "https://drive.google.com/file/d/1L2TM9Y_KZsl4x28CJe9H82_P39gGyhTe"


def get_lxrt_model(name, max_seq_len=64):
    from types import SimpleNamespace
    args = SimpleNamespace()
    args.llayers = 9
    args.rlayers = 5
    args.xlayers = 5
    args.from_scratch = False

    model = LXRTEncoder(args, max_seq_len)
    model.load(cached_path(model_name_to_url(name)))
    return model


from pprint import pprint
from typing import Dict
from torchvision import models


def group_wise_lr(model, group_lr_conf: Dict, path=""):
    assert path is not None
    path = path.strip()
    assert type(group_lr_conf) == dict

    confs = []
    nms = []
    for k, v in group_lr_conf.items():
        assert hasattr(model, k)
        assert type(k) == str
        assert type(v) == dict
        all_primitives = all([type(vl) == float or type(vl) == int for vl in v.values()])
        all_dicts = all([type(vl) == dict for vl in v.values()])
        assert all_primitives or all_dicts
        if all_primitives:
            attr = getattr(model, k)

            named_params = attr.named_parameters()
            names, params = zip(*named_params)
            conf = dict(params=params, **v)
            names = list(map(lambda n: k + "." + n, names))
            confs.append(conf)
            nms.extend(names)
        else:
            cfs, names = group_wise_lr(getattr(model, k), v, path=path + k + ".")
            names = list(map(lambda n: k + "." + n, names))
            confs.extend(cfs)
            nms.extend(names)

    assert sum([len(d["params"]) for d in confs]) == len(nms)
    if path == "":
        left_out_names, left_out_params = zip(*[(k, p) for k, p in model.named_parameters() if k not in nms])
        model_param_names = set(list(zip(*model.named_parameters()))[0])
        assert set(list(left_out_names) + nms) == model_param_names
        confs.append({"params": left_out_params})
        for c in confs:
            c["params"] = (n for n in c["params"])
        return confs, nms, left_out_names
    else:
        return confs, nms


if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    confs, names, left_out_names = group_wise_lr(model, {"layer4": {"lr": 0.3},
                                                         "layer3": {"0": {"conv2": {"lr": 0.001}},
                                                                    "1": {"lr": 0.003}}})
    print(left_out_names)
    pprint(confs)

confs, names, left_out_names = group_wise_lr(model, {"layer4": {"lr": 0.3},
                                                     "layer3": {"0": {"conv2": {"lr": 0.001}},
                                                                "1": {"lr": 0.003}}})

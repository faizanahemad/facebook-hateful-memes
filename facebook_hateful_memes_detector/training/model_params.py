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
        param_names = []
        assert hasattr(model, k)
        assert type(k) == str
        assert type(v) == dict
        # Separate the primitives and objects
        # Process the objects first
        # Process primitives with remaining names
        for kl, vl in v.items():
            if type(vl) == dict:
                cfs, names = group_wise_lr(getattr(model, k), {kl: vl}, path=path + k + ".")
                confs.extend(cfs)
                param_names.extend(names)

        primitives = {kl: vl for kl, vl in v.items() if type(vl) == float or type(vl) == int}
        attr = getattr(model, k)
        remaining_params = [(k, p) for k, p in attr.named_parameters() if k not in param_names]
        if len(remaining_params) > 0:
            names, params = zip(*remaining_params)
            conf = dict(params=params, **primitives)
            confs.append(conf)
            param_names.extend(names)

        param_names = list(map(lambda n: k + "." + n, param_names))
        nms.extend(param_names)


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
    confs, names, left_out_names = group_wise_lr(model, {"layer4": {"lr": 0.3},
                                                         "layer3": {"0": {"conv2": {"lr": 0.001}},
                                                                    "lr": 0.003}})

    print(left_out_names)
    pprint(confs)


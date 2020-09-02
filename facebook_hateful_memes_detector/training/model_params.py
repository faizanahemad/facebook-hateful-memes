from pprint import pprint
from typing import Dict
from torchvision import models
import torch
from torch import nn


def group_wise_lr(model, group_lr_conf: Dict, path=""):
    """
    Refer https://pytorch.org/docs/master/optim.html#per-parameter-options
    Refer https://discuss.pytorch.org/t/different-learning-rate-for-a-specific-layer/33670/11?u=faizan


    torch.optim.SGD([
        {'params': model.base.parameters()},
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], lr=1e-2, momentum=0.9)


    to


    cfg = {"classifier": {"lr": 1e-3},
           "lr":1e-2, "momentum"=0.9}
    confs, names = group_wise_lr(model, cfg)
    torch.optim.SGD([confs], lr=1e-2, momentum=0.9)



    :param model:
    :param group_lr_conf:
    :return:
    """
    assert type(group_lr_conf) == dict
    _ = group_wise_finetune(model, group_lr_conf)
    confs = []
    nms = []
    for kl, vl in group_lr_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int or type(vl) == bool

        if type(vl) == dict:
            try:
                assert hasattr(model, kl)
            except AssertionError as e:
                print("key = ", kl, "\n", "model = ", model)
            cfs, names = group_wise_lr(getattr(model, kl), vl, path=path + kl + ".")
            confs.extend(cfs)
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_lr_conf.items() if type(vk) == float or type(vk) == int or type(vk) == bool}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms and p.requires_grad]
    if len(remaining_params) > 0:
        names, params = zip(*remaining_params)
        conf = dict(params=params, **primitives)
        confs.append(conf)
        nms.extend(names)

    plen = sum([len(list(c["params"])) for c in confs])
    assert len(list(filter(lambda p: p.requires_grad, model.parameters()))) == plen
    assert set([k for k, p in model.named_parameters() if p.requires_grad]) == set(nms)
    assert plen == len(nms)
    if path == "":
        for c in confs:
            c["params"] = (n for n in c["params"])
    return confs, nms


def group_wise_finetune(model, group_finetune_conf: Dict, path=""):

    assert type(group_finetune_conf) == dict
    nms = []
    for kl, vl in group_finetune_conf.items():
        assert type(kl) == str
        assert type(vl) == dict or type(vl) == float or type(vl) == int or type(vl) == bool

        if type(vl) == dict:
            try:
                assert hasattr(model, kl)
            except AssertionError as e:
                print("key = ",kl, "\n", "model = ", model)
            names = group_wise_finetune(getattr(model, kl), vl, path=path + kl + ".")
            names = list(map(lambda n: kl + "." + n, names))
            nms.extend(names)

    primitives = {kk: vk for kk, vk in group_finetune_conf.items() if type(vk) == bool}
    remaining_params = [(k, p) for k, p in model.named_parameters() if k not in nms]
    if len(remaining_params) > 0 and "finetune" in primitives:
        names, params = zip(*remaining_params)
        nms.extend(names)
        finetune = primitives["finetune"]
        assert type(finetune) == bool
        for p in params:
            p.requires_grad = finetune

    return nms

if __name__ == "__main__":
    model = models.resnet18(pretrained=True)
    print(list(model.children())[0].__class__ == nn.Conv2d)

    test_configs = [
        # Give same Lr to all model params
        {"lr": 0.3},

        # For the below 3 cases, you will need to pass the optimiser overall optimiser params for remaining model params.
        # This is because we did not specify optimiser params for all top-level submodules, so defaults need to be supplied
        # Refer https://pytorch.org/docs/master/optim.html#per-parameter-options

        # Give same Lr to layer4 only
        {"layer4": {"lr": 0.3}},

        # Give one LR to layer4 and another to rest of model. We can do this recursively too.
        {"layer4": {"lr": 0.3},
         "lr": 0.5},

        # Give one LR to layer4.0 and another to rest of layer4
        {"layer4": {"0": {"lr": 0.001},
                    "lr": 0.3}},

        # More examples
        {"layer4": {"lr": 0.3,
                    "0": {"lr": 0.001}}},

        {"layer3": {"0": {"conv2": {"lr": 0.001}},
                    "1": {"lr": 0.003}}},

        {"layer4": {"lr": 0.3},
         "layer3": {"0": {"conv2": {"lr": 0.001}},
                    "lr": 0.003},
         "lr": 0.001},

        {"layer4": {"lr": 0.3},
         "layer3": {"0": {"conv2": {"lr": 0.001}},
                    "1": {"lr": 0.003}}}
    ]

    for cfg in test_configs:
        confs, names = group_wise_lr(model, cfg)
        print("#" * 140)
        pprint(cfg)
        print("-" * 80)
        pprint(confs)
        print("#" * 140)

    # Test Fine Tuning Conf

    test_configs = [
        # Give same Lr to all model params
        {"lr": 0.3},


        {"layer4": {"lr": 0.3, "finetune": True}},

        {"layer3": {"0": {"conv2": {"lr": 0.001}},
                    "1": {"lr": 0.003, "finetune": True}}},


        {"layer4": {"lr": 0.3, "finetune": True},
         "layer3": {"0": {"conv2": {"lr": 0.001}},
                    "1": {"lr": 0.003, "finetune": True},
                    "finetune": False},
         }
    ]

    for cfg in test_configs:
        print("#" * 140)
        for p in model.parameters():
            p.requires_grad = False

        names = group_wise_finetune(model, cfg)
        print(cfg, names)
        print("-" * 80)
        for n, p in model.named_parameters():
            if p.requires_grad:
                print(n)

        print("#" * 140)


import torch
import os
from types import SimpleNamespace
from mmf.common.registry import registry
from mmf.utils.build import build_config, build_trainer
from mmf.utils.configuration import Configuration
from mmf.utils.distributed import distributed_init, infer_init_method
from mmf.utils.env import set_seed, setup_imports
from mmf.utils.flags import flags
from mmf.utils.logger import Logger
import argparse


def get_vilbert(device):
    args = argparse.Namespace(config_override=None)
    args.opts = ['config=projects/hateful_memes/configs/vilbert/from_cc.yaml', 'model=vilbert',
                 'dataset=hateful_memes', 'run_type=val',
                 'checkpoint.resume_zoo=vilbert.finetuned.hateful_memes.from_cc_original', 'evaluation.predict=true']

    configuration = Configuration(args)
    configuration.args = args
    config = configuration.get_config()
    config.start_rank = 0
    config.device_id = 0
    setup_imports()
    configuration.import_user_dir()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    config.training.seed = set_seed(config.training.seed)
    registry.register("seed", config.training.seed)
    print(f"Using seed {config.training.seed}")

    config = build_config(configuration)

    # Logger should be registered after config is registered
    registry.register("writer", Logger(config, name="mmf.train"))
    trainer = build_trainer(config)
    trainer.load()
    trainer.model.to(device)
    return trainer.model


def get_visual_bert(device):
    args = argparse.Namespace(config_override=None)
    args.opts = ['config=projects/hateful_memes/configs/visual_bert/from_coco.yaml', 'model=visual_bert',
                 'dataset=hateful_memes', 'run_type=val',
                 'checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.from_coco', 'evaluation.predict=true']

    configuration = Configuration(args)
    configuration.args = args
    config = configuration.get_config()
    config.start_rank = 0
    config.device_id = 0
    setup_imports()
    configuration.import_user_dir()
    config = configuration.get_config()

    if torch.cuda.is_available():
        torch.cuda.set_device(config.device_id)
        torch.cuda.init()

    config.training.seed = set_seed(config.training.seed)
    registry.register("seed", config.training.seed)
    print(f"Using seed {config.training.seed}")

    config = build_config(configuration)

    # Logger should be registered after config is registered
    registry.register("writer", Logger(config, name="mmf.train"))
    trainer = build_trainer(config)
    trainer.load()
    trainer.model.to(device)
    return trainer.model

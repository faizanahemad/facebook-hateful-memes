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
from transformers.tokenization_auto import AutoTokenizer
from mmf.datasets.processors.bert_processors import BertTokenizer


def get_model(device, opts):
    args = argparse.Namespace(config_override=None)
    args.opts = opts
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

    config = build_config(configuration)

    # Logger should be registered after config is registered
    registry.register("writer", Logger(config, name="mmf.train"))
    trainer = build_trainer(config)
    # trainer.load()
    ready_trainer(trainer)
    trainer.model.to(device)
    return trainer.model


def ready_trainer(trainer):
    from mmf.utils.logger import Logger, TensorboardLogger
    trainer._set_device()
    trainer.run_type = trainer.config.get("run_type", "train")
    writer = registry.get("writer", no_warning=True)
    if writer:
        trainer.writer = writer
    else:
        trainer.writer = Logger(trainer.config)
        registry.register("writer", trainer.writer)

    trainer.config_based_setup()

    trainer.load_model_and_optimizer()


def tokenizer_conf():
    s = SimpleNamespace()
    s.tokenizer_config = SimpleNamespace()
    s.tokenizer_config.type = 'bert-base-uncased'
    s.tokenizer_config.params = {'do_lower_case': True}
    s.max_seq_length = 128
    s.mask_probability = 0.0
    return s


def get_vilbert(device):
    opts = ['config=projects/hateful_memes/configs/vilbert/from_cc.yaml', 'model=vilbert',
            'dataset=hateful_memes', 'run_type=val',
            'checkpoint.resume_zoo=vilbert.finetuned.hateful_memes.from_cc_original', 'evaluation.predict=true']
    return {"model": get_model(device, opts), "tokenizer": BertTokenizer(tokenizer_conf())}


def get_visual_bert(device):
    opts = ['config=projects/hateful_memes/configs/visual_bert/from_coco.yaml', 'model=visual_bert',
            'dataset=hateful_memes', 'run_type=val',
            'checkpoint.resume_zoo=visual_bert.finetuned.hateful_memes.from_coco', 'evaluation.predict=true']
    return {"model": get_model(device, opts), "tokenizer": BertTokenizer(tokenizer_conf())}

from typing import Union, Callable, Tuple

from .generic import *
from ..utils import CNNHead, DecoderEnsemblingHead, MultiLayerTransformerDecoderHead


def fb_1d_loss_builder(n_dims, n_tokens, n_out, dropout, **kwargs):
    loss = kwargs.pop("loss", "classification")
    classification_head = kwargs.pop("classification_head", "cnn1d")
    if classification_head == "cnn1d":
        head = CNNHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
    elif classification_head == "head_ensemble":
        head = DecoderEnsemblingHead(n_dims, n_tokens, n_out, dropout, loss, **kwargs)
    elif classification_head == "decoder_ensemble":
        gaussian_noise = kwargs.pop("gaussian_noise", 0.5)
        n_classifier_layers = kwargs.pop("n_classifier_layers", 3)
        n_classifier_decoders = kwargs.pop("n_classifier_decoders", 2)
        attention_drop_proba = kwargs.pop("attention_drop_proba", 0.0)
        head = MultiLayerTransformerDecoderHead(n_dims, n_tokens, n_out, dropout,
                                                gaussian_noise, attention_drop_proba, loss, n_layers=n_classifier_layers,
                                                n_queries=16, n_decoders=n_classifier_decoders)
    else:
        raise NotImplementedError
    return head


def train_and_predict(model_fn: Union[Callable, Tuple], datadict, batch_size, epochs,
                      accumulation_steps=1, scheduler_init_fn=None,
                      model_call_back=None, validation_epochs=None,
                      sampling_policy=None, class_weights=None,
                      prediction_iters=1, evaluate_in_train_mode=False,
                      consistency_loss_weight=0.0, num_classes=2,
                      aug_1: Callable=identity, aug_2: Callable=identity,
                      show_model_stats=False, give_probas=True):
    train_df = datadict["train"]
    dev_df = datadict["dev"]
    test_df = datadict["test"]
    metadata = datadict["metadata"]
    dataset = convert_dataframe_to_dataset(train_df, metadata, consistency_loss_weight == 0)
    dev_dataset = convert_dataframe_to_dataset(dev_df, metadata, False)
    test_dataset = convert_dataframe_to_dataset(test_df, metadata, False)
    if callable(model_fn):
        model, optimizer = model_fn()
    else:
        model, optimizer = model_fn
    if show_model_stats:
        model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
        params = sum([np.prod(p.size()) for p in model_parameters])
        print("Trainable Params = %s" % (params), "\n", model)
        show_model_stats = not show_model_stats
    validation_strategy = dict(validation_epochs=validation_epochs,
                               train=dict(method=validate, args=[model, batch_size, dataset], kwargs=dict(display_detail=False)),
                               val=dict(method=validate, args=[model, batch_size, dev_dataset], kwargs=dict(display_detail=False,
                                                                                                            prediction_iters=prediction_iters,
                                                                                                            evaluate_in_train_mode=evaluate_in_train_mode)))
    validation_strategy = validation_strategy if validation_epochs is not None else None
    if consistency_loss_weight > 0:
        from torch.utils.data import ConcatDataset
        tmodel = ModelWrapperForConsistency(model, num_classes, consistency_loss_weight)
        train_dataset = LabelConsistencyDatasetWrapper(dataset, ConcatDataset((dev_dataset, test_dataset)), num_classes, aug_1, aug_2)
        collate_fn = label_consistency_collate
    else:
        tmodel = model
        train_dataset = dataset
        collate_fn = my_collate
    tmodel.to(get_device())
    train_losses, learning_rates, validation_stats = train(tmodel, optimizer, scheduler_init_fn, batch_size, epochs, train_dataset,
                                                           model_call_back=model_call_back, validation_strategy=validation_strategy,
                                                           accumulation_steps=accumulation_steps, plot=True,
                                                           sampling_policy=sampling_policy, class_weights=class_weights, collate_fn=collate_fn)
    return predict(model, datadict, batch_size, prediction_iters=prediction_iters, evaluate_in_train_mode=evaluate_in_train_mode, give_probas=give_probas), model, validation_stats


def predict(model, datadict, batch_size, prediction_iters=1, evaluate_in_train_mode=False, give_probas=True):
    metadata = datadict["metadata"]
    test = datadict["test"]
    ids = test["id"] if "id" in test.columns else test["ID"]
    id_name = "id" if "id" in test.columns else "ID"
    test_dataset = convert_dataframe_to_dataset(test, metadata, False)
    proba_list, all_probas_list, predictions_list, labels_list = generate_predictions(model, batch_size, test_dataset, collate_fn=my_collate,
                                                                                      prediction_iters=prediction_iters,
                                                                                      evaluate_in_train_mode=evaluate_in_train_mode,)
    probas = pd.DataFrame({id_name: ids, "proba": proba_list, "label": predictions_list})
    if "submission_format" in datadict and type(datadict["submission_format"]) == pd.DataFrame and len(datadict["submission_format"]) == len(probas):
        submission_format = datadict["submission_format"]
        assert set(submission_format.id) == set(probas.id)
        sf = submission_format.merge(probas.rename(columns={"proba": "p", "label": "l"}), how="inner", on="id")
        sf["proba"] = sf["p"]
        sf["label"] = sf["l"]
        cols = ["id", "proba", "label"] if give_probas else ["id", "label"]
        sf = sf[cols]
    else:
        sf = pd.DataFrame({id_name: ids, "proba": all_probas_list, "label": predictions_list})
    return sf




# The Hateful Memes Challenge README


The Hateful Memes Challenge is a dataset and benchmark created by Facebook AI to drive and measure progress on multimodal reasoning and understanding. The task focuses on detecting hate speech in multimodal memes.


Please see the paper for further details:


[The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes
D. Kiela, H. Firooz, A. Mohan, V. Goswami, A. Singh, P. Ringshia, D. Testuggine](
https://arxiv.org/abs/2005.04790)


# Dataset details
The files for this folder are arranged as follows:


img/                -        the PNG images
train.jsonl        -        the training set
dev.jsonl        -        the development set
test.jsonl        -        the “seen” test set


An additional “unseen” test set will be released at a later date under the NeurIPS 2020 competition. Please see https://ai.facebook.com/hatefulmemes. The competition rules are provided on the competition website.


The .jsonl format contains one JSON-encoded example per line, each of which has the following fields:


‘text’        - the text occurring in the meme
‘img’        - the path to the image in the img/ directory
‘label’        - the label for the meme (0=not-hateful, 1=hateful), provided for train and dev


The metric to use is AUROC. You may also report accuracy in addition, since this is more interpretable. To compute these metrics, we recommend the roc_auc_score and accuracy_score methods in sklearn.metrics, with default settings.


# License
The dataset is licensed under the terms in the `LICENSE.txt` file.


# Image Attribution
If you wish to display example memes in your paper, please provide the following attribution:


*Image is a compilation of assets, including ©Getty Image.*


# Citations
If you wish to cite this work, please use the following BiBTeX:

```
@inproceedings{Kiela2020TheHM,
  title={The Hateful Memes Challenge: Detecting Hate Speech in Multimodal Memes},
  author={Douwe Kiela and Hamed Firooz and Aravind Mohan and Vedanuj Goswami and Amanpreet Singh and Pratik Ringshia and Davide Testuggine},
  year={2020}
}
```


# Contact
If you have any questions or comments on the dataset, please contact hatefulmemeschallenge@fb.com.
# facebook-hateful-memes
Facebook hateful memes challenge using multi-modal learning. More info about it here: https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set

Commands
go into data dir and wget the data file
```bash
unzip -j -P KexZs4tn8hujn1nK data.zip
```

```bash
mkdir img
mv *.png img
```

```bash
conda install -y -c anaconda openjdk
pip install  more-itertools nltk pydot spacy statsmodels tabulate Cython dill flair gensim nltk pydot graphviz scipy pandas seaborn matplotlib bidict torch torchvision transformers fasttext contractions pytorch-nlp spacy-transformers stanza
pip install git+https://github.com/myint/language-check.git 
pip install pycontractions
python -m spacy download en_core_news_sm en_core_news_md en_core_news_lg
pip install allennlp==1.0.0rc4 allennlp-models==1.0.0rc4
pip install stanza
python -c "import nltk;nltk.download('tagsets');nltk.download('punkt');nltk.download('averaged_perceptron_tagger');nltk.download('maxent_ne_chunker');nltk.download('words');import stanza;stanza.download('en');nltk.download('stopwords');nltk.download('vader_lexicon');nltk.download('treebank')"
python -m spacy download en_trf_distilbertbaseuncased_lg
git clone https://github.com/huggingface/torchMoji.git && cd torchMoji && pip install -e . && python scripts/download_weights.py
# edit: vi torchmoji/lstm.py and change `input, batch_sizes, _, _ = input` line 78
# look at: https://github.com/huggingface/torchMoji/blob/master/examples/score_texts_emojis.py
pip install -U maxfw
pip install pytextrank # https://github.com/DerwenAI/pytextrank
pip install git+https://github.com/LIAAD/yake
pip install multi-rake # CFLAGS="-Wno-narrowing" pip install cld2-cffi
pip install textblob
pip install rake-nltk
pip install nlpaug
git lfs install

```

## Build Wikipedia TF-IDF
```bash
git clone https://github.com/marcocor/wikipedia-idf.git && cd wikipedia-idf/src
wget https://dumps.wikimedia.your.org/enwikisource/20200520/enwikisource-20200520-pages-articles.xml.bz2
mkdir enwiki
# https://github.com/marcocor/wikipedia-idf
wget https://raw.githubusercontent.com/attardi/wikiextractor/master/WikiExtractor.py
python WikiExtractor.py -o enwiki --compress --json enwikisource-20200520-pages-articles.xml.bz2
# Change Line 57 of src/wikipediaidf.py: stems, token_to_stem_mapping = stem(tokens) if stemmer else None, None => stems, token_to_stem_mapping = stem(tokens) if stemmer else (None, None)
python wikipediaidf.py -i enwiki/**/*.bz2 -o tfidf -s english -c 64


```

# TODO
- Read Data
- EDA
- Create Baselines on Text only
- Create Baseline on Image Only
- Create Multi-modal baseline

# Approach
- Image augmentation
    - Use some crazy image augmentation networks 
    - http://raywzy.com/Old_Photo/
    - Ricap and cutout
- Text Augmentation (Data Aug by Back translatiob)
- Try 
    - image captioned text as another part of model input
        - For the captioned text use multiple regions of image (Random slices?)
    - use a previous layer of Resnet and feed into attention model as a 3d array (14 x 14 x 256) so that it can look at regions of image
        - For image vectors use both row and column numbers for position embedding
    - ALBERT and Reformer
    - Start with Pretrained Resnet and Transformers
    - Relative position embedding?
    - Analyse the images
        - Check what kind of augmentations will help. 
        - See if adding emotion recognition pre-trained net on images can help
        - See if sentiment identification pre-trained network on images and text can help
        - See if object detection models like YOLOv4 can help
        
    - Take a few base level classifiers made of LSTM/GRU and use their penultimate layer output as a feature 
    - Start with pretrained VLBert as base network, add more attention layers and incorporate side features with those.
    - answer this: where should I look and which detector pipeline (what lens should I use to look) should I use to look at that point
    - This problem is more around image understanding rather than recognition. As such disentangled feature extraction from images can help.
- analyse performance of different networks against adversarial attacks
- Sarcasm detection pretrain

- Pretrain
    - MLM with COCO
    - Is this caption correct with COCO
    - VCR and VQA
    
- Can I find which words /  topics are sensitive from a Knowledge Base like Wikipedia and then use that info along with the word embedding
- Sentiment classification datasets
- Image Captioning Datasets
- Avoid Embedding matrices, use bert base / bert large embeddings of tokens
- Composing Models using self-attention
- show training and prediction time for each model

- Features
    - YOLOv4 objects and their relative sizes as features
    - P(Hateful | Word) for each word using bayes theorem and assumption of independence

- Models: Combine multiple Trained Models as ensembles (Text-only, Image-only, Multi-modal models)
    - Text Models can use back translation augmentation
    - Fasttext pretrained word embeddings + Multi-head attention
    - Bert pretrained word embeddings + Multi-head attention
    - Bidirectional Text GRU model
    - Image Captioning model with self-attention layers
    - VL-Bert pretrained
    - VisualBert pretrained
    - VL-Bert pretrained then fine tuned with Text and image augmentations
    - VisualBert pretrained then fine tuned with text and image augmentations
    - Faster RCNN / YOLOv4 based image features model
    - MMBT by Facebook
    - Plain Resnet based classifier? 
    - Disentangeled VAE with self-attention based model
    
    
- Training
    - Build Robustness into the network and more generalisation capability by planning Adversarial attacks on it.
    - Adversarial NLP inputs like hyphen in between phone numbers or spelling change but pronunciation intact
    
    
- Inference
    - Try multi-inference and then soft vote, by doing NLP and image augmentations
    
    
## Possible Pretrained Models List
    - https://github.com/Holmeyoung/crnn-pytorch
    - https://github.com/OpenNMT/OpenNMT-py
    - https://github.com/CSAILVision/semantic-segmentation-pytorch
    - https://github.com/clovaai/CRAFT-pytorch
    - https://github.com/clovaai/deep-text-recognition-benchmark
    - https://github.com/facebookresearch/detectron2
    - https://github.com/thunlp/ERNIE : Wikipedia Knowledge base
    
## Possible Extra Datasets
    - Sarcasm
        - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
        - https://github.com/MirunaPislar/Sarcasm-Detection
        - https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home
        - https://github.com/ef2020/SarcasmAmazonReviewsCorpus/wiki
        - https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
        - https://github.com/sahilswami96/SarcasmDetection_CodeMixed
        
    - Emoji
    - General NLP: https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html
        
    
## Other References
    - https://gist.github.com/jerheff/8cf06fe1df0695806456
    - LSTM
        - https://stackoverflow.com/questions/53010465/bidirectional-lstm-output-question-in-pytorch
        - https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        - https://discuss.pytorch.org/t/rnn-output-vs-hidden-state-dont-match-up-my-misunderstanding/43280
        

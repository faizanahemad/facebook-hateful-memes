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
conda install python=3.8
pip install --pre torch torchvision -f https://download.pytorch.org/whl/nightly/cu101/torch_nightly.html
pip install fasttext
# pip install torch torchvision 
# pip install torch==1.5.1+cu101 torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html 
# pip install torchvision==0.6.1+cu101 -f https://download.pytorch.org/whl/torch_stable.html
# pip install https://download.pytorch.org/whl/nightly/cu101/torch-1.6.0.dev20200625%2Bcu101-cp37-cp37m-linux_x86_64.whl


pip install transformers==3.0.2  more-itertools nltk pydot spacy statsmodels tabulate Cython dill flair gensim nltk pydot graphviz scipy pandas seaborn matplotlib bidict contractions pytorch-nlp spacy-transformers stanza demjson omegaconf torchtext textblob rake-nltk nlpaug annoy torch_optimizer vaderSentiment maxfw pytextrank pycontractions yacs jsonlines lmdb editdistance fastBPE regex requests sacremoses subword_nmt mosestokenizer diskcache "pytorch-pretrained-bert>=0.6.1" gpustat gputil gdown fvcore opencv-python git+https://github.com/cocodataset/panopticapi.git fcache 'git+https://github.com/faizanahemad/detectron2.git@old' 'git+https://github.com/myint/language-check.git' 'git+https://github.com/LIAAD/yake' 'git+https://github.com/faizanahemad/ImageCaptioning.pytorch.git' 'git+https://github.com/ruotianluo/meshed-memory-transformer.git' nlp imgaug albumentations timm imagecorruptions langdetect datasets




python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
python -m spacy download en_core_web_lg
python -c "import nltk;nltk.download('tagsets');nltk.download('punkt');nltk.download('averaged_perceptron_tagger');nltk.download('maxent_ne_chunker');nltk.download('words');import stanza;stanza.download('en');nltk.download('stopwords');nltk.download('vader_lexicon');nltk.download('treebank');nltk.download('wordnet');import gensim.downloader as api;api.load(\"glove-twitter-25\");api.load(\"glove-twitter-50\");api.load(\"glove-wiki-gigaword-50\");api.load(\"word2vec-google-news-300\");api.load(\"conceptnet-numberbatch-17-06-300\");"


git clone https://github.com/huggingface/torchMoji.git && cd torchMoji && pip install --no-deps -e . && python scripts/download_weights.py
# edit: vi torchmoji/lstm.py and change `input, batch_sizes, _, _ = input` line 78
# look at: https://github.com/huggingface/torchMoji/blob/master/examples/score_texts_emojis.py

# Requires GLIBC 2.18
CFLAGS="-Wno-narrowing" pip install cld2-cffi
pip install multi-rake

# wget -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
git clone https://github.com/facebookresearch/mmf.git && cd mmf && git checkout a79bb9985a3c1347741796d350030ced20043195 && pip install --no-dependencies --editable .
cd ..
git clone https://gitlab.com/vedanuj/vqa-maskrcnn-benchmark.git && cd vqa-maskrcnn-benchmark && python setup.py clean && python setup.py build && python setup.py develop
cd ~
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop

git clone git@github.com:faizanahemad/facebook-hateful-memes.git && cd facebook-hateful-memes && pip install --no-dependencies --editable .
mkdir mygit && cd mygit && git clone https://github.com/faizanahemad/facebook-hateful-memes.git && cd facebook-hateful-memes && pip install --no-dependencies --editable .


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

# Replace wikipediaidf.py with models/external/wikipediaidf.py

python wikipediaidf.py -i enwiki/**/*.bz2 -o tfidf -s english -c 64


```

File Structure inside wiki-idf cloned dir
```
.
├── enwiki
├── LICENSE.md
├── README.md
└── src
    ├── enwiki
    ├── enwikisource-20200520-pages-articles.xml.bz2
    ├── tfidf_stems.csv
    ├── tfidf_terms.csv
    ├── WikiExtractor.py
    └── wikipediaidf.py
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
    - Super-resolution and then resize
- Try 
    - Take a attention or conv based text classifier and overfit, make heatmap of which word is used for classification and invert those words to create negative examples. 
    - Analyse the images
        - Check what kind of augmentations will help. 
        - See if adding emotion recognition pre-trained net on images can help
        - See if sentiment identification pre-trained network on images and text can help
        - See if object detection models like YOLOv4 can help
    - This problem is more around image understanding rather than recognition. As such disentangled feature extraction from images can help.
    - Cluster based OverSampling
        - Cluster-Based Over Sampling: In this case, the K-means clustering algorithm is independently applied to minority and majority class instances. This is to identify clusters in the dataset. Subsequently, each cluster is oversampled such that all clusters of the same class have an equal number of instances and all classes have the same size.  After oversampling of each cluster, all clusters of the same class contain the same number of observations.
    - Ensembling / Stacking
        - https://arxiv.org/pdf/1704.01664.pdf
- analyse performance of different networks against adversarial attacks
- Can I find which words /  topics are sensitive from a Knowledge Base like Wikipedia and then use that info along with the word embedding


- Features
    - YOLOv4 objects and their relative sizes as features

- Models: Combine multiple Trained Models as ensembles (Text-only, Image-only, Multi-modal models)
    - Disentangeled VAE with self-attention based model
    
    
- Training
    - Adversarial Images
    - Adversarial NLP inputs like hyphen in between phone numbers or spelling change but pronunciation intact
    
    
- Inference
    - Try multi-inference and then soft vote, by doing NLP and image augmentations
    
    
## Possible Pretrained Models List
    - https://github.com/facebookresearch/detectron2
    - https://github.com/facebookresearch/mmf
    - https://github.com/open-mmlab/mmdetection
    - https://github.com/TuSimple/simpledet
    - https://github.com/Holmeyoung/crnn-pytorch
    - https://github.com/OpenNMT/OpenNMT-py
    - https://github.com/CSAILVision/semantic-segmentation-pytorch
    - https://github.com/clovaai/CRAFT-pytorch
    - https://github.com/clovaai/deep-text-recognition-benchmark
    - https://github.com/thunlp/ERNIE : Wikipedia Knowledge base
    - https://github.com/jackroos/VL-BERT
    - https://github.com/facebookresearch/detr
    - https://github.com/cadene/pretrained-models.pytorch
    - https://github.com/rwightman/pytorch-image-models/
    
## Possible Extra Datasets
    - Sarcasm
        - https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
        - https://analyticsindiamag.com/10-popular-datasets-for-sentiment-analysis/
        - https://www.kaggle.com/c/sentiment-analysis-on-movie-reviews/data
        - https://github.com/MirunaPislar/Sarcasm-Detection
        - https://www.kaggle.com/rmisra/news-headlines-dataset-for-sarcasm-detection/home
        - https://github.com/ef2020/SarcasmAmazonReviewsCorpus/wiki
        - https://github.com/rishabhmisra/News-Headlines-Dataset-For-Sarcasm-Detection
        - https://github.com/sahilswami96/SarcasmDetection_CodeMixed
        
    - Emoji
    - General NLP: https://pytorchnlp.readthedocs.io/en/latest/source/torchnlp.datasets.html
    - https://visualgenome.org/
    - https://www.datasetlist.com/
    - https://archive.ics.uci.edu/ml/index.php
    - https://www.stateoftheart.ai/
    - https://paperswithcode.com/sota
        
    
## Other References
    - https://gist.github.com/jerheff/8cf06fe1df0695806456
    - https://github.com/umbertogriffo/focal-loss-keras https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    - https://discuss.pytorch.org/t/zero-grad-optimizer-or-net/1887/12
    - https://discuss.pytorch.org/t/how-to-implement-accumulated-gradient/3822/18
    - LSTM
        - https://stackoverflow.com/questions/53010465/bidirectional-lstm-output-question-in-pytorch
        - https://stackoverflow.com/questions/48302810/whats-the-difference-between-hidden-and-output-in-pytorch-lstm
        - https://discuss.pytorch.org/t/rnn-output-vs-hidden-state-dont-match-up-my-misunderstanding/43280
        

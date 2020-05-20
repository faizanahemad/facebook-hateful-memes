# facebook-hateful-memes
Facebook hateful memes challenge using multi-modal learning. More info about it here: https://ai.facebook.com/blog/hateful-memes-challenge-and-data-set

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
        
    - P(Hateful | Word) for each word using bayes theorem and assumption of independence
    - Take a few base level classifiers made of LSTM/GRU and use their penultimate layer output as a feature 
    - Start with pretrained VLBert as base network, add more attention layers and incorporate side features with those.
    - answer this: where should I look and which detector pipeline (what lens should I use to look) should I use to look at that point
    - Build Robustness into the network and more generalisation capability by planning Adversarial attacks on it.
    - Adversarial NLP inputs like hyphen in between phone numbers or spelling change but pronunciation intact
    - This problem is more around image understanding rather than recognition. As such disentangled feature extraction from images can help.
- analyse performance of different networks against adversarial attacks

- Pretrain
    - MLM with COCO
    - Is this caption correct with COCO
    - VCR and VQA
    
- Can I find which words /  topics are sensitive from a Knowledge Base like Wikipedia and then use that info along with the word embedding
- Sentiment classification datasets
- Image Captioning Datasets

#! /bin/sh

# Look here: https://github.com/IBM/MAX-Toxic-Comment-Classifier/blob/master/Dockerfile
wget https://max-cdn.cdn.appdomain.cloud/max-toxic-comment-classifier/1.0.0/assets.tar.gz --output-document=assets/assets.tar.gz
tar -x -C assets/ -f assets/assets.tar.gz -v


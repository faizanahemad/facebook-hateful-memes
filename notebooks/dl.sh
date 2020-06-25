#! /bin/sh

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M-subword.zip
unzip wiki-news-300d-1M-subword.bin.zip

wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.bin.zip
unzip wiki-news-300d-1M-subword.bin.zip

rm -rf crawl-300d-2M-subword.zip
rm -rf wiki-news-300d-1M-subword.bin.zip

# rsync --delete -ravzh /home/ahemf/mygit/facebook-hateful-memes/data ahemf@dev-dsk-ahemf-p3-2-83906738.us-west-2.amazon.com:/home/ahemf/mygit/facebook-hateful-memes

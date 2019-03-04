import urllib.request
import sys

def download():
    print("downloading...")
    url = "http://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz"
    path = "./data/images.tar.gz"
    urllib.request.urlretrieve(url1,filename = path)
    print("finish !!")
    
download()
#!tar -zxvf ./data/annotations.tar.gz

#pwd -> cat
#mkdir -p train/bengal train/birman train/russian_blue
#mkdir -p test/bengal test/birman test/russian_blue

#!cp ../../images/*Bengal_* train/bengal
#!cp ../../images/*Birman_* train/birman
#!cp ../../images/*Russian_Blue_* train/russian_blue

#!find train/bengal -name "Bengal*" | head -20 | xargs -I% mv % test/bengal
#!find train/birman -name "Birman*" | head -20 | xargs -I% mv % test/birman
#!find train/russian_blue -name "Russian_Blue*" | head -20 | xargs -I% mv % test/russian_blue



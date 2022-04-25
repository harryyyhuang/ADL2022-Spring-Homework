if [ ! -d data/ ]; then
    mkdir -p data/
fi

cd data

curl -L https://www.dropbox.com/sh/tlhsslcnb5dt95e/AADSF6gwiYRSrn5lnFb48Dc-a?dl=1 > data.zip

unzip data.zip

rm data.zip


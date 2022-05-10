if [ ! -d data/ ]; then
    mkdir -p data/
fi

cd data

curl -L https://www.dropbox.com/sh/ci5yzr9bhnkaroj/AADLFLFfKSdR_YQc1e4B4oCBa?dl=1 > data.zip

unzip data.zip

rm data.zip


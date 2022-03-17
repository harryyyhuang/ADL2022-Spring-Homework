if [ ! -d ckpt/ ]; then
    mkdir -p ckpt/
fi

cd ckpt

curl -L https://www.dropbox.com/sh/zoo30porsrxktkq/AADT49AcnYDNQNV4l2XnYz_Va?dl=1 > ckpt.zip

unzip ckpt.zip

rm ckpt.zip

cd ..

if [ ! -d cache/ ]; then
    mkdir -p cache/
fi

cd cache

curl -L https://www.dropbox.com/sh/yhniyx1n0jcuekm/AADzEQTgixIfplXuGehNqL92a?dl=1 > cache.zip

unzip cache.zip

rm cache.zip
if [ ! -d ckpt/ ]; then
    mkdir -p ckpt/
fi

cd ckpt

curl -L https://www.dropbox.com/sh/fgmfb3gf91if9lt/AADRIdTv_m_ZlDvb3YV8ad0ta?dl=1 > ckpt.zip

unzip ckpt.zip

rm ckpt.zip


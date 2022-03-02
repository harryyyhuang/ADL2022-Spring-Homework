if [ ! -d ckpt/intent ]; then
    mkdir -p ckpt/intent
fi
if [ ! -d ckpt/slot ]; then
    mkdir -p ckpt/slot
fi

if [ ! -f ckpt/intent/baseline.pt ]; then
    wget https://www.dropbox.com/s/999tobypta099pp/baseline.pt?dl=1 -O ckpt/intent/baseline.pt
fi

if [ ! -f ckpt/slot/baseline.pt ]; then
    wget https://www.dropbox.com/s/b4yagn8l3mqp780/baseline.pt?dl=1 -O ckpt/slot/baseline.pt
fi
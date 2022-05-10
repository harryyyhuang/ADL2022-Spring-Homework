if [ ! -d summary/ ]; then
    mkdir -p summary/
fi

cd summary

curl -L https://www.dropbox.com/sh/j5f6fzeyqzc3ar8/AABSqiO3W2kNS_s-ZniqWiiTa?dl=1 > summary.zip

unzip summary.zip

rm summary.zip


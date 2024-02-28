# Build a simple HMM-GMM from scratch🔥
簡易的隱藏式馬可夫模型(語音辨識)
![](https://i.imgur.com/UYkgLIv.png)

## Installation
```bash
# Step 1 使用 git 下載專案
git clone https://github.com/Amiannn/Sample-HMM-GMM.git
cd Sample-HMM-GMM

# Step 2 使用 Miniconda 建立虛擬 python 環境
conda create --name hmm python=3.7
conda activate hmm

# Step 3 安裝套件
pip3 install -r requirements.txt
```

## Dataset
- [Free Spoken Digit Dataset (FSDD)](https://github.com/Jakobovski/free-spoken-digit-dataset)
- 將FSDD下載之後放入`datasets`的資料夾中

## Train
```bash
python3 train.py
```

## Eval
```bash
python3 eval.py
```

## Reference
- [Digital Speech Processing](http://ocw.aca.ntu.edu.tw/ntu-ocw/ocw/cou/104S204)
- [Isolated-Spoken-Digit-Recognition](https://github.com/SIFANWU/Isolated-Spoken-Digit-Recognition.git)
- [Speech Processing for Machine Learning](https://haythamfayek.com/2016/04/21/speech-processing-for-machine-learning.html)
- [HMM-GMM](https://zhuanlan.zhihu.com/p/258826836)

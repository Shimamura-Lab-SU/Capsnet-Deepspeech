 
# カプセルネットワークを用いた音声認識

## 実行方法
### 従来法(カプセルネットワークなし)

論文[DeepSpeech2](http://arxiv.org/pdf/1512.02595v1.pdf)を元に実装した
[deepspeech.pytorch](https://github.com/SeanNaren/deepspeech.pytorch/blob/master/README.md)を利用する．

deepspeech-pytorch.masterフォルダのtrain.pyを実行することで，訓練を行う．
test.pyを実行することで，テストを行う．

### 提案法(カプセルネットワークあり)
従来法を元に提案法を実装した．
[カプセルネットワーク](https://arxiv.org/pdf/1710.09829.pdf)の実装にあたり，[CapsNet-PyTorch](https://github.com/motokimura/capsnet_pytorch)を利用している．

capsnet-deepspeechフォルダのtrain.pyを実行する．
test.pyを実行することで，テストを行う．

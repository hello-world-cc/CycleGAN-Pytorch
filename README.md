# CycleGAN-Pytorch
<b>paper：</b>https://arxiv.org/abs/1703.10593<br>
<b>official PyTorch implementation：</b>https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix<br>
CycleGAN is one of the most interesting works I have read. Although the idea behind cycleGAN looks quite intuitive after you read the paper: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, the official PyTorch implementation by junyanz is difficult to understand for beginners (The code is really well written but it's just that it has multiple things implemented together). As I am writing a simpler version of the code for some other work, I thought of making my version of cycleGAN public for those who are looking for an easier implementation of the paper.
## Requirements
* The code has been written in Python (3.6.9) and PyTorch (1.4)
## How to run
* To download datasets (eg. horse2zebra)<br>
Note that，the horse2zebra dataset contains some single channel images，so you should delete those single channel images after you download the dataset
```Python
sh ./download_dataset.sh horse2zebra
```
* To run training
```Python
python main.py --mode train
```
* To run testing
```Python
python main.py --mode test
```
## Results
* For horse to zebra dataset. ( Real - Generated )

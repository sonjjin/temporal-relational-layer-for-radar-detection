# Introduction
This repo is reproduces the following paper
Li, Peizhao, et al. "Exploiting temporal relations on radar perception for autonomous driving." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.

It is based on [WACV2021] Oriented Object Detection in Aerial Images with Box Boundary-Aware Vectors
https://github.com/yijingru/BBAVectors-Oriented-Object-Detection

# Dependencies
Ubuntu 18.04, Python 3.7.16, PyTorch 1.10.0, OpenCV-Python 4.7.0

# Install
you have to install Dota_devkit. 

0. go to the Dota_devkit
```
    cd datasets/DOTA_devkit
```
1. install swig
```
    sudo apt-get install swig
```
2. create the c++ extension for python
```
    swig -c++ -python polyiou.i
    python setup.py build_ext --inplace
```


## Remove Images That Do Not Have Objects [Relate to NAN Loss]
For Issue [About Loss NaN](https://github.com/yijingru/BBAVectors-Oriented-Object-Detection/issues/15), @navidasj96 has found that removing images that do not have any objects inside will help resolve the NAN loss issue. 

### About dataset TXT Files
The `train.txt` and `test.txt` contain the list of image names without suffix, example:
```
000000
000001
000002
000003
```
the format of the ground-truth:
Format: `x1, y1, x2, y2, x3, y3, x4, y4, category, difficulty`

Examples: 
```
275.0 463.0 411.0 587.0 312.0 600.0 222.0 532.0 tennis-court 0
341.0 376.0 487.0 487.0 434.0 556.0 287.0 444.0 tennis-court 0
428.0 6.0 519.0 66.0 492.0 108.0 405.0 50.0 bridge 0
```

## Train Model
```ruby
python main_radiate.py --num_epoch 20 --batch_size 32 --train_mode train_good_weather --backbone ResNet34 --device 1 --grad_step 4
```

## Test Model
```ruby
python test_seq.py --backbone ResNet18 --train_mode train_good_and_bad_weather --exp 1 --device 0
```

## Evaluate Model
```ruby
python best_eval.py --backbone ResNet18 --train_mode train_good_and_bad_weather --exp 0 --device 1
```

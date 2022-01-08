# Single Image Deblur using Distortion Prior

# Requirements
python v3.x
torchvision

# Downloads
1. Please download pre-trained weight here: 
https://drive.google.com/file/d/1Ymx1ggYnX2EDxgT010eWiclMXYrwH4Se/view?usp=sharing

2. Please download GoPro dataset and put it to ./dataset folder
https://drive.google.com/file/d/1y4wvPdOG3mojpFCHTqLgriexhbjoWVkK/view?usp=sharing

# Test inference time and PSNR
3. To test inference time and PSNR, please run:
python main.py --mode test --test_model 'path to model.pkl' --data_dir 'path to goPro test folder' 

# Expected Results:
PSNR = 32.02dB
Average inference time on Tesla v100: 13.9ms, on Titan X GPU: 26ms.

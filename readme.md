## Image Registration


I this work we use Deep Learning to perform multi-modal image registration.

Image Registration is the process of finding motion parameters between two images. 
Those parameters will then be used to warp one image to the other.

Image Registration is a very important step in many medical imaging applications.
For example, in the case of PET/CT, the PET image is registered to the CT image.
This is done to align the two images and to be able to perform quantitative analysis
on the PET image.

In this work, we use a Deep Learning approach to perform image registration.
Our goal is to register Infra-Red (IR) images to RGB images.

### Dataset
We generate our dataset using SkyData dataset. we generate 28000 pairs of images for training and 8000 pairs for testing.
we use the following random transformations on the images and annonate the transformation parameters which will be used
during the training phase to evaluate models performance.

## Environment

1. Conda
```bash
curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

2. create environment
```bash
cd ImageRegistration566/
conda env create -f environment.yaml
```

3. training

```bash
cd model/
#you can change the parameters in the train.sh
bash train.sh
```

4. inference

inference is run on the test set, and the results are saved in the results folder model/results folder as a csv file.
```bash
cd model/
#you can provide arguments for which model to use when running test.py 
python test.py 
```
## Results

|  |  mean | std | min | 25% | 50% | 75% | max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | 
| dhn | 151.15789074013813 | 159.1039243375901 | 77.63887023925781 | 136.12919998168948 | 142.96229553222656 | 150.19552993774414 | 9075.8671875 |
| clkn | 54.59398943394345 | 427.6451853616394 | 3.67910361289978 | 21.1573543548584 | 29.344375610351566 | 42.19117736816406 | 24603.240234375 |
| dlkfm | 48.91476940911059 | 667.9669204874064 | 0.1582658439874649 | 0.8468896895647049 | 1.2868676781654358 | 17.729602813720703 | 27323.13671875 |
| mhn | 169.07957261124898 | 112.20111584763956 | 79.58358001708984 | 150.19924545288086 | 160.55944061279297 | 170.4832763671875 | 5155.302734375 |
| **M70** | 145.3518497891957 | **23.03793943951943** | 84.47472381591797 | 138.02655029296875 | 144.6538314819336 | 151.15814208984375 | 1601.10302734375 |
| **M120**  | 145.76519307982787 | **8.750537040499681** | 118.6453857421875 | 139.4173126220703 | 145.77223205566406 | 151.84590911865234 | **173.41989135742188** |
| **M175** | 145.1455664184723 | 8.59215348896344 | 118.56593322753906 | 138.96845245361328 | 145.21278381347656 | 151.22668838500977 | **172.32846069335938** |

The results show that our model is able to register IR images to RGB with reasonable accuracy. other models often overshoot the transformation parameters and the results are not accurate.

## References

[1]...
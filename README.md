# ImageColoration
Exercise of CNN design for coloring black and white images

## How to run
1. Install MiniConda
2. Enter the MiniConda shell `conda activate`
3. Create the environment with `conda env create environment.yml`
4. Run `main.py` and specify number of epochs with `-e` flag \
   i.e. `$ python3 main.py -e 10` will run the the script with 10 epocs

Note: 
- Data is taken from the `dummyData/ColorfulOriginal` folder. Modify this through `dataset` variable in `main.py`

- Output will be dumped into a folder created at runtime `./output/Colored/` and `./output/Gray/`

Developed and tested on `Ubuntu 20.04` using `Python 3.9.7` \
Based around *Colorful Image Colorization, [2016, Zhang, Et.al]* \
https://github.com/richzhang/colorization
# ImageColoration
Exercise of CNN design for estimating color for black and white images

## How to run
1. Install MiniConda
2. Enter the MiniConda shell `conda activate`
3. Create the environment with `conda env create environment.yml`
4. Activate your newly created environment `conda activate elavrImgCol`
5. Run `main.py` from the root folder and specify number of epochs with `-e` flag \
   i.e. `$ python3 main.py -e 10` will run the the script with 10 epocs \
   5.1 To run the framework with the pretrained model, use the following command \
   `$ python3 main.py -e 1 --ts NC_Dataset/Training/ --vs NC_Dataset/Validation/ --pretrained checkpoints/model.pth --skip_training `

Note: 
- By default the data is extracted from the `dummyData/NC_Dataset` folder. Modify this in the `dataset` variable in `main.py`

- Output will be dumped into a folder created at runtime `./output/Colored/` and `./output/Gray/`

## Available command-line flags
| Flag | Type | Usage | Description |
|:---:|---|---|---|
| -h | bool | -h | Help. Prints available command-line flags |
| -e | int | -e 10 | Specifies the number of epochs over which the training is done |
| -b | bool | -b | Stops the training after performing one forward-feed operation. Used only during debugging |
| --ts | str | --ts relative/path/to | Specifies a relative path to the training dataset |
| --vs | str | --vs relative/path/to | Specifies a relative path to the validation dataset |
| --pretrained | str | --pretrained relative/path/to | Specifies a relative path to a pretrained model |
| --skip_training | bool | --skip_training | Skips the training and proceeds straight to evaluation |



## File and Folder Structure
```
├── main.py
├── environment.yml
├── outputs/
├── dummyData/
├── NC_Dataset/
├── model/
└── src/
```
`main.py` Is the main entry point of the software, see above for instructions on how to use it \
`environment.yml` Defines a conda environment \
`outputs/` Is an empty folder that contains locally generated output, i.e. output during training/validation is dumped here \
`dummyData/` Contains dummyData used during development/debugging \
`NC_Dataset/` Contains a randomly split NC_dataset in 90%-10% ratio \
`model/` Contains the pretrained model \
`src/` Short for *source*; location of the source code \


Developed and tested on `Ubuntu 20.04` using `Python 3.9.7` \
Based around *Colorful Image Colorization, [2016, Zhang, Et.al]* \
https://github.com/richzhang/colorization
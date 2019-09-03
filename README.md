## This branch at this moment is compatible with only 2017 Framework. 

# ExoPieSlimmer
A set of py scripts to slim down the big tuples (output of ExoPieElement) 

## To install the setup, go to ExoPieElement/Readme and follow the cmssw installation instructions and then do following: 

git clone git@github.com:ExoPie/ExoPieSlimmer.git 

git clone git@github.com:ExoPie/ExoPieUtils.git

## What changes to make: 

In order to run the skimmer, you have to change the input file path and also setinterative to True. 

isCondor = False

runInteractive = True

inputpath= "/eos/cms/store/group/phys_exotica/bbMET/ExoPieElementTuples/MC_2017miniaodV2_V1/" 


## How to run the skimmer 

#### Intractive 1: 
if you want to run skimmer on txt files of datasets, then use this command:

python SkimTree.py -F -inDir inputdirectory_name_where_txt_files_are_located -runOnTXT -D outputDirectory_name

but you need to keep `isCondor=Flase` and `runInteractive=True`

## some details 

variables.py: variables form input files are listed here

triggers.py: trigger list

outputTree.py: branches to be saved in the tree (output)

filters.py: event filter (met) 

SkimTree.py: main file 

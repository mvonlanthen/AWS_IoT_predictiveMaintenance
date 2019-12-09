'''
downloadDataset.py
------------------
Download, unzip and prepare the "Turbofan engine degradation simulation dataset" 
into one csv file. 

The orignal dataset is splited into 4 batches which are themselves splitted into 
train and test sets. Moreover, the ground truth for the test sets are in 
seperated files. To make the manipulation of the dataset a bit easier, this 
script concatenates all these files into one csv.

More info about this dataset are available here 
"https://data.nasa.gov/dataset/Turbofan-engine-degradation-simulation-data-set/vrks-gjie"

Usage
-----
* download and prepare dataset. The dataset is downloaded at /my_work_dir/turbofanDataset:
      python downloadDataset.py
* print help:
      python downloadDataset.py -h
* download to a specific location. Fails if already exist
      python downloadDataset.py -t /target/directory/for/dataset
* download to a specific location. Overwrite if already exist
      python downloadDataset.py -t /target/directory/for/dataset -o
'''

# import modules
# --------------
import os
import argparse
import requests
import shutil
import zipfile

import pandas as pd


# functions
# ---------
def parseArguments():
    '''
    Parse script input arguments

    Usage
    -----
    args = parseArguments()
    '''
    parser = argparse.ArgumentParser(
        prog=None,
        usage=None,
        description=(
            'Download, unzip and prepare the "Turbofan engine degradation '
            'simulation dataset" into one csv file. more info available '
            'here "https://data.nasa.gov/dataset/Turbofan-engine-degradation-simulation-data-set/vrks-gjie"'
        )
    )
    parser.add_argument(
        '-t','--target-dir',
        help=(
            'download location of the dataset. Default is /working_dir/turbofanDataset. '
            'If already exist, the script fails. Use -o to change this behavior.'
        ),
        type=str,
        default=os.path.join(os.getcwd(),'turbofanDataset')
    )
    parser.add_argument(
        '-o','--overwrite',
        help='overwrite the target dir if already exist.',
        action='store_true',
    )
    return parser.parse_args()


def downloadDataset(targetDir,unzip=True,keepZip=True,overwrite=True):
    '''
    Download the dataset and unzip it.

    Usage
    -----
    downloadDataset(targetDir,unzip=True,keepZip=True)
    '''
    # Check if the target folder exist. Fail or overwrite depending on arguments
    if os.path.isdir(targetDir):
        if overwrite:
            print('remove existing taget directory "{}"'.format(targetDir))
            shutil.rmtree(targetDir)
        else:
            raise OSError('traget directory already existing.')
    os.mkdir(targetDir)

    # download the zip-file
    url = 'https://ti.arc.nasa.gov/m/project/prognostic-repository/CMAPSSData.zip'
    r = requests.get(url=url,verify=False,stream=True)
    r.raw.decode_content = True
    zFilePath = os.path.join(targetDir,'turbofanDataset.zip')
    with open(zFilePath,'wb') as fwb:
        shutil.copyfileobj(r.raw,fwb) 

    # unzip the file
    with zipfile.ZipFile(zFilePath,'r') as zf:
        zf.extractall(targetDir)


def readPrepareDataset(dataDir):
    '''
    Read and prepare the dataset, then return it as one large pandas DataFrame.
    '''
    # The files come without headers. Here are the headers
    columns = [
        'id',  #id of the engine
        'cycle', #time in term of cycle
        # operation settings of the engine during the cycle
        'setting1','setting2','setting3', 
        # 21 sensors recording
        's1','s2','s3','s4', 's5', 's6', 's7', 's8', 's9', 's10', 
        's11','s12', 's13', 's14','s15', 's16', 's17', 's18', 's19', 's20', 
        's21'
    ]

    # operation condition. Each batch (e.g. train_FD001.txt and test_FD001.txt 
    # is one batch) represents a similar fleet of engine running at similar 
    # operating conditions. These operation contidions are defined in the readme.txt 
    # of the dataset. We summarize these conditions here
    opConditions = {
        1:{'conditions':'ONE (Sea Level)','fault_modes':'ONE (HPC Degradation)'},
        2:{'conditions':'SIX',            'fault_modes':'ONE (HPC Degradation)'},
        3:{'conditions':'ONE (Sea Level)','fault_modes':'TWO (HPC Degradation, Fan Degradation)'},
        4:{'conditions':'SIX',            'fault_modes':'TWO (HPC Degradation, Fan Degradation)'},
    }

    ### load the file labelled "train_FD..."
    dfTrain = list()
    engineID_max = 0
    for i in range(1,5):
        # read the file
        dataFile = 'train_FD{:03d}.txt'.format(i)
        print('reading and processing file {}'.format(dataFile))
        df = pd.read_csv(os.path.join(dataDir,dataFile),delimiter=' ',header=None)
        df.drop(df.columns[[26, 27]],axis=1,inplace=True)
        df.columns = columns

        # add the source file and the orginial id
        df['source'] = dataFile
        df['org_id'] = df['id']

        # add operation conditions
        for k,v in opConditions[i].items():
            df[k] = v

        # the engine id (column 'id') restart at 1 in each file. So when we assemble 
        # the files we have a problem. let's fix that
        df['id'] = df['id']+engineID_max
        engineID_max = df.iloc[-1]['id']
        # print('   min engine ID: {}'.format(df.iloc[0]['id']))
        # print('   max engine ID: {}'.format(df.iloc[-1]['id']))


        # append
        dfTrain.append(df)
        
    # concatenate
    dfTrain = pd.concat(dfTrain)
    dfTrain.reset_index(drop=True,inplace=True)

    # add the Remaining Useful Life (RUL) as a new feature
    rul = pd.DataFrame(dfTrain.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    dfTrain = dfTrain.merge(rul, on=['id'], how='left')
    dfTrain['RUL'] = dfTrain['max'] - dfTrain['cycle']
    dfTrain.drop('max', axis=1, inplace=True)

    # add the failure column
    dfTrain['fail'] = 0
    dfTrain.loc[dfTrain['RUL']==0,'fail'] = 1

    ### load the file labelled "test_FD..."
    dfTests = list()
    engineID_max = 0
    for i in range(1,5):
        dataFile = 'test_FD{:03d}.txt'.format(i)
        print('reading and processing file {}'.format(dataFile))
        df = pd.read_csv(os.path.join(dataDir,dataFile),delimiter=' ',header=None)
        df.drop(df.columns[[26, 27]],axis=1,inplace=True)
        df.columns = columns
        
        # add the source file and the orginial id
        df['source'] = dataFile
        df['org_id'] = df['id']

        # add operation conditions
        for k,v in opConditions[i].items():
            df[k] = v
        
        # Load the ground truth RUL values
        rulFile = 'RUL_FD{:03d}.txt'.format(i)
        dfRul = pd.read_csv(os.path.join(dataDir,rulFile),delimiter=' ',header=None)    
        dfRul.drop(dfRul.columns[1],axis=1,inplace=True)
        dfRul.index += 1
        dfRul.columns = ['RUL_end']
        
        # Merge RUL and timeseries and compute RUL per timestamp
        df = df.merge(dfRul,left_on=df.columns[0], right_index=True, how='left')
        rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
        rul.columns = ['id','max']
        df = df.merge(rul,on=['id'],how='left') # We get the number of cycles per series
        df['RUL'] = df['max']+df['RUL_end']-df['cycle'] # The RUL is the number of cycles per series + RUL - how many cycles have already ran
        df.drop(['max','RUL_end'], axis=1, inplace=True)
        
        # the engine id (column 'id') restart at 1 in each file. So when we assemble 
        # the files we have a problem. let's fix that
        df['id'] = df['id']+engineID_max
        engineID_max = df.iloc[-1]['id']
        # print('   min engine ID: {}'.format(df.iloc[0]['id']))
        # print('   max engine ID: {}'.format(df.iloc[-1]['id']))

        # append
        dfTests.append(df)

    # concatenate
    dfTest = pd.concat(dfTests)
    dfTest.reset_index(drop=True,inplace=True)

    # add the failure column
    dfTest['fail'] = 0
    dfTest.loc[dfTest['RUL']==0,'fail'] = 1

    # concatenate dfTrain and dfTest, then return
    trainMaxId = dfTrain['id'].max()
    dfTest['id'] = dfTest['id']+trainMaxId
    return pd.concat([dfTrain,dfTest])


def writeDataset(df,targetFile,shuffle=True):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df.to_csv(targetFile,sep=',',index=False)


def main(targetDir,overwrite):
    # download and unzip
    downloadDataset(targetDir=targetDir,unzip=True,keepZip=True,overwrite=overwrite)

    # prepare dataset
    df = readPrepareDataset(dataDir=targetDir)

    # shuffle and write dataset
    targetFile = os.path.join(targetDir,'assembledDataset.csv')
    print('writing assembled dataset to "{}"'.format(targetFile))
    writeDataset(df,targetFile,shuffle=True)
    


# script
# ------
if __name__=='__main__':
    # read input arguments
    args = parseArguments()

    # main function
    main(targetDir=args.target_dir,overwrite=args.overwrite)


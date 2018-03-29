import shutil
import glob
import csv
import numpy as np
import time
import os
import sys
import pandas as pd

def zScoreScale(inputs):
    meanData = np.mean(inputs);
    stdData = np.std(inputs, ddof=1);
    scores = [round(((x - meanData) / stdData), 6) for x in inputs]
    
    return scores;



'''
    Softmax Scaling
    Calculate the sigmoid function for feature scaling with inputs (array)
    # Citation: Same above
    Ex: 
        # Apply Sigmoid in list
        sigmoid_inputs = [1, 2, 3, 4]
        print("Sigmoid Function: {}".format(sigmoid(sigmoid_inputs)))
    Parameters: 
        inputs: single row of list
'''
def sigmoidScale(inputs):
    scores = [round(1 / float(1 + np.exp(- x)), 6) for x in inputs]

    return scores;


'''
    Linear Scaling
    This technique normalizes data in range [0, 1].
    linear scaling to [0,1]
    Citation  : Same above
    Parameters: 
        inputs: single row of list
'''
def linearScale(inputs):
    minData = np.min(inputs);
    maxData = np.max(inputs);
    scores = [round((x-minData)/(maxData-minData), 6) for x in inputs]

    return scores;


'''
    Min-Max Normalization
    This technique normalize data in range [-1, 1].
    linear scaling to [-1,1]
    Citation  : Same above
    Parameters: 
        inputs: single row of list
'''
def minMaxScale(inputs):
    min_new = -1;
    max_new = 1;
    minData = np.min(inputs);
    maxData = np.max(inputs);
    scores = [round((((x-minData)/(maxData-minData))*(max_new-min_new)+min_new), 6) for x in inputs]

    return scores;

def readPSSMFile(inputFilePath, SeqName=False, SelfStore=False):    
    # A technque to read PSSM profiles using panda
    # Start index is in col 2 to get all content only.
    # if set to 1 means select with their protein amino acid in first col.
    startCol = 2; 
    if SeqName == True:
        startCol = 1; 
    # 22 amino acid matrix, 42 means with their frequence
    endCol = 22; 
    # Set some of unuseful rows by number of row 
    setSkipRows = (0,1,2);
    # Read data
    pssmData = pd.read_csv(inputFilePath, 
                            skiprows=setSkipRows,
                            # White space delimater ignores more spaces
                            delim_whitespace=True,
                            # No header
                            header=None,
                            usecols=range(startCol, endCol)
                            )
    # Remove last 5 rows
    maxRow = (len(pssmData.index)) - 6 
    # Select only content of PSSM profiles
    pssmData = pssmData.loc[0:maxRow, :]

    if SelfStore == True:
        self.pssm = pssmData;
        
    else:    
        return pssmData;

input_pssm = sys.argv[1]
output_data = sys.argv[2]
Scale = None
COLUMN_NAMES=['Data']
COLLECT = pd.DataFrame(columns=COLUMN_NAMES)

start_time = time.time()
aminoAcid = ['A','R','N','D','C', 'Q','E','G','H','I','L','K','M','F','P','S','T','W','Y','V']
outputTXT= ''
print('generate PSSM feature, please wait ...')
# reader = csv.reader(open(input_pssm), delimiter = '\t')
pssmOptimize  = readPSSMFile(input_pssm,SeqName=True)
# print(pssmOptimize.head())
# data = []
# for row in reader:
#     for col in row:
#         rgb = col.split()
#         data += [[str(x) for x in rgb]]
# pssmOptimize = data[2:len(data)-5]
# sequenceLength = int(pssmOptimize[len(pssmOptimize)-1][0])
sequenceLength = len(pssmOptimize.index)

result = []
for aa in aminoAcid:
#     for row in pssmOptimize:
#         if(row[1] == char):
#             result.append([round(float(x)/sequenceLength,6) for x in row[2:22]])
#         else:
#             result.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])

#     result1.append(np.sum(result, axis = 0))
# result2 = np.array(result1, dtype=object)
# for item in result2.flatten():
#     outputTXT += str(item) + ','
# outputTXT += '0'  + '\n'
    getData = pssmOptimize[pssmOptimize.iloc[:,0].isin(list([aa]))]
    # print("getData : {}".format(getData))
    if getData.empty:
        result.append([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])					
    else:
        #print(aa,'. Not empty!');
        # Sum the same amino acid data and divided by sequence lenght
        getNew = (getData.iloc[:,1:21].sum()/sequenceLength).tolist();
        # print("get new = {}".format(getNew))
        # Scale, Round and store
        if Scale == "sigmoid scale":
            result.append(sigmoidScale(getNew))
            
        elif Scale == "linear scale":
            result.append(linearScale(getNew))
            
        elif Scale == "min-max scale":
            result.append(minMaxScale(getNew))
            
        elif Scale == "z-score normalization":
            result.append(zScoreScale(getNew))
            
        else:
            result.append([round(n, 6) for n in getNew])
            # print("result : {}".format(result))
            
# convert to numpy array (list object) and than flattening the data
newDataFormat = np.array(result, dtype=object)
# print(newDataFormat)
# Insert to data frame
COLLECT.loc[0] = [newDataFormat.flatten()]
# increase
final_data = pd.DataFrame(COLLECT['Data'].values.tolist())
# print(final_data)
final_data.to_csv(output_data, encoding='utf-8', sep=',',index=False, header=False)
print("--- generating finished ---")
print("--- %s seconds ---" % (time.time() - start_time))


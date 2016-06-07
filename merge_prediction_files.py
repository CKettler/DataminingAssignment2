import pandas as pd
import glob

path =r'data/pred_files' # use your path
allFiles = glob.glob(path + "/*.csv")


print allFiles
frame = pd.DataFrame()
list_ = []
for file_ in allFiles:
    df = pd.read_csv(file_,index_col=None, header=0)
    list_.append(df)
frame = pd.concat(list_)
frame.to_csv('data/prediction_file.csv', index=False)
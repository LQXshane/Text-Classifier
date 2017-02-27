from preprocess import relabel
import sys
import pandas as pd
from IPython import embed

if '__main__':

    file_in = sys.argv[1]
    file_out = sys.argv[2]

    # file_in = "../../Crwdworkers/Trump/Categories_Trump.csv"
    # # file_sar = "../../Sarcasm/FINAL_sarcasm.csv"

    positive_labels = ['3,3,1','3,2,2',  #as border 2
                       '2,5,0', '2,4,1','2,3,2',
                       '1,6,0', '1,5,1', '1,4,2',
                       '1,3,3', '0,7,0', '0,6,1', '0,5,2', '0,4,3']

    negative_labels = ['7,0,0', '6,1,0', '5,2,0', '5,1,1',
                       '4,3,0', '4,2,1'] #as border 3


    '''
    below: no GT
    '''
    # positive_labels = ['3,3,1', '3,2,2', '2,3,2', '1,3,3']
    #
    # negative_labels = ['7,0,0', '0,7,0', '6,1,0', '0,6,1', '1,6,0',
    #                    '5,2,0', '2,5,0', '0,5,2', '5,1,1', '1,5,1',
    #                    '4,3,0', '0,4,3', '4,2,1', '2,4,1', '1,4,2']

    data = relabel(file_in, positive_labels, negative_labels)
    # sarcasm = pd.read_csv(file_sar)
    #
    # sarcasm.columns = ['contents',
    #                  'Bernie',
    #                  'Trump',
    #                  'Clinton',
    #                  'Cruz',
    #                  'Num of Candidate',
    #                  'Sarcasm_s1',
    #                  'Sarcasm_s2',
    #                  'Sarcasm_s3',
    #                  'Sarcasm_s4',
    #                  'Sarcasm_s5',
    #                  'Sarcasm_s6',
    #                  'Sarcasm_s7',
    #                  'Number of Sarc',
    #                  'Short and link']
    #
    # res = pd.merge(data, sarcasm, how='left', on=['contents'])

    print(len(data))
    # print (positive_labels, negative_labels)
    print("Percentage of positive samples: %0.2f"%(float(len(data.label[data.label=='1']))*100/len(data.label)))

    # embed()

    data.to_csv(file_out, index=False)

    # embed()
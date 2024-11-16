import pandas as pd
import numpy as np
import os.path as op
from scipy.stats import chi2, rankdata


"""     !!!  SELECT DATASETS FOR COMPARISON  !!!     """

data1 = pd.read_csv(r"Z:\Shared\bryjalab\users\Branislav\Summer2024\Collagen Migration Assay DATA\data 9-8-24!\position_3\analysed\Track_stats.csv")
data2 = pd.read_csv(r"Z:\Shared\bryjalab\users\Branislav\Summer2024\Collagen Migration Assay DATA\data 9-8-24!\position_6\analysed\Track_stats.csv")


"""     !!!  SELECT SAVE PATH   !!!     """

save_path = r"Z:\Shared\bryjalab\users\Branislav\Summer2024\Collagen Migration Assay DATA\data 9-8-24!"


"""     !!!  SELECT COMPARED POSITIONS !!!     """

positionA = 'position3'
positionB = 'position6'


"""               RUN :)                """






directions1 = data1['MEAN_DIRECTION_RAD']
directions2 = data2['MEAN_DIRECTION_RAD']

def mardia_watson_wheeler_test(directions1, directions2):
    def calculate_mardia_watson_wheeler_test(directions1, directions2):
        # Combine the data
        combined_data = np.concatenate((directions1, directions2))
        n1 = len(directions1)
        n2 = len(directions2)
        N = n1 + n2
        
        # Calculate the ranks of the combined data
        ranks = rankdata(rankdata(combined_data)) + 1
        
        # Separate the ranks into the original datasets
        ranks1 = ranks[:n1]
        ranks2 = ranks[n1:]
        
        # Calculate the sum of ranks for each sample
        R1 = np.sum(np.sin(2 * np.pi * ranks1 / N))
        R2 = np.sum(np.cos(2 * np.pi * ranks1 / N))
        S1 = np.sum(np.sin(2 * np.pi * ranks2 / N))
        S2 = np.sum(np.cos(2 * np.pi * ranks2 / N))
        
        # Compute the test statistic
        W = (2 / N) * (R1**2 + R2**2 + S1**2 + S2**2)
        
        # Determine the degrees of freedom (df)
        df = 2
        
        # Compute the p-value
        p_value = chi2.sf(W, df)
        
        return W, p_value
    W, p_value = calculate_mardia_watson_wheeler_test(directions1, directions2)

    with open((op.join(save_path, f'Mardia_Watson_Wheeler_test({positionA}-{positionB}).txt')), 'w') as file:
        try:
            print(f'Test Statistic (W): {W:.3f}, P-Value: {p_value:.3e}')
            file.write(f'Test Statistic (W): {W:.3f}, P-Value: {p_value:.3e}')
            if p_value < 0.05:
                print('Difference between the distributions is significant - H0 rejected.')
                file.write(f'\n    Difference between the distributions is significant - H0 rejected.')
            else:
                print('Difference between the distributions is not significant - failed to reject H0.')
                file.write(f'\n    Difference between the distributions is not significant - failed to reject H0.')
        finally:
            file.close()
mardia_watson_wheeler_test(directions1, directions2)


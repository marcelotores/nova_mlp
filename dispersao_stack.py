# importing the required library
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import ut

# initialize list elements
a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17,a18,a19,a20,a21,a22,a23,a24,classe
0.00234148759345,2.484356589,396172138.406,13184.3097039,0.533112719953,1.0,51724.5129095,2.26348078086,2.77132864597,0.00115404970126,0.591052858553,0.588400990655,1.0,1.0,0.575196926811,112.209716643,0.00586893339572,52731.8212198,12873885.5732,3244547962.24,1.1757859734,0.00389183850907,13183.210839,5.78451996503e-10,c1_p1
0.0017612463819,3.57921097038,547622633.699,15717.3902495,0.482912651321,1.0,61734.8578461,2.30358288427,2.88598703609,0.00101015859196,0.655669947095,0.559400297869,1.0,1.0,0.535587818571,122.730858612,0.00413639880613,62856.5988948,16667976.6742,4549368771.07,1.4108370606,0.00924710051938,15714.2137405,8.33375426413e-10,c1_p1
0.00138247125144,11.1400148653,453716163.474,17196.3659849,0.352436908577,1.0,67619.4390025,2.24454047573,3.05410669331,0.00066332953423,0.859475949385,0.429789504039,1.0,1.0,0.429124544956,129.192630132,0.00369170961006,68770.2761819,18860866.8231,5329815737.43,2.40158108246,0.00416053097226,17190.7933439,2.59381598778e-09,c1_p1

# Create the pandas DataFrame with column name is provided explicitly
df = pd.DataFrame(data, columns=['c1', 'c2', 'c3', 'c4', 'c5', 'c6'])
print(df)
exit()
# read a csv file
df = pd.read_csv('tips.csv')
dataSet = ut.im_data(4, True)
print(dataSet)
exit()
# scatter plot with regression
# line(by default)
sns.lmplot(x=1, y=2, data=dataSet)

# Show the plot
plt.show()
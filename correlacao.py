#from ydata_profiling import ProfileReport
import ut
import seaborn as sb
import numpy as np
import matplotlib.pyplot as mp


data_df = ut.im_data(10, data_frame=True)
print(data_df)

#SC
data_df = data_df.iloc[:, :24]

mask = np.triu(np.ones_like(data_df.corr()))
#dataplot = sb.heatmap(data_df.corr(), cmap="YlGnBu", annot=True, mask=mask)

#tr s anotações
dataplot = sb.heatmap(data_df.corr(), cmap="YlGnBu", mask=mask)

#q
#dataplot = sb.heatmap(data_df.corr(), cmap="YlGnBu")
mp.show()

#profile = ProfileReport(data_df)

#profile

#profile.to_file("your_report.html")
#Airflow libraries
from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.models import Variable
# Native libraries
from urllib.request import urlretrieve
import matplotlib.pyplot as plt
import csv
import pandas as pd
import numpy as np
# Native libraries
import os
import math
# Essential Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Preprocessing
from sklearn.preprocessing import MinMaxScaler
# Algorithms
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from sklearn.cluster import KMeans
from tslearn.clustering import KernelKMeans
from sklearn.decomposition import PCA
from scipy import interpolate
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score
import heapq


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

project_cfg = {
    'owner': 'airflow',
    'email': ['cey.akbulut@gmail.com'],
    'email_on_failure': False,
    'start_date': datetime(2022, 10, 29),
    'retries': 1,
    'retry_delay': timedelta(hours=1),
}

dag = DAG('New_pipeline',
          default_args=project_cfg,
          schedule_interval=timedelta(days=1))


# Fetching environment variable
dag_config = Variable.get("variables_config", deserialize_json=True)
data_path1 = dag_config["data_path1"]
data_path2 = dag_config["data_path2"]
dfcomplt = dag_config["dfcomplete"]
silkmsdtw = dag_config["silhscore_kmsdtw"]


# Checking if Data is availabe
def data_is_available(_file_name1=data_path1,_file_name2=data_path2, **kwargs):
    dataset1 = pd.read_csv(_file_name1)
    dataset2 = pd.read_csv(_file_name2)
    if dataset1.empty:
        print("No Data Fetched")
    else:
        print("{} records have been fetched".format(dataset1.shape[0]))
    return "{} records have been fetched".format(dataset1.shape[0])

def preprocessing(_file_name1=data_path1,_file_name2=data_path2, **kwargs):
    metadata=pd.read_csv(_file_name2, index_col=0, header=0)  
    data=pd.read_csv(_file_name1, index_col=0, header=0).T
    data.index.rename('File', inplace=True)
    data=np.sqrt(data)
    
    md=metadata[metadata['Max severity']>=6]
    a=md['Patient Id'].value_counts()
    b=pd.DataFrame(a[a>=3])
    c=b.index.to_list() 
    new=md[md['Patient Id'].isin(c)].reset_index()
    neww=new[(new['Secondary DNI / DNR']==0) & (new['DNI / DNR']==0)& (new['WHO grade']>=4)]
    newsimple=neww.drop(columns=['Will worsen in the future', 'DNI / DNR',
       'Secondary DNI / DNR', 'Age', 'BMI', 'Charlson index', 'SOFA score',
       'Horowitz index', 'FiO2'])
    pd.options.mode.chained_assignment = None
    DayRatiodf=pd.DataFrame(columns=['File', 'Patient Id', 'Time point - days since admission',
       'Time point - days until outcome', 'WHO grade', 'Max severity',
       'Patient died','Days_ratio'])  
    for i in newsimple['Patient Id'].unique():
       trying=newsimple[newsimple['Patient Id']==i]
       trying.sort_values(by=['Time point - days since admission'], inplace=True)
       checkpoint=trying[trying['WHO grade']>=6]
       if not checkpoint.empty:
          while trying['WHO grade'].iloc[0]<6:
                trying=trying.iloc[1: , :]
          trying['Days_ratio']=(trying['Time point - days since admission']
                      -trying['Time point - days since admission'].iloc[0])
          DayRatiodf=DayRatiodf.append(trying,True)
          
    ##Drops Days_ratio with NaNs
    DayRatiodf.dropna(subset=['Days_ratio'],how='any',inplace=True)
    #Drops days until outcome with NaNs
    DayRatiodf.dropna(subset=['Time point - days until outcome'],how='any',inplace=True)
    ################################################################################
    
    #Statistically important genes according to the paper
    Genes=['AHSG','A2M','PLG','F2','SERPINA4','SERPINC1','F12','C1QB','C1QC','F11','PF4','LPA','C5','TTR','VWF',
       'F5','HPX','TFRC','KLKB1','C4B','F10','B2M','SHBG','C1QA','SERPING1']
    
    NewDatalogChange=pd.DataFrame(columns=['File', 'Patient Id', 'Time point - days since admission',
        'Time point - days until outcome', 'WHO grade', 'Max severity',
        'Patient died','Days_ratio',*Genes])
        
    for i in DayRatiodf['Patient Id'].unique():
       trying=DayRatiodf[DayRatiodf['Patient Id']==i]
       trying.sort_values(by=['Days_ratio'], inplace=True)
       for gene in Genes:
          a=data.loc[trying['File'],[gene]]
          trying=trying.join(a,on='File',how='inner')
          trying[gene]=trying[gene]-trying[gene].iloc[0]
       NewDatalogChange=NewDatalogChange.append(trying,True)

    dictPatientsInterpolate={}
    for patient in NewDatalogChange['Patient Id'].unique():
       d=NewDatalogChange[NewDatalogChange['Patient Id']==patient]
       if d['Patient Id'].value_counts().iloc[0]>2:
          x=d['Days_ratio']
          xnew = np.arange(d['Days_ratio'].iloc[0], d['Days_ratio'].iloc[-1]+1, 1)
          patientdf=pd.DataFrame(xnew, columns=['days'])
          patientdf['Patient Id']=patient
          patientdf['Patient died']=d['Patient died'].unique()[0]
           #interpolation of WHO grade                   
          y=d['WHO grade']
          tck = interpolate.splrep(x, y, s=0, k=2)
          ynew = interpolate.splev(xnew, tck, der=0)
          patientdf['WHO grade']=np.round(ynew,decimals=0).astype('int64')
            #interpolation of genes              
          for gene in Genes:
              y=d[gene]
              tck = interpolate.splrep(x, y, s=0, k=2)
              ynew = interpolate.splev(xnew, tck, der=0)
              patientdf[gene]=ynew 
       dictPatientsInterpolate[patient]=patientdf
     
    lower=-1
    upper=6
    dfsurvived=pd.DataFrame()
    dfdied=pd.DataFrame()    
    for key in dictPatientsInterpolate.keys():
        dfpatient=dictPatientsInterpolate[key]
        if dfpatient['days'].iloc[0]<=(lower+1) and dfpatient['days'].iloc[-1]>=(upper-1):
           dfpatientnew=dfpatient[(dfpatient['days']>lower) & (dfpatient['days']<upper)]
           if dfpatientnew['Patient died'].unique()[0]==0:
              dfsurvived=dfsurvived.append(dfpatientnew)
           else:
              dfdied=dfdied.append(dfpatientnew)          
    dfsurvived.drop_duplicates(subset=['days','Patient Id'], inplace=True)     
         
    dfcomplete=dfdied.append(dfsurvived)
    dfcomplete_gng=dfdied.append(dfsurvived)
    dfcomplete.set_index('days',inplace=True)
    dfcomplete.dropna(how='any',inplace=True,axis=1)
    dfcomplete.to_csv(dfcomplt)

def kmeansdtw(_file=dfcomplt, **kwargs):
    df= pd.read_csv(_file)
    
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
    
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        km_sdtw = TimeSeriesKMeans(n_clusters=i, metric="softdtw", max_iter=100,
                           max_iter_barycenter=5,
                           metric_params={"gamma": .5},
                           random_state=0).fit(mySeries)

        inertia.append(km_sdtw.inertia_)
        ch_index.append(calinski_harabasz_score(sklearn_X,km_sdtw.labels_)) 
        db_index.append(davies_bouldin_score(sklearn_X,km_sdtw.labels_))
        silh_score.append(silhouette_score(sklearn_X,km_sdtw.labels_))
        
    
    km_sdtw = TimeSeriesKMeans(n_clusters=5, metric="softdtw", max_iter=100,
                           max_iter_barycenter=5,
                           metric_params={"gamma": .5},
                           random_state=0).fit(mySeries)
   
    df_subset = df[["days","Patient Id", "C5"]]
    
    mySeries1=[]
    patient_list= df_subset["Patient Id"].unique()
    for pat in range(len(df_subset["Patient Id"].unique())+1):
       df_new1 = df_subset.loc[df_subset["Patient Id"] == patient_list[pat-1]]
       del df_new1["Patient Id"]
       df_new1.set_index("days",inplace=True)
       df_new1.sort_index(inplace=True)
       mySeries1.append(df_new1)  
    del mySeries1[0]
    
    fig, axs = plt.subplots(1,1,figsize=(15,15))
    fig.suptitle('Soft DTW - Clusters')
    a=['black','yellow','red','aqua','orange','blue','magenta','saddlebrown','darkviolet','fuchsia']
    labels= km_sdtw.labels_
    for label in set(labels):
        for i in range(len(labels)):
            axs.plot(mySeries1[i],c=a[labels[i]],alpha=0.4,linewidth=2.0)  
               
    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/kmeansdtw.png')
    plt.close(fig)                       
            
    
def kmeansdba(_file=dfcomplt, **kwargs):
    df = pd.read_csv(_file)
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
    
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        km_dba = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=100,
                           max_iter_barycenter=5,
                           random_state=0).fit(mySeries)
        inertia.append(km_dba.inertia_)
        ch_index.append(calinski_harabasz_score(sklearn_X,km_dba.labels_)) 
        db_index.append(davies_bouldin_score(sklearn_X,km_dba.labels_))
        silh_score.append(silhouette_score(sklearn_X,km_dba.labels_))
    
    km_dba = TimeSeriesKMeans(n_clusters=2, metric="dtw", max_iter=100,
                           max_iter_barycenter=5,
                           random_state=0).fit(mySeries)
                           
    df_subset = df[["days","Patient Id", "C5"]]
    
    mySeries1=[]
    patient_list= df_subset["Patient Id"].unique()
    for pat in range(len(df_subset["Patient Id"].unique())+1):
       df_new1 = df_subset.loc[df_subset["Patient Id"] == patient_list[pat-1]]
       del df_new1["Patient Id"]
       df_new1.set_index("days",inplace=True)
       df_new1.sort_index(inplace=True)
       mySeries1.append(df_new1)  
    del mySeries1[0]
                   
    fig, axs = plt.subplots(1,1,figsize=(15,15))
    fig.suptitle('DTW - Clusters')
    a=['black','yellow','red','aqua','orange','blue','magenta','saddlebrown','darkviolet','fuchsia']
    labels= km_dba.labels_
    for label in set(labels):
       for i in range(len(labels)):
            axs.plot(mySeries1[i],c=a[labels[i]],alpha=0.4,linewidth=2.0)        
            
    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/kmeansdba.png')
    plt.close(fig)                       

def kernelkm(_file=dfcomplt, **kwargs):
    df = pd.read_csv(_file)
    
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
    
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
      seed = 0
      np.random.seed(seed)
      gak_km = KernelKMeans(n_clusters=i,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
      gak_km.fit_predict(mySeries)
  
      inertia.append(gak_km.inertia_)
      ch_index.append(calinski_harabasz_score(sklearn_X,gak_km.labels_)) 
      db_index.append(davies_bouldin_score(sklearn_X,gak_km.labels_))
      silh_score.append(silhouette_score(sklearn_X,gak_km.labels_))
     
    
    seed = 0
    np.random.seed(seed)
    gak_km = KernelKMeans(n_clusters=4,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
    gak_km.fit_predict(mySeries)
    
    df_subset = df[["days","Patient Id", "C5"]]
    
    mySeries1=[]
    patient_list= df_subset["Patient Id"].unique()
    for pat in range(len(df_subset["Patient Id"].unique())+1):
       df_new1 = df_subset.loc[df_subset["Patient Id"] == patient_list[pat-1]]
       del df_new1["Patient Id"]
       df_new1.set_index("days",inplace=True)
       df_new1.sort_index(inplace=True)
       mySeries1.append(df_new1)  
    del mySeries1[0]
                           
    fig, axs = plt.subplots(1,1,figsize=(15,15))
    fig.suptitle('Kernel K-Means1 Clusters')
    a=['black','yellow','red','aqua','orange','blue','magenta','saddlebrown','darkviolet','fuchsia']
    labels= gak_km.labels_
    for label in set(labels):
        for i in range(len(labels)):
            axs.plot(mySeries1[i],c=a[labels[i]],alpha=0.4,linewidth=2.0)       
    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/kernelkmeans.png')
    plt.close(fig)   

def evaluation_kmeansdtw(_file=dfcomplt, **kwargs):
    df= pd.read_csv(_file)
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
    
    rows = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        km_sdtw = TimeSeriesKMeans(n_clusters=i, metric="softdtw", max_iter=100,
                           max_iter_barycenter=5,
                           metric_params={"gamma": .5},
                           random_state=0).fit(mySeries)

        inertia.append(km_sdtw.inertia_)
        ch_index.append(calinski_harabasz_score(sklearn_X,km_sdtw.labels_)) 
        db_index.append(davies_bouldin_score(sklearn_X,km_sdtw.labels_))
        silh_score.append(silhouette_score(sklearn_X,km_sdtw.labels_))
    
    figure, axis = plt.subplots(2,2,figsize=(15,15))    
    #Plot Inertia for different number of clusters 
    ymin = min(inertia)
    ymin_index = inertia.index(ymin)
    xmin = rows[ymin_index]
    axis[0, 0].plot(rows, inertia, marker="o", c='black')
    axis[0, 0].plot(xmin, ymin, 'ro',markersize=18)
    axis[0, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 0].set_ylabel("Value",fontsize = 14)
    axis[0, 0].set_title("Inertia for different number of clusters -KMeansSDTW ",fontsize = 14, fontweight="bold")

    #Plot Calinski-Harabasz Index for different number of clusters   
    ymax = max(ch_index)
    ymax_index = ch_index.index(ymax)
    xmax = rows[ymax_index]
    axis[0, 1].plot(rows,ch_index, marker="o", c='black')
    axis[0, 1].plot(xmax, ymax, 'ro',markersize=18)
    axis[0, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 1].set_ylabel("Value",fontsize = 14)
    axis[0, 1].set_title("Calinski-Harabasz Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Davies-Bouldin Index for different number of clusters  
    ymin2 = min(db_index)
    ymin_index2 = db_index.index(ymin2)
    xmin2 = rows[ymin_index2]
    axis[1, 0].plot(rows,db_index, marker="o", c='black')
    axis[1, 0].plot(xmin2, ymin2, 'ro',markersize=18)
    axis[1, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 0].set_ylabel("Value",fontsize = 14)
    axis[1, 0].set_title("Davies-Bouldin Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Silhouette Score for different number of clusters 
    ymax4 = max(silh_score)
    ymax_index4 = silh_score.index(ymax4)
    xmax4 = rows[ymax_index4]
    axis[1, 1].plot(rows,silh_score, marker="o", c='black')
    axis[1, 1].plot(xmax4, ymax4, 'ro',markersize=18)
    axis[1, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 1].set_ylabel("Value",fontsize = 14)
    axis[1, 1].set_title("Silhouette Score for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/evaluation_kmeansdtw.png')
    plt.close(figure) 
              
def evaluation_kmeansdba(_file=dfcomplt, **kwargs):
    df= pd.read_csv(_file)
    listPatients=[]
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
   
    rows = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        km_dba = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=100,
                           max_iter_barycenter=5,
                           random_state=0).fit(mySeries)
        inertia.append(km_dba.inertia_)
        ch_index.append(calinski_harabasz_score(sklearn_X,km_dba.labels_)) 
        db_index.append(davies_bouldin_score(sklearn_X,km_dba.labels_))
        silh_score.append(silhouette_score(sklearn_X,km_dba.labels_))
    
    figure, axis = plt.subplots(2,2,figsize=(15,15))    
    #Plot Inertia for different number of clusters 
    ymin = min(inertia)
    ymin_index = inertia.index(ymin)
    xmin = rows[ymin_index]
    axis[0, 0].plot(rows, inertia, marker="o", c='black')
    axis[0, 0].plot(xmin, ymin, 'ro',markersize=18)
    axis[0, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 0].set_ylabel("Value",fontsize = 14)
    axis[0, 0].set_title("Inertia for different number of clusters -KMeansSDTW ",fontsize = 14, fontweight="bold")

    #Plot Calinski-Harabasz Index for different number of clusters   
    ymax = max(ch_index)
    ymax_index = ch_index.index(ymax)
    xmax = rows[ymax_index]
    axis[0, 1].plot(rows,ch_index, marker="o", c='black')
    axis[0, 1].plot(xmax, ymax, 'ro',markersize=18)
    axis[0, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 1].set_ylabel("Value",fontsize = 14)
    axis[0, 1].set_title("Calinski-Harabasz Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Davies-Bouldin Index for different number of clusters  
    ymin2 = min(db_index)
    ymin_index2 = db_index.index(ymin2)
    xmin2 = rows[ymin_index2]
    axis[1, 0].plot(rows,db_index, marker="o", c='black')
    axis[1, 0].plot(xmin2, ymin2, 'ro',markersize=18)
    axis[1, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 0].set_ylabel("Value",fontsize = 14)
    axis[1, 0].set_title("Davies-Bouldin Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Silhouette Score for different number of clusters 
    ymax4 = max(silh_score)
    ymax_index4 = silh_score.index(ymax4)
    xmax4 = rows[ymax_index4]
    axis[1, 1].plot(rows,silh_score, marker="o", c='black')
    axis[1, 1].plot(xmax4, ymax4, 'ro',markersize=18)
    axis[1, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 1].set_ylabel("Value",fontsize = 14)
    axis[1, 1].set_title("Silhouette Score for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/evaluation_kmeansdba.png')
    plt.close(figure) 

def evaluation_kernelkm(_file=dfcomplt, **kwargs):
    df = pd.read_csv(_file)
    listPatients=[]
    mySeries=[]
    pat=1
    patient_list=[]
    patient_list= df["Patient Id"].unique()
    for pat in range(len(df["Patient Id"].unique())+1):
       df_new = df.loc[df["Patient Id"] == patient_list[pat-1]]
       del df_new["Patient Id"]
       df_new.set_index("days",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
    del mySeries[0]
    
    mySeries=np.array(mySeries)
    n, sz, d = mySeries.shape
    sklearn_X = mySeries.reshape((n, -1))
    
    rows = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    ch_index=[]
    db_index=[]
    inertia=[]
    silh_score = []
    for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
        km_dba = TimeSeriesKMeans(n_clusters=i, metric="dtw", max_iter=100,
                           max_iter_barycenter=5,
                           random_state=0).fit(mySeries)
        inertia.append(km_dba.inertia_)
        ch_index.append(calinski_harabasz_score(sklearn_X,km_dba.labels_)) 
        db_index.append(davies_bouldin_score(sklearn_X,km_dba.labels_))
        silh_score.append(silhouette_score(sklearn_X,km_dba.labels_))
    
    figure, axis = plt.subplots(2,2,figsize=(15,15))    
    #Plot Inertia for different number of clusters 
    ymin = min(inertia)
    ymin_index = inertia.index(ymin)
    xmin = rows[ymin_index]
    axis[0, 0].plot(rows, inertia, marker="o", c='black')
    axis[0, 0].plot(xmin, ymin, 'ro',markersize=18)
    axis[0, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 0].set_ylabel("Value",fontsize = 14)
    axis[0, 0].set_title("Inertia for different number of clusters -KMeansSDTW ",fontsize = 14, fontweight="bold")

    #Plot Calinski-Harabasz Index for different number of clusters   
    ymax = max(ch_index)
    ymax_index = ch_index.index(ymax)
    xmax = rows[ymax_index]
    axis[0, 1].plot(rows,ch_index, marker="o", c='black')
    axis[0, 1].plot(xmax, ymax, 'ro',markersize=18)
    axis[0, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[0, 1].set_ylabel("Value",fontsize = 14)
    axis[0, 1].set_title("Calinski-Harabasz Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Davies-Bouldin Index for different number of clusters  
    ymin2 = min(db_index)
    ymin_index2 = db_index.index(ymin2)
    xmin2 = rows[ymin_index2]
    axis[1, 0].plot(rows,db_index, marker="o", c='black')
    axis[1, 0].plot(xmin2, ymin2, 'ro',markersize=18)
    axis[1, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 0].set_ylabel("Value",fontsize = 14)
    axis[1, 0].set_title("Davies-Bouldin Index for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    #Plot Silhouette Score for different number of clusters 
    ymax4 = max(silh_score)
    ymax_index4 = silh_score.index(ymax4)
    xmax4 = rows[ymax_index4]
    axis[1, 1].plot(rows,silh_score, marker="o", c='black')
    axis[1, 1].plot(xmax4, ymax4, 'ro',markersize=18)
    axis[1, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
    axis[1, 1].set_ylabel("Value",fontsize = 14)
    axis[1, 1].set_title("Silhouette Score for different number of clusters -KMeansSDTW",fontsize = 14, fontweight="bold")

    plt.savefig('/home/ceyda/airflow/ML_Pipeline2/Results/evaluation_kernelkm.png')
    plt.close(figure) 
                
start = DummyOperator(
    task_id="start",
    dag=dag,
)


task_1 = PythonOperator(
    task_id='data_is_available',
    provide_context=True,
    python_callable=data_is_available,
    dag=dag,
)


task_2 = PythonOperator(
    task_id='preprocessing',
    provide_context=True,
    python_callable=preprocessing,
    dag=dag,
)


task_3_1 = PythonOperator(
    task_id='kmeansdtw',
    provide_context=True,
    python_callable=kmeansdtw,
    dag=dag,
)


task_3_2 = PythonOperator(
    task_id='kmeansdba',
    provide_context=True,
    python_callable=kmeansdba,
    dag=dag,
)


task_3_3 = PythonOperator(
    task_id='kernelkm',
    provide_context=True,
    python_callable=kernelkm,
    dag=dag,
)


evaluation = DummyOperator(
    task_id="evaluation",
    dag=dag,
)


task_4_1 = PythonOperator(
    task_id='evaluation_kmeansdtw',
    provide_context=True,
    python_callable=evaluation_kmeansdtw,
    dag=dag,
)


task_4_2 = PythonOperator(
    task_id='evaluation_kmeansdba',
    provide_context=True,
    python_callable=evaluation_kmeansdba,
    dag=dag,
)


task_4_3 = PythonOperator(
    task_id='evaluation_kernelkm',
    provide_context=True,
    python_callable=evaluation_kmeansdba,
    dag=dag,
)


finish = DummyOperator(
    task_id="finish",
    dag=dag,
)

start >> task_1 >> task_2 >> [task_3_1, task_3_2, task_3_3] >> evaluation >> [task_4_1, task_4_2, task_4_3] >> finish

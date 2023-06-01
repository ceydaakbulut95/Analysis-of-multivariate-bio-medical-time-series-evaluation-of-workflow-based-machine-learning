from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import os
import os.path
from sklearn import preprocessing
from airflow.models import Variable
import math
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.clustering import TimeSeriesKMeans
from tslearn.clustering import KShape
from tslearn.clustering import KernelKMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance
from sklearn import cluster
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics.cluster import rand_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics.cluster import adjusted_rand_score
from pyts.classification import BOSSVS
from pyts.datasets import load_basic_motions
from pyts.multivariate.classification import MultivariateClassifier
from yellowbrick.cluster import SilhouetteVisualizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA # for PCA calculation


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


project_cfg = {
    'owner': 'airflow',
    'email': ['cey.akbulut@gmail.com'],
    'email_on_failure': False,
    'start_date': datetime(2022, 9, 29),
    'retries': 1,
    'retry_delay': timedelta(hours=1),
}

dag = DAG('ML-pipeline',
          default_args=project_cfg,
          schedule_interval=timedelta(days=1))


# Fetching environment variable
dag_config = Variable.get("variables_config", deserialize_json=True)
data_path = dag_config["data_path"]
x_scaled = dag_config["x_scaled"]
df_static_cols = dag_config["df_static_cols"]
myseries_array = dag_config["myseries_array"]
kmsdtwresult = dag_config["kmsdtwresult"]
scalingbeforecomb = dag_config["scalingbeforecomb"]
kmdbaresult = dag_config["kmdbaresult"]
kmkernelresult = dag_config["kmkernelresult"]
bossclassresult = dag_config["bossclassresult"]


# Checking if Data is availabe
def data_is_available(_file_name=data_path, **kwargs):
    dataset = pd.read_csv(_file_name)
    if dataset.empty:
        print("No Data Fetched")
    else:
        print("{} records have been fetched".format(dataset.shape[0]))
    return "{} records have been fetched".format(dataset.shape[0])


def scaling(_file_name=data_path, **kwargs):
   df = pd.read_csv(_file_name)
   df_minmax = df.iloc[:,2:18]
   df_static= df.iloc[:,0:2]
   x = df_minmax.values #returns a numpy array
   min_max_scaler = preprocessing.MinMaxScaler()
   x_sc= min_max_scaler.fit_transform(x)
   x_sc= pd.DataFrame(x_sc)
   df = pd.merge(df_static,x_sc, left_index=True, right_index=True)
   df.to_csv(x_scaled)
   x_sc.to_csv(scalingbeforecomb)
   df_static.to_csv(df_static_cols)

def km_sdtw_result(_file_name=x_scaled,**kwargs):
   df = pd.read_csv(_file_name)
   df = df.iloc[:,1:18]
   mySeries=[]
   namesofMySeries=[]
   pat=1
   for pat in range(len(df["Patient_ID"].unique())+1):
       df_new = df.loc[df['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]
   
   km_sdtw = TimeSeriesKMeans(n_clusters=4, metric="softdtw", max_iter=100,
                           max_iter_barycenter=5,
                           metric_params={"gamma": .5},
                           random_state=0).fit(mySeries)  
   km_sdtw.cluster_centers_.shape
   pd.DataFrame(km_sdtw.labels_).to_csv(kmsdtwresult)
   
   mySeries=np.array(mySeries)
   n, sz, d = mySeries.shape
   sklearn_X = mySeries.reshape((n, -1))
   pd.DataFrame(sklearn_X).to_csv(myseries_array)
   
def km_dba_result(_file_name=x_scaled,**kwargs):
   df = pd.read_csv(_file_name)
   df = df.iloc[:,1:18]
   mySeries=[]
   namesofMySeries=[]
   pat=1
   for pat in range(len(df["Patient_ID"].unique())+1):
       df_new = df.loc[df['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]
   
   km_dba = TimeSeriesKMeans(n_clusters=4, metric="dtw", max_iter=100,
                           max_iter_barycenter=5,
                           random_state=0).fit(mySeries)
   km_dba.cluster_centers_.shape
   pd.DataFrame(km_dba.labels_).to_csv(kmdbaresult)            
   
def km_kernel_result(_file_name=x_scaled,**kwargs):
   df = pd.read_csv(_file_name)
   df = df.iloc[:,1:18]
   mySeries=[]
   namesofMySeries=[]
   pat=1
   for pat in range(len(df["Patient_ID"].unique())+1):
       df_new = df.loc[df['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]
   
   seed = 0
   np.random.seed(seed)
   gak_km = KernelKMeans(n_clusters=4,
                      kernel="gak",
                      kernel_params={"sigma": "auto"},
                      n_init=20,
                      verbose=True,
                      random_state=seed)
   gak_km.fit_predict(mySeries)
   pd.DataFrame(gak_km.labels_).to_csv(kmkernelresult)             

'''
def boss_classification_result(_file_name=x_scaled,**kwargs):
   df = pd.read_csv(_file_name)
   df = df.iloc[:,1:18]
   mySeries=[]
   namesofMySeries=[]
   pat=1
   for pat in range(len(df["Patient_ID"].unique())+1):
       df_new = df.loc[df['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]

   mySeries_train = mySeries[:]
   mySeries_test = mySeries[:]
  
   del mySeries_train[3]
   del mySeries_train[4]
   del mySeries_train[7]
   del mySeries_train[11]
   del mySeries_test[0]
   del mySeries_test[0] 
   del mySeries_test[0]
   del mySeries_test[1]
   del mySeries_test[2]
   del mySeries_test[2]
   del mySeries_test[2]
   del mySeries_test[3]
   del mySeries_test[3]
   del mySeries_test[3]
   del mySeries_test[3]
   del mySeries_test[3]

   y_train =[0,0,0,1,1,1,2,2,2,3,3,3]
   y_test = [0,1,2,3]
   clf = MultivariateClassifier(BOSSVS())
   clf.fit(mySeries_train, y_train)
   clf.predict(mySeries_test)
   print(clf.score(mySeries_test, y_test))
   pd.DataFrame(clf.predict(mySeries_test)).to_csv(bossclassresult)             
'''

def evaluation_with_scores(_file_name=x_scaled,**kwargs):
   df = pd.read_csv(_file_name)
   df = df.iloc[:,1:18]
   mySeries=[]
   namesofMySeries=[]
   pat=1
   for pat in range(len(df["Patient_ID"].unique())+1):
       df_new = df.loc[df['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]

   mySeries = np.array(mySeries)
   n, sz, d = mySeries.shape
   sklearn_X = mySeries.reshape((n, -1))
  
   rows = [2, 3, 4, 5, 6, 7, 8, 9, 10]
   ch_index=[]
   db_index=[]
   inertia=[]
   rand_scr=[]
   adjusted_rand_scr=[]
   silh_score = []

   for i in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
     true_labels=[0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3]
     #Create KMeans instance for different number of clusters
     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=100, random_state=30)
     km.fit(sklearn_X)
     q, mod = divmod(i, 3)

     #Create SilhouetteVisualizer instance with KMeans instance
     #Fit the visualizer
     fig, ax = plt.subplots(3, 3, figsize=(15,8))
     visualizer = SilhouetteVisualizer(km, colors='yellowbrick',ax=ax[q-1][mod])
     visualizer.fit(sklearn_X)
     score = silhouette_score(sklearn_X, km.labels_, metric='euclidean') 

     inertia.append(km.inertia_)
     ch_index.append(calinski_harabasz_score(sklearn_X,km.labels_)) 
     db_index.append(davies_bouldin_score(sklearn_X,km.labels_))
     rand_scr.append(rand_score(km.labels_,true_labels))
     adjusted_rand_scr.append(adjusted_rand_score(km.labels_,true_labels))
     silh_score.append(silhouette_score(sklearn_X,km.labels_))
      #-----------------------------------------------------------------------------------------
      #Plots
      #-----------------------------------------------------------------------------------------
     
   figure, axis = plt.subplots(3,2,figsize=(25,25))    
   #Plot Inertia for different number of clusters 
   ymin = min(inertia)
   ymin_index = inertia.index(ymin)
   xmin = rows[ymin_index]
   axis[0, 0].plot(rows, inertia, marker="o", c='black')
   axis[0, 0].plot(xmin, ymin, 'ro',markersize=18)
   axis[0, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[0, 0].set_ylabel("Value",fontsize = 14)
   axis[0, 0].set_title("Inertia for different number of clusters ",fontsize = 14, fontweight="bold")

  #Plot Calinski-Harabasz Index for different number of clusters   
   ymax = max(ch_index)
   ymax_index = ch_index.index(ymax)
   xmax = rows[ymax_index]
   axis[1, 0].plot(rows,ch_index, marker="o", c='black')
   axis[1, 0].plot(xmax, ymax, 'ro',markersize=18)
   axis[1, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[1, 0].set_ylabel("Value",fontsize = 14)
   axis[1, 0].set_title("Calinski-Harabasz Index for different number of clusters",fontsize = 14, fontweight="bold")

  #Plot Davies-Bouldin Index for different number of clusters  
   ymin2 = min(db_index)
   ymin_index2 = db_index.index(ymin2)
   xmin2 = rows[ymin_index2]
   axis[0, 1].plot(rows,db_index, marker="o", c='black')
   axis[0, 1].plot(xmin2, ymin2, 'ro',markersize=18)
   axis[0, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[0, 1].set_ylabel("Value",fontsize = 14)
   axis[0, 1].set_title("Davies-Bouldin Index for different number of clusters ",fontsize = 14, fontweight="bold")

  #Plot Rand Score for different number of clusters  
   ymax2 = max(rand_scr)
   ymax_index2 = rand_scr.index(ymax2)
   xmax2 = rows[ymax_index2]
   axis[1, 1].plot(rows,rand_scr, marker="o", c='black')
   axis[1, 1].plot(xmax2, ymax2, 'ro',markersize=18)
   axis[1, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[1, 1].set_ylabel("Value",fontsize = 14)
   axis[1, 1].set_title("Rand Score for different number of clusters ",fontsize = 14, fontweight="bold")

  #Plot Adjusted Rand Score for different number of clusters 
   ymax3 = max(adjusted_rand_scr)
   ymax_index3 = adjusted_rand_scr.index(ymax3)
   xmax3 = rows[ymax_index3]
   axis[2, 0].plot(rows,adjusted_rand_scr, marker="o", c='black')
   axis[2, 0].plot(xmax3, ymax3, 'ro',markersize=18)
   axis[2, 0].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[2, 0].set_ylabel("Value",fontsize = 14)
   axis[2, 0].set_title("Adjusted Rand Score for different number of clusters ",fontsize = 14, fontweight="bold")

  #Plot Silhouette Score for different number of clusters 
   ymax4 = max(silh_score)
   ymax_index4 = silh_score.index(ymax4)
   xmax4 = rows[ymax_index4]
   axis[2, 1].plot(rows,silh_score, marker="o", c='black')
   axis[2, 1].plot(xmax4, ymax4, 'ro',markersize=18)
   axis[2, 1].set_xlabel("Number of clusters",c='black',fontsize = 14)
   axis[2, 1].set_ylabel("Value",fontsize = 14)
   axis[2, 1].set_title("Silhouette Score for different number of clusters ",fontsize = 14, fontweight="bold")
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/evaluation.png')
   plt.close(fig)
   


def evaluation_with_pca(_file_name=data_path,_file_name_2=kmsdtwresult,_file_name_3=kmdbaresult,_file_name_4=kmkernelresult, **kwargs):
   df = pd.read_csv(_file_name)
   km_sdtw_labels = pd.read_csv(_file_name_2)
   km_sdtw_labels = km_sdtw_labels.iloc[0:16,1]
   km_dtw_labels = pd.read_csv(_file_name_3)
   km_dtw_labels = km_dtw_labels.iloc[0:16,1]
   km_kernel_labels = pd.read_csv(_file_name_4)
   km_kernel_labels = km_kernel_labels.iloc[0:16,1]
   df_minmax = df.iloc[:,2:18]
   df_static_cols = df.iloc[:,0:2]
   x = df_minmax.values #returns a numpy array
   min_max_scaler = preprocessing.MinMaxScaler()
   x_scaled = min_max_scaler.fit_transform(x)   
  
   plt.rcParams["figure.figsize"] = [7.50, 3.50]
   plt.rcParams["figure.autolayout"] = True  
   fig1 = plt.figure("Figure 1")
   pca = PCA()
   X_pca = pca.fit(x_scaled)
   plt.plot(np.cumsum(pca.explained_variance_ratio_))
   plt.xlabel('number of components')
   plt.ylabel('cumulative explained variance');
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/PCA_cumulative_explained_variance.png')
 
   fig2 = plt.figure("Figure 2")
   Principal_components=pca.fit_transform(x_scaled)
   PC_values = np.arange(pca.n_components_) + 1
   plt.plot(PC_values, pca.explained_variance_, 'ro-', linewidth=2)
   plt.title('Scree Plot')
   plt.xlabel('Principal Component')
   plt.ylabel('Eigenvalues')
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/PCA_eigenvalues.png')
  
   num_components = 1
   pca = PCA(num_components)  
   X_pca = pca.fit_transform(x_scaled) # fit and reduce dimension
   X_pca_df = pd.DataFrame(X_pca)
   pca_ds = pd.merge(df_static_cols, X_pca_df, left_index=True, right_index=True)
 
   mySeries=[]
   namesofMySeries=[] 
   pat=1
   for pat in range(len(pca_ds["Patient_ID"].unique())+1):
       df_new = pca_ds.loc[pca_ds['Patient_ID'] == pat]
       del df_new['Patient_ID']
       df_new.set_index("Date",inplace=True)
       df_new.sort_index(inplace=True)
       mySeries.append(df_new)  
       namesofMySeries.append(pat)
   del mySeries[0]

   fig, axs = plt.subplots(1,1,figsize=(15,15))
   fig.suptitle('Soft DTW - Clusters')
   a = ['black','yellow','red','aqua','orange','blue','magenta','saddlebrown','darkviolet','fuchsia']
   labels_sdtw=km_sdtw_labels.values.tolist()
   labels_dtw=km_dtw_labels.values.tolist()
   labels_kern=km_kernel_labels.values.tolist()

   for label in set(labels_sdtw):
     for i in range(len(labels_sdtw)):
            axs.plot(mySeries[i],c=a[labels_sdtw[i]],alpha=0.4,linewidth=2.0)     
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/Soft_DTW-Clusters.png')
 

   fig, axs = plt.subplots(1,1,figsize=(15,15))
   fig.suptitle('DTW - Clusters')
   for label in set(labels_dtw):
     for i in range(len(labels_dtw)):
            axs.plot(mySeries[i],c=a[labels_dtw[i]],alpha=0.4,linewidth=2.0)      
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/DTW-Clusters.png') 
   
   fig, axs = plt.subplots(1,1,figsize=(15,15))
   fig.suptitle('Kernel K-Means Clusters')
   for label in set(labels_kern):
     for i in range(len(labels_kern)):
            axs.plot(mySeries[i],c=a[labels_kern[i]],alpha=0.4,linewidth=2.0)   
   plt.savefig('/home/ceyda/airflow/ML_Pipeline/Results/Kernel-KMeans-Clusters.png')  
 
   
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


task_2 = DummyOperator(
    task_id="preprocessing",
    dag=dag,
)


task_3 = PythonOperator(
    task_id='scaling',
    provide_context=True,
    python_callable=scaling,
    dag=dag,
)


task_4_1 = PythonOperator(
    task_id='km_sdtw_result',
    provide_context=True,
    python_callable=km_sdtw_result,
    dag=dag,
)

task_4_2 = PythonOperator(
    task_id='km_dba_result',
    provide_context=True,
    python_callable=km_dba_result,
    dag=dag,
)


task_4_3 = PythonOperator(
    task_id='km_kernel_result',
    provide_context=True,
    python_callable=km_kernel_result,
    dag=dag,
)

'''
task_4_4 = PythonOperator(
    task_id='boss_classification_result',
    provide_context=True,
    python_callable=boss_classification_result,
    dag=dag,
)

'''
task_5 = PythonOperator(
    task_id='evaluation_with_scores',
    provide_context=True,
    python_callable=evaluation_with_scores,
    dag=dag,
)

task_6 = PythonOperator(
    task_id='evaluation_with_pca',
    provide_context=True,
    python_callable=evaluation_with_pca,
    dag=dag,
)

finish = DummyOperator(
    task_id="finish",
    dag=dag,
)

start >> task_1 >> task_2 >> task_3 >> [task_4_1, task_4_2, task_4_3] >> task_5 >> task_6 >> finish

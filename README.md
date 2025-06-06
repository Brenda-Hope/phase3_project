# phase3_project
Import libraries
import pandas as pd 
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
Data loading
df = pd.read_csv(r"C:\Users\HP\Documents\PHASE3\project_3\bigml_59c28831336c6604c800002a.csv")
df
state	account length	area code	phone number	international plan	voice mail plan	number vmail messages	total day minutes	total day calls	total day charge	...	total eve calls	total eve charge	total night minutes	total night calls	total night charge	total intl minutes	total intl calls	total intl charge	customer service calls	churn
0	KS	128	415	382-4657	no	yes	25	265.1	110	45.07	...	99	16.78	244.7	91	11.01	10.0	3	2.70	1	False
1	OH	107	415	371-7191	no	yes	26	161.6	123	27.47	...	103	16.62	254.4	103	11.45	13.7	3	3.70	1	False
2	NJ	137	415	358-1921	no	no	0	243.4	114	41.38	...	110	10.30	162.6	104	7.32	12.2	5	3.29	0	False
3	OH	84	408	375-9999	yes	no	0	299.4	71	50.90	...	88	5.26	196.9	89	8.86	6.6	7	1.78	2	False
4	OK	75	415	330-6626	yes	no	0	166.7	113	28.34	...	122	12.61	186.9	121	8.41	10.1	3	2.73	3	False
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3328	AZ	192	415	414-4276	no	yes	36	156.2	77	26.55	...	126	18.32	279.1	83	12.56	9.9	6	2.67	2	False
3329	WV	68	415	370-3271	no	no	0	231.1	57	39.29	...	55	13.04	191.3	123	8.61	9.6	4	2.59	3	False
3330	RI	28	510	328-8230	no	no	0	180.8	109	30.74	...	58	24.55	191.9	91	8.64	14.1	6	3.81	2	False
3331	CT	184	510	364-6381	yes	no	0	213.8	105	36.35	...	84	13.57	139.2	137	6.26	5.0	10	1.35	2	False
3332	TN	74	415	400-4344	no	yes	25	234.4	113	39.85	...	82	22.60	241.4	77	10.86	13.7	4	3.70	0	False
3333 rows × 21 columns

​
print(df.info())
print(df.describe())
print(df.isnull().sum())
​
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3333 entries, 0 to 3332
Data columns (total 21 columns):
 #   Column                  Non-Null Count  Dtype  
---  ------                  --------------  -----  
 0   state                   3333 non-null   object 
 1   account length          3333 non-null   int64  
 2   area code               3333 non-null   int64  
 3   phone number            3333 non-null   object 
 4   international plan      3333 non-null   object 
 5   voice mail plan         3333 non-null   object 
 6   number vmail messages   3333 non-null   int64  
 7   total day minutes       3333 non-null   float64
 8   total day calls         3333 non-null   int64  
 9   total day charge        3333 non-null   float64
 10  total eve minutes       3333 non-null   float64
 11  total eve calls         3333 non-null   int64  
 12  total eve charge        3333 non-null   float64
 13  total night minutes     3333 non-null   float64
 14  total night calls       3333 non-null   int64  
 15  total night charge      3333 non-null   float64
 16  total intl minutes      3333 non-null   float64
 17  total intl calls        3333 non-null   int64  
 18  total intl charge       3333 non-null   float64
 19  customer service calls  3333 non-null   int64  
 20  churn                   3333 non-null   bool   
dtypes: bool(1), float64(8), int64(8), object(4)
memory usage: 524.2+ KB
None
       account length    area code  number vmail messages  total day minutes  \
count     3333.000000  3333.000000            3333.000000        3333.000000   
mean       101.064806   437.182418               8.099010         179.775098   
std         39.822106    42.371290              13.688365          54.467389   
min          1.000000   408.000000               0.000000           0.000000   
25%         74.000000   408.000000               0.000000         143.700000   
50%        101.000000   415.000000               0.000000         179.400000   
75%        127.000000   510.000000              20.000000         216.400000   
max        243.000000   510.000000              51.000000         350.800000   

       total day calls  total day charge  total eve minutes  total eve calls  \
count      3333.000000       3333.000000        3333.000000      3333.000000   
mean        100.435644         30.562307         200.980348       100.114311   
std          20.069084          9.259435          50.713844        19.922625   
min           0.000000          0.000000           0.000000         0.000000   
25%          87.000000         24.430000         166.600000        87.000000   
50%         101.000000         30.500000         201.400000       100.000000   
75%         114.000000         36.790000         235.300000       114.000000   
max         165.000000         59.640000         363.700000       170.000000   

       total eve charge  total night minutes  total night calls  \
count       3333.000000          3333.000000        3333.000000   
mean          17.083540           200.872037         100.107711   
std            4.310668            50.573847          19.568609   
min            0.000000            23.200000          33.000000   
25%           14.160000           167.000000          87.000000   
50%           17.120000           201.200000         100.000000   
75%           20.000000           235.300000         113.000000   
max           30.910000           395.000000         175.000000   

       total night charge  total intl minutes  total intl calls  \
count         3333.000000         3333.000000       3333.000000   
mean             9.039325           10.237294          4.479448   
std              2.275873            2.791840          2.461214   
min              1.040000            0.000000          0.000000   
25%              7.520000            8.500000          3.000000   
50%              9.050000           10.300000          4.000000   
75%             10.590000           12.100000          6.000000   
max             17.770000           20.000000         20.000000   

       total intl charge  customer service calls  
count        3333.000000             3333.000000  
mean            2.764581                1.562856  
std             0.753773                1.315491  
min             0.000000                0.000000  
25%             2.300000                1.000000  
50%             2.780000                1.000000  
75%             3.270000                2.000000  
max             5.400000                9.000000  
state                     0
account length            0
area code                 0
phone number              0
international plan        0
voice mail plan           0
number vmail messages     0
total day minutes         0
total day calls           0
total day charge          0
total eve minutes         0
total eve calls           0
total eve charge          0
total night minutes       0
total night calls         0
total night charge        0
total intl minutes        0
total intl calls          0
total intl charge         0
customer service calls    0
churn                     0
dtype: int64
Preprocessing
# drop irrelevant identifiers
# drop irrelevant identifiers
df.drop(['phone number'], axis=1, inplace=True)
Encode target
# we are Converting Categorical Target (Churn: True/False → 1/0)
df['churn'] = df['churn'].map({True: 1, False: 0})
df
state	account length	area code	international plan	voice mail plan	number vmail messages	total day minutes	total day calls	total day charge	total eve minutes	total eve calls	total eve charge	total night minutes	total night calls	total night charge	total intl minutes	total intl calls	total intl charge	customer service calls	churn
0	KS	128	415	no	yes	25	265.1	110	45.07	197.4	99	16.78	244.7	91	11.01	10.0	3	2.70	1	0
1	OH	107	415	no	yes	26	161.6	123	27.47	195.5	103	16.62	254.4	103	11.45	13.7	3	3.70	1	0
2	NJ	137	415	no	no	0	243.4	114	41.38	121.2	110	10.30	162.6	104	7.32	12.2	5	3.29	0	0
3	OH	84	408	yes	no	0	299.4	71	50.90	61.9	88	5.26	196.9	89	8.86	6.6	7	1.78	2	0
4	OK	75	415	yes	no	0	166.7	113	28.34	148.3	122	12.61	186.9	121	8.41	10.1	3	2.73	3	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3328	AZ	192	415	no	yes	36	156.2	77	26.55	215.5	126	18.32	279.1	83	12.56	9.9	6	2.67	2	0
3329	WV	68	415	no	no	0	231.1	57	39.29	153.4	55	13.04	191.3	123	8.61	9.6	4	2.59	3	0
3330	RI	28	510	no	no	0	180.8	109	30.74	288.8	58	24.55	191.9	91	8.64	14.1	6	3.81	2	0
3331	CT	184	510	yes	no	0	213.8	105	36.35	159.6	84	13.57	139.2	137	6.26	5.0	10	1.35	2	0
3332	TN	74	415	no	yes	25	234.4	113	39.85	265.9	82	22.60	241.4	77	10.86	13.7	4	3.70	0	0
3333 rows × 20 columns

print(df.dtypes)
state                      object
account length              int64
area code                   int64
phone number               object
international plan         object
voice mail plan            object
number vmail messages       int64
total day minutes         float64
total day calls             int64
total day charge          float64
total eve minutes         float64
total eve calls             int64
total eve charge          float64
total night minutes       float64
total night calls           int64
total night charge        float64
total intl minutes        float64
total intl calls            int64
total intl charge         float64
customer service calls      int64
churn                       int64
dtype: object
The above is the display of our data types
# idenifying the columns that are text based
print(df.select_dtypes(include='object').nunique())
​
state                 51
international plan     2
voice mail plan        2
dtype: int64
Label Encoding
# encoding changing yes/no to binary in international plan and voice mail plan
le = LabelEncoder()
df['international plan'] = le.fit_transform(df['international plan'])
df['voice mail plan'] = le.fit_transform(df['voice mail plan'])
df
state	account length	area code	international plan	voice mail plan	number vmail messages	total day minutes	total day calls	total day charge	total eve minutes	total eve calls	total eve charge	total night minutes	total night calls	total night charge	total intl minutes	total intl calls	total intl charge	customer service calls	churn
0	KS	128	415	0	1	25	265.1	110	45.07	197.4	99	16.78	244.7	91	11.01	10.0	3	2.70	1	0
1	OH	107	415	0	1	26	161.6	123	27.47	195.5	103	16.62	254.4	103	11.45	13.7	3	3.70	1	0
2	NJ	137	415	0	0	0	243.4	114	41.38	121.2	110	10.30	162.6	104	7.32	12.2	5	3.29	0	0
3	OH	84	408	1	0	0	299.4	71	50.90	61.9	88	5.26	196.9	89	8.86	6.6	7	1.78	2	0
4	OK	75	415	1	0	0	166.7	113	28.34	148.3	122	12.61	186.9	121	8.41	10.1	3	2.73	3	0
...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...	...
3328	AZ	192	415	0	1	36	156.2	77	26.55	215.5	126	18.32	279.1	83	12.56	9.9	6	2.67	2	0
3329	WV	68	415	0	0	0	231.1	57	39.29	153.4	55	13.04	191.3	123	8.61	9.6	4	2.59	3	0
3330	RI	28	510	0	0	0	180.8	109	30.74	288.8	58	24.55	191.9	91	8.64	14.1	6	3.81	2	0
3331	CT	184	510	1	0	0	213.8	105	36.35	159.6	84	13.57	139.2	137	6.26	5.0	10	1.35	2	0
3332	TN	74	415	0	1	25	234.4	113	39.85	265.9	82	22.60	241.4	77	10.86	13.7	4	3.70	0	0
3333 rows × 20 columns

We have manged to convert the internation plan and the voice mail plan into binary using the label encoding since it only has two features which is "Yes/No" thus making it more effecient for modelling.
a
df.drop(['state', 'area code'], axis=1, inplace=True)
​
Scaling
# Feature Scaling (for numeric columns)
scaler = StandardScaler()
num_cols = ['account length', 'total day minutes', 'total day calls', 'total day charge',
            'total eve minutes', 'total eve calls', 'total eve charge',
            'total night minutes', 'total night calls', 'total night charge',
            'total intl minutes', 'total intl calls', 'total intl charge',
            'customer service calls']
​
df[num_cols] = scaler.fit_transform(df[num_cols])
df.head()
account length	international plan	voice mail plan	number vmail messages	total day minutes	total day calls	total day charge	total eve minutes	total eve calls	total eve charge	total night minutes	total night calls	total night charge	total intl minutes	total intl calls	total intl charge	customer service calls	churn
0	0.676489	0	1	25	1.566767	0.476643	1.567036	-0.070610	-0.055940	-0.070427	0.866743	-0.465494	0.866029	-0.085008	-0.601195	-0.085690	-0.427932	0
1	0.149065	0	1	26	-0.333738	1.124503	-0.334013	-0.108080	0.144867	-0.107549	1.058571	0.147825	1.059390	1.240482	-0.601195	1.241169	-0.427932	0
2	0.902529	0	0	0	1.168304	0.675985	1.168464	-1.573383	0.496279	-1.573900	-0.756869	0.198935	-0.755571	0.703121	0.211534	0.697156	-1.188218	0
3	-0.428590	1	0	0	2.196596	-1.466936	2.196759	-2.742865	-0.608159	-2.743268	-0.078551	-0.567714	-0.078806	-1.303026	1.024263	-1.306401	0.332354	0
4	-0.654629	1	0	0	-0.240090	0.626149	-0.240041	-1.038932	1.098699	-1.037939	-0.276311	1.067803	-0.276562	-0.049184	-0.601195	-0.045885	1.092641	0
Classification
Logistic regression _ this we used to predict categorical outcomes moreso the binary outcome. it also model the probability that a given outcome belong to a certain class
Split Data
#  Define Features (X) and Target (y)
X = df.drop('churn', axis=1)
y = df['churn']
​
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train, X_test, y_train, y_test)
      account length  international plan  voice mail plan  \
817         3.564766                   0                0   
1373        0.174180                   0                0   
679        -0.654629                   1                0   
56          1.002990                   0                0   
1993       -0.378359                   0                0   
...              ...                 ...              ...   
1095        0.123950                   0                0   
1130        0.525797                   0                0   
1294       -0.880668                   0                0   
860         1.706223                   0                0   
3174       -1.634132                   0                1   

      number vmail messages  total day minutes  total day calls  \
817                       0          -1.547490        -0.420393   
1373                      0          -1.244511         0.227466   
679                       0           0.782694        -1.118088   
56                        0          -0.970912        -0.121381   
1993                      0           0.670684        -0.221052   
...                     ...                ...              ...   
1095                      0           1.737537         0.974997   
1130                      0          -2.656577        -1.915454   
1294                      0          -1.692553        -1.217759   
860                       0          -0.010560         0.526479   
3174                     43          -2.752061         1.124503   

      total day charge  total eve minutes  total eve calls  total eve charge  \
817          -1.547012          -0.735222        -1.863202         -0.736317   
1373         -1.244572          -0.143579         0.496279         -0.144672   
679           0.782853           2.485289         0.546480          2.486405   
56           -0.971297          -0.413763        -1.913404         -0.413812   
1993          0.670519           1.288198        -1.160378          1.289196   
...                ...                ...              ...               ...   
1095          1.737699          -0.046944        -0.909370         -0.047225   
1130         -2.656317          -0.397986        -0.557958         -0.397571   
1294         -1.692831           1.203395         0.546480          1.203349   
860          -0.011051          -0.508426         1.500313         -0.508940   
3174         -2.752450          -1.417584         0.847691         -1.418448   

      total night minutes  total night calls  total night charge  \
817              1.252376           0.914473            1.252751   
1373             0.158761          -0.363275            0.158501   
679              0.140963           0.198935            0.140923   
56              -1.187987           1.425573           -1.186239   
1993             0.259619           0.505594            0.259577   
...                   ...                ...                 ...   
1095            -0.792466          -1.947682           -0.790727   
1130             1.003198          -2.152122            1.002260   
1294            -0.321796           1.272243           -0.320508   
860              0.548349          -0.414384            0.549619   
3174             2.472557           0.250045            2.474444   

      total intl minutes  total intl calls  total intl charge  \
817            -1.303026          0.617898          -1.306401   
1373           -2.198627         -0.194831          -2.195396   
679            -0.550721          1.836992          -0.550091   
56             -0.801489         -1.007560          -0.802194   
1993           -2.055331         -0.601195          -2.049442   
...                  ...               ...                ...   
1095           -1.517970         -0.601195          -1.518698   
1130            0.882241         -1.007560           0.882917   
1294           -0.371601          0.211534          -0.377599   
860            -0.120832          0.617898          -0.125496   
3174           -0.586545          0.617898          -0.589897   

      customer service calls  
817                 0.332354  
1373                1.852927  
679                -0.427932  
56                 -0.427932  
1993               -1.188218  
...                      ...  
1095               -0.427932  
1130               -0.427932  
1294               -0.427932  
860                 0.332354  
3174                0.332354  

[2666 rows x 17 columns]       account length  international plan  voice mail plan  \
438         0.299758                   0                0   
2674       -0.855553                   0                0   
1345       -0.076974                   0                0   
1957        1.153683                   0                0   
2148       -0.127205                   0                0   
...              ...                 ...              ...   
2577        1.404837                   0                0   
2763        0.375104                   0                1   
3069        1.178798                   0                1   
1468       -0.654629                   0                1   
582         0.073719                   0                0   

      number vmail messages  total day minutes  total day calls  \
438                       0          -0.454929        -0.370558   
2674                      0          -1.297762         0.825491   
1345                      0          -3.301096        -5.005247   
1957                      0           0.606415        -1.068253   
2148                      0          -0.656915         0.077960   
...                     ...                ...              ...   
2577                      0           0.097778        -0.420393   
2763                     19          -0.442076         0.177631   
3069                     26          -0.386989        -0.470229   
1468                     27          -1.143518         0.077960   
582                       0          -0.285996         0.426808   

      total day charge  total eve minutes  total eve calls  total eve charge  \
438          -0.454989           2.556286         0.295472          2.556011   
2674         -1.297499           0.323819         1.199103          0.324003   
1345         -3.301162          -0.816080         1.500313         -0.815203   
1957          0.606790           0.061524        -0.457554          0.061823   
2148         -0.656975           0.467786        -1.361185          0.467854   
...                ...                ...              ...               ...   
2577          0.098044           0.237045        -0.758764          0.238157   
2763         -0.442027          -0.307267         0.897892         -0.307084   
3069         -0.386940          -0.798331         1.349708         -0.798961   
1468         -1.143039           0.114772         1.349708          0.115187   
582          -0.286487          -0.898910        -0.507756         -0.898729   

      total night minutes  total night calls  total night charge  \
438             -0.226871           1.170023           -0.228221   
2674            -0.246647           2.090002           -0.245800   
1345            -0.667877          -0.618824           -0.667679   
1957            -0.883436           0.658924           -0.883013   
2148             0.530551          -0.465494            0.532041   
...                   ...                ...                 ...   
2577            -0.094372          -0.772154           -0.096384   
2763            -0.161610           0.812254           -0.162303   
3069             0.344656          -0.618824            0.343074   
1468            -0.127991           0.710034           -0.127146   
582             -0.632280           0.863364           -0.632523   

      total intl minutes  total intl calls  total intl charge  \
438             1.168834         -0.601195           1.174826   
2674            0.918065          0.617898           0.922722   
1345           -1.231378         -1.413924          -1.226789   
1957           -0.013360         -1.007560          -0.019348   
2148           -0.085008          1.024263          -0.085690   
...                  ...               ...                ...   
2577           -0.622369          0.211534          -0.616434   
2763           -0.729841         -1.007560          -0.735851   
3069           -0.120832         -0.601195          -0.125496   
1468           -2.162803          1.024263          -2.168859   
582             0.165760          1.430627           0.166413   

      customer service calls  
438                -0.427932  
2674               -1.188218  
1345                1.852927  
1957               -0.427932  
2148               -0.427932  
...                      ...  
2577                0.332354  
2763                1.092641  
3069               -0.427932  
1468                1.092641  
582                -0.427932  

[667 rows x 17 columns] 817     0
1373    1
679     1
56      0
1993    0
       ..
1095    0
1130    0
1294    0
860     0
3174    0
Name: churn, Length: 2666, dtype: int64 438     0
2674    0
1345    1
1957    0
2148    0
       ..
2577    0
2763    0
3069    0
1468    0
582     0
Name: churn, Length: 667, dtype: int64
Model Training and Evaluation
Linear Regression
​
#  Initialize and train model
log_model = LogisticRegression(max_iter=1000, random_state=42)
log_model.fit(X_train, y_train)
​
y_pred_log = log_model.predict(X_test)
​
# Evaluation
print("Classification Report:\n", classification_report(y_test, y_pred_log))
sns.heatmap(confusion_matrix(y_test, y_pred_log), annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.show()
Classification Report:
               precision    recall  f1-score   support

           0       0.87      0.98      0.92       566
           1       0.57      0.16      0.25       101

    accuracy                           0.85       667
   macro avg       0.72      0.57      0.58       667
weighted avg       0.82      0.85      0.82       667


Logistic Regression 
#  ROC Curve
y_prob = log_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
​
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Logistic Regression ROC Curve")
plt.legend()
plt.grid()
plt.show()
​
​

Decision Tree
)
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
# ROC Curve
y_prob_dt = dt_m
# Initialize the model
dt_model = DecisionTreeClassifier(random_state=42)
​
# Fit the model
dt_model.fit(X_train, y_train)
​
# Make predictions
y_pred_dt = dt_model.predict(X_test)
​
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred_dt))
# Confusion Matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt))
# ROC Curve
y_prob_dt = dt_model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob_dt)
​
# Plot ROC
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob_dt):.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Decision Tree ROC Curve")
plt.legend()
plt.grid()
plt.show()
Classification Report:
              precision    recall  f1-score   support

           0       0.95      0.96      0.96       566
           1       0.76      0.73      0.74       101

    accuracy                           0.92       667
   macro avg       0.85      0.85      0.85       667
weighted avg       0.92      0.92      0.92       667

Confusion Matrix:
[[542  24]
 [ 27  74]]

This model is perfectly fit as it accuracy is close to 1.

Linear Regression
# Initialize the model
lin_model = LinearRegression()
​
# Fit the model
lin_model.fit(X_train, y_train)
​
# Predict probabilities (continuous outputs)
y_prob_lin = lin_model.predict(X_test)
​
# Convert probabilities to binary classification (threshold = 0.5)
y_pred_lin = np.where(y_prob_lin >= 0.5, 1, 0)
​
# Classification metrics
print("Classification Report:")
print(classification_report(y_test, y_pred_lin))
​
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_lin))
​
# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob_lin)
auc_score = roc_auc_score(y_test, y_prob_lin)
​
# Plot ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Linear Regression ROC Curve")
plt.legend()
plt.grid()
plt.show()
Classification Report:
              precision    recall  f1-score   support

           0       0.86      0.99      0.92       566
           1       0.65      0.11      0.19       101

    accuracy                           0.86       667
   macro avg       0.75      0.55      0.55       667
weighted avg       0.83      0.86      0.81       667

Confusion Matrix:
[[560   6]
 [ 90  11]]

linear regression model is not fit hence not appropriate for use in our dataset

# Get coefficients (as importance)
importances = log_model.coef_[0]  # model.coef_ is a 2D array: shape (1, n_features)
​
# Get feature names
feature_names = X.columns
​
# Create a DataFrame
fi_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
​
#  Sort by absolute importance
fi_df['Abs_Importance'] = fi_df['Importance'].abs()
fi_df.sort_values('Abs_Importance', ascending=False)
​
# Plot
​
plt.figure(figsize=(10,8))
sns.barplot(data=fi_df, x='Abs_Importance', y='Feature', palette='viridis')
plt.title('Logistic Regression Feature Importance (based on Coefficients)')
plt.tight_layout()
plt.show()
​
​

Conclusion
Logistic Regression and Decision Tree models are effective for churn prediction.
Linear Regression is included for comparison but not ideal for classification.

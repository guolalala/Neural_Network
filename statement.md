### 查看数据集的样本个数和原始特征纬度

#### data_test_a.shape

```
 (2574, 51)
```

#### data_train.shape 

```
(3968, 52)
```

#### data_train.columns

```
Index(['id', 'base_month_income', 'base_car_km', 'base_loan_rate',
       'base_loan_monthrepay', 'base_car_years', 'base_ages',
       'br_scoreafautofin', 'ir_id_x_cell_cnt', 'ir_m12_id_x_home_addr_cnt',
       'ir_allmatch_days', 'als_m12_id_tot_mons', 'als_m12_id_max_monnum',
       'als_m12_id_nbank_oth_orgnum', 'als_m12_id_nbank_allnum',
       'als_lst_cell_bank_inteday', 'als_m12_id_nbank_tot_mons',
       'als_m12_id_nbank_max_monnum', 'als_m12_id_rel_allnum',
       'als_m12_id_nbank_cons_orgnum', 'als_m12_cell_bank_tra_allnum',
       'als_m12_cell_nbank_cf_allnum', 'als_m12_id_nbank_finlea_allnum',
       'als_m12_cell_af_allnum', 'als_m12_cell_pdl_allnum',
       'als_m6_id_nbank_allnum', 'als_m6_id_max_monnum',
       'als_m6_id_nbank_oth_orgnum', 'als_m6_id_nbank_tot_mons',
       'als_m6_id_nbank_max_monnum', 'als_m3_id_nbank_allnum',
       'als_m3_id_max_monnum', 'als_m3_id_nbank_oth_orgnum',
       'als_m3_id_nbank_tot_mons', 'als_m3_id_nbank_max_monnum',
       'als_d15_id_nbank_oth_allnum', 'fico_result', 'fico_app_cnt',
       'base_is_guarantee', 'base_nation', 'base_is_local', 'base_has_rebuy',
       'base_sex', 'zx_result', 'zx_is_baihu', 'zx_is_current_ovd',
       'zx_is_credictcard_current_ovd', 'zx_is_lian3_lei6', 'loan_is_black',
       'base_has_driverlicensce', 'zx_account_status', 'fraud_flag'],
      dtype='object')
<class 'pandas.core.frame.DataFrame'>
```



### 数据类型

```
RangeIndex: 3968 entries, 0 to 3967
Data columns (total 52 columns):
 #   Column                          Non-Null Count  Dtype  
---  ------                          --------------  -----  
 0   id                              3968 non-null   object 
 1   base_month_income               3968 non-null   int64  
 2   base_car_km                     3968 non-null   int64  
 3   base_loan_rate                  3968 non-null   int64  
 4   base_loan_monthrepay            3968 non-null   int64  
 5   base_car_years                  3968 non-null   int64  
 6   base_ages                       3968 non-null   int64  
 7   br_scoreafautofin               2710 non-null   float64
 8   ir_id_x_cell_cnt                3154 non-null   float64
 9   ir_m12_id_x_home_addr_cnt       2685 non-null   float64
 10  ir_allmatch_days                2814 non-null   float64
 11  als_m12_id_tot_mons             2692 non-null   float64
 12  als_m12_id_max_monnum           2692 non-null   float64
 13  als_m12_id_nbank_oth_orgnum     1292 non-null   float64
 14  als_m12_id_nbank_allnum         2403 non-null   float64
 15  als_lst_cell_bank_inteday       2403 non-null   float64
 16  als_m12_id_nbank_tot_mons       2403 non-null   float64
 17  als_m12_id_nbank_max_monnum     2403 non-null   float64
 18  als_m12_id_rel_allnum           1815 non-null   float64
 19  als_m12_id_nbank_cons_orgnum    1535 non-null   float64
 20  als_m12_cell_bank_tra_allnum    1329 non-null   float64
 21  als_m12_cell_nbank_cf_allnum    1776 non-null   float64
 22  als_m12_id_nbank_finlea_allnum  560 non-null    float64
 23  als_m12_cell_af_allnum          740 non-null    float64
 24  als_m12_cell_pdl_allnum         1262 non-null   float64
 25  als_m6_id_nbank_allnum          2007 non-null   float64
 26  als_m6_id_max_monnum            2281 non-null   float64
 27  als_m6_id_nbank_oth_orgnum      978 non-null    float64
 28  als_m6_id_nbank_tot_mons        2007 non-null   float64
 29  als_m6_id_nbank_max_monnum      2007 non-null   float64
 30  als_m3_id_nbank_allnum          1623 non-null   float64
 31  als_m3_id_max_monnum            1844 non-null   float64
 32  als_m3_id_nbank_oth_orgnum      760 non-null    float64
 33  als_m3_id_nbank_tot_mons        1623 non-null   float64
 34  als_m3_id_nbank_max_monnum      1623 non-null   float64
 35  als_d15_id_nbank_oth_allnum     386 non-null    float64
 36  fico_result                     3430 non-null   float64
 37  fico_app_cnt                    3430 non-null   float64
 38  base_is_guarantee               914 non-null    float64
 39  base_nation                     2838 non-null   float64
 40  base_is_local                   3108 non-null   float64
 41  base_has_rebuy                  1089 non-null   float64
 42  base_sex                        2793 non-null   float64
 43  zx_result                       3501 non-null   float64
 44  zx_is_baihu                     666 non-null    float64
 45  zx_is_current_ovd               50 non-null     float64
 46  zx_is_credictcard_current_ovd   19 non-null     float64
 47  zx_is_lian3_lei6                134 non-null    float64
 48  loan_is_black                   15 non-null     float64
 49  base_has_driverlicensce         3968 non-null   int64  
 50  zx_account_status               14 non-null     float64
 51  fraud_flag                      3968 non-null   int64  
dtypes: float64(43), int64(8), object(1)
memory usage: 1.6+ MB
```



### 一些基本统计量

```
       base_month_income  base_car_km  ...  zx_account_status   fraud_flag
count        3968.000000  3968.000000  ...          14.000000  3968.000000
mean            2.834929     3.515877  ...           1.857143     0.069808
std             1.305867     1.588892  ...           0.770329     0.254856
min             1.000000     1.000000  ...           1.000000     0.000000
25%             2.000000     2.000000  ...           1.000000     0.000000
50%             3.000000     4.000000  ...           2.000000     0.000000
75%             4.000000     4.000000  ...           2.000000     0.000000
max             6.000000     6.000000  ...           3.000000     1.000000
[8 rows x 51 columns]
```



### 查看缺失值

```
There are 43 columns in train dataset with missing values.
```

### 查看缺失特征中缺失率大于50%的特征

```
{'als_m12_id_nbank_oth_orgnum': 0.6743951612903226, 'als_m12_id_rel_allnum': 0.5425907258064516, 'als_m12_id_nbank_cons_orgnum': 0.6131552419354839, 'als_m12_cell_bank_tra_allnum': 0.665070564516129, 'als_m12_cell_nbank_cf_allnum': 0.5524193548387096, 'als_m12_id_nbank_finlea_allnum': 0.8588709677419355, 'als_m12_cell_af_allnum': 0.813508064516129, 'als_m12_cell_pdl_allnum': 0.6819556451612904, 'als_m6_id_nbank_oth_orgnum': 0.7535282258064516, 'als_m3_id_nbank_allnum': 0.5909778225806451, 'als_m3_id_max_monnum': 0.5352822580645161, 'als_m3_id_nbank_oth_orgnum': 0.8084677419354839, 'als_m3_id_nbank_tot_mons': 0.5909778225806451, 'als_m3_id_nbank_max_monnum': 0.5909778225806451, 'als_d15_id_nbank_oth_allnum': 0.9027217741935484, 'base_is_guarantee': 0.7696572580645161, 'base_has_rebuy': 0.725554435483871, 'zx_is_baihu': 0.8321572580645161, 'zx_is_current_ovd': 0.9873991935483871, 'zx_is_credictcard_current_ovd': 0.9952116935483871, 'zx_is_lian3_lei6': 0.9662298387096774, 'loan_is_black': 0.9962197580645161, 'zx_account_status': 0.9964717741935484}
```



### 查看训练集测试集中特征属性只有一值的特征

```
['base_is_guarantee', 'base_nation', 'base_is_local', 'base_has_rebuy', 'base_sex', 'zx_result', 'zx_is_baihu', 'zx_is_current_ovd', 'zx_is_credictcard_current_ovd', 'zx_is_lian3_lei6', 'loan_is_black']
There are 11 columns in train dataset with one unique value.
```



### 查看特征的数值类型

```
['id']:'object'
```



### 数值类别型变量分析

```
3    1176
2    1026
1     670
4     594
5     389
6     113
Name: base_month_income, dtype: int64
2    1191
4    1119
6     785
3     449
1     271
5     153
Name: base_car_km, dtype: int64
2    1559
3    1409
1    1000
Name: base_loan_rate, dtype: int64
5    1498
4     893
3     606
2     558
1     413
Name: base_loan_monthrepay, dtype: int64
5    1966
4     894
2     595
3     293
1     220
Name: base_car_years, dtype: int64
1    1233
2    1108
3     926
4     378
5     323
Name: base_ages, dtype: int64
6.0    880
5.0    862
4.0    465
3.0    338
2.0    165
Name: br_scoreafautofin, dtype: int64
2.0    2039
3.0     822
4.0     293
Name: ir_id_x_cell_cnt, dtype: int64
2.0    1978
3.0     535
4.0     172
Name: ir_m12_id_x_home_addr_cnt, dtype: int64
2.0    725
6.0    502
3.0    499
7.0    409
5.0    351
4.0    328
Name: ir_allmatch_days, dtype: int64
2.0    792
6.0    702
3.0    530
4.0    379
5.0    289
Name: als_m12_id_tot_mons, dtype: int64
2.0    1114
3.0     659
4.0     400
5.0     300
6.0     219
Name: als_m12_id_max_monnum, dtype: int64
2.0    691
3.0    327
5.0    149
4.0    125
Name: als_m12_id_nbank_oth_orgnum, dtype: int64
2.0    656
4.0    451
3.0    420
5.0    368
7.0    305
6.0    203
Name: als_m12_id_nbank_allnum, dtype: int64
2.0    945
4.0    688
3.0    566
5.0    204
Name: als_lst_cell_bank_inteday, dtype: int64
2.0    846
6.0    521
3.0    483
4.0    318
5.0    235
Name: als_m12_id_nbank_tot_mons, dtype: int64
2.0    1161
3.0     571
4.0     280
5.0     230
6.0     161
Name: als_m12_id_nbank_max_monnum, dtype: int64
4.0    729
2.0    705
3.0    381
Name: als_m12_id_rel_allnum, dtype: int64
2.0    824
3.0    433
4.0    278
Name: als_m12_id_nbank_cons_orgnum, dtype: int64
2.0    623
4.0    370
3.0    336
Name: als_m12_cell_bank_tra_allnum, dtype: int64
4.0    750
2.0    684
3.0    342
Name: als_m12_cell_nbank_cf_allnum, dtype: int64
2.0    422
3.0    107
4.0     31
Name: als_m12_id_nbank_finlea_allnum, dtype: int64
2.0    501
3.0    158
4.0     81
Name: als_m12_cell_af_allnum, dtype: int64
2.0    564
3.0    501
4.0    197
Name: als_m12_cell_pdl_allnum, dtype: int64
2.0    672
3.0    403
7.0    278
5.0    276
4.0    231
6.0    147
Name: als_m6_id_nbank_allnum, dtype: int64
2.0    1029
3.0     527
4.0     302
5.0     240
6.0     183
Name: als_m6_id_max_monnum, dtype: int64
2.0    547
3.0    249
4.0    182
Name: als_m6_id_nbank_oth_orgnum, dtype: int64
2.0    911
4.0    606
3.0    490
Name: als_m6_id_nbank_tot_mons, dtype: int64
2.0    1007
3.0     465
4.0     222
5.0     178
6.0     135
Name: als_m6_id_nbank_max_monnum, dtype: int64
2.0    672
3.0    336
4.0    193
7.0    167
6.0    143
5.0    112
Name: als_m3_id_nbank_allnum, dtype: int64
2.0    872
3.0    416
4.0    228
5.0    187
6.0    141
Name: als_m3_id_max_monnum, dtype: int64
2.0    454
3.0    179
4.0    127
Name: als_m3_id_nbank_oth_orgnum, dtype: int64
2.0    996
3.0    413
4.0    214
Name: als_m3_id_nbank_tot_mons, dtype: int64
2.0    833
3.0    363
4.0    180
5.0    141
6.0    106
Name: als_m3_id_nbank_max_monnum, dtype: int64
2.0    252
3.0     92
4.0     42
Name: als_d15_id_nbank_oth_allnum, dtype: int64
4.0    1156
5.0     474
7.0     458
8.0     395
3.0     339
6.0     279
2.0     178
9.0     151
Name: fico_result, dtype: int64
2.0    3222
3.0     208
Name: fico_app_cnt, dtype: int64
1.0    914
Name: base_is_guarantee, dtype: int64
1.0    2838
Name: base_nation, dtype: int64
1.0    3108
Name: base_is_local, dtype: int64
1.0    1089
Name: base_has_rebuy, dtype: int64
1.0    2793
Name: base_sex, dtype: int64
1.0    3501
Name: zx_result, dtype: int64
1.0    666
Name: zx_is_baihu, dtype: int64
1.0    50
Name: zx_is_current_ovd, dtype: int64
1.0    19
Name: zx_is_credictcard_current_ovd, dtype: int64
1.0    134
Name: zx_is_lian3_lei6, dtype: int64
1.0    15
Name: loan_is_black, dtype: int64
2    2806
1     717
3     445
Name: base_has_driverlicensce, dtype: int64
2.0    6
1.0    5
3.0    3
Name: zx_account_status, dtype: int64
0    3691
1     277
Name: fraud_flag, dtype: int64
```


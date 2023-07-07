# Pandas
import pandas as pd
# from pandas import Series,DataFrame

# Numpy and Matplotlib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn import ensemble

# Deep Learning
import torch

# 导入数据
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

# 控制窗口显示的最大行数和最大列数
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 1000)

# head()输出前五行
print(train.head())
print(test.head())
# print(train.info)
# print(test.info)
print(".train文件的不同特征的NA数量：")
print(train.isnull().sum())
print(".test文件的不同特征的NA数量：")
print(test.isnull().sum())
print(train['SalePrice'].describe())

# 右边尾长，右偏，做对数变换，让它符合正态分布的假设
# Return unbiased skew within groups.
print("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice)
plt.show()
train['Skewed_SalePrice'] = np.log(train['SalePrice'])
print("Skew is:", train['Skewed_SalePrice'].skew())
plt.hist(train.Skewed_SalePrice)
plt.show()

# MSSubClass数据图
sns.catplot("MSSubClass", "SalePrice", data=train, kind="bar", height=3, aspect=3)
sns.catplot("MSSubClass", "Skewed_SalePrice", data=train, kind="bar", height=3, aspect=3)
fig, (axis1) = plt.subplots(1, 1)
sns.countplot('MSSubClass', data=train)
print(train['MSSubClass'].value_counts())
plt.show()

# MSZoning数据图
sns.catplot("MSZoning", "SalePrice", data=train, kind="bar", height=3, aspect=3)
sns.catplot("MSZoning", "Skewed_SalePrice", data=train, kind="bar", height=3, aspect=3)
fig, (axis1) = plt.subplots(1, 1, figsize=(10, 3))
sns.countplot("MSZoning", data=train, ax=axis1)
print(train["MSZoning"].value_counts())
plt.show()

# select value为数字的feature
numerical_features = train.select_dtypes(include=[np.number])
print(numerical_features.dtypes)
# find the corretation between the feature and target
# .corr():Compute pairwise correlation of columns, excluding NA/null values.
corr = numerical_features.corr()
print(corr['Skewed_SalePrice'].sort_values(ascending=False)[:], '\n')

# OverallQual数据图
# unique():去除重复的元素，返回一个无元素重复的数组或列表
print(train["OverallQual"].unique())
sns.catplot("OverallQual", "SalePrice", data=train, kind="bar", height=3, aspect=3)
plt.show()
sns.catplot("OverallQual", "Skewed_SalePrice", data=train, kind="bar", height=3, aspect=3)
plt.show()
sns.countplot(x=train["OverallQual"])
print(train["OverallQual"].value_counts())
quality_pivot_median = train.pivot_table(index='OverallQual', values='SalePrice', aggfunc=np.median)
print(quality_pivot_median)
quality_pivot_mean = train.pivot_table(index="OverallQual", values="SalePrice", aggfunc=np.mean)
print(quality_pivot_mean)
# 可以看出正相关
quality_pivot_median.plot(kind="bar")
plt.xlabel("OverallQual")
plt.ylabel("Median")
plt.show()
quality_pivot_mean.plot(kind="bar")
plt.xlabel("OverallQual")
plt.ylabel("Mean")
plt.show()

# GrLivArea数据图
sns.regplot(x="GrLivArea", y="SalePrice", data=train)
plt.show()
sns.catplot("GrLivArea", "SalePrice", data=train, kind="bar", height=3, aspect=3)
plt.show()
# 总体上正相关
# remove outliers
print("GrLivArea中位数：")
GrLivArea_median = train["GrLivArea"].median()
print(GrLivArea_median)
GrLivArea_median2 = abs(train["GrLivArea"] - GrLivArea_median)
print(GrLivArea_median2)
print("中位数差的中位数：")
pre_GrLivArea_mad = np.median(GrLivArea_median2)
print(pre_GrLivArea_mad)
# GrLivArea_median2=GrLivArea_median2.median()
print("MAD:")
GrLivArea_MAD = 1.4826 * pre_GrLivArea_mad
print(GrLivArea_MAD)
train = train[train["GrLivArea"] < (GrLivArea_median + 3 * GrLivArea_MAD)]
train = train[train["GrLivArea"] > (GrLivArea_median - 3 * GrLivArea_MAD)]
sns.regplot(x="GrLivArea", y="SalePrice", data=train)
plt.show()

# GarageCars数据图
sns.regplot(x="GarageCars", y="SalePrice", data=train)
plt.show()
sns.catplot("GarageCars", "SalePrice", data=train, kind="bar", height=3, aspect=3)
plt.show()

# GarageArea数据图
sns.regplot(x="GarageArea", y="SalePrice", data=train)
plt.show()
sns.catplot("GarageArea", "SalePrice", data=train, kind="bar", height=3, aspect=3)
plt.show()
print("GarageArea中位数：")
GarageArea_median = train["GarageArea"].median()
print(GarageArea_median)
GarageArea_median2 = abs(train["GarageArea"] - GarageArea_median)
print(GarageArea_median2)
print("中位数差的中位数：")
pre_GarageArea_mad = np.median(GarageArea_median2)
print(pre_GarageArea_mad)
print("MAD:")
GarageArea_MAD = 1.4826 * pre_GarageArea_mad
print(GarageArea_MAD)
train = train[train["GarageArea"] < (GarageArea_median + 3 * GarageArea_MAD)]
train = train[train["GarageArea"] > (GarageArea_median - 3 * GarageArea_MAD)]
sns.regplot(x="GarageArea", y="SalePrice", data=train)
plt.show()

# remove NA
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:19])
nulls.columns = ["Null Count"]
nulls.index.name = "Feature"
print(nulls)

# MiscFeature
print("unique values:")
print(train.MiscFeature.unique())

# value不是数值的特征
categories = train.select_dtypes(exclude=[np.number])
print(categories.dtypes)
print(categories.describe(include="all"))

# Neighborhood
train["Neighborhood"].value_counts().plot(kind="bar")
plt.show()
sns.factorplot(x="Neighborhood", y="SalePrice", data=train, kind="bar", aspect=3)
sns.factorplot(x="Neighborhood", y="Skewed_SalePrice", data=train, kind="bar", aspect=3)
plt.show()

# Condition1
print(train['Condition1'].value_counts())

# Condition2
print(train['Condition2'].value_counts())

# Condition1 & Condition2 关系
g = sns.factorplot(x='Condition1', y='Skewed_SalePrice', col='Condition2', data=train, kind='bar', col_wrap=4,
                   aspect=0.8)
g.set_xticklabels(rotation=90)

# SaleCondition
print(train['SaleCondition'].value_counts())

# SaleType
print(train['SaleType'].value_counts())
g = sns.factorplot(x='SaleCondition', y='Skewed_SalePrice', col='SaleType', data=train, kind='bar', col_wrap=4,
                   aspect=0.8)
g.set_xticklabels(rotation=90)
plt.show()

# Data Trasformation
print("Original: \n")
print(train.Street.value_counts(), "\n")

# one hot encoding
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print('Encoded: \n')
print(train.enc_street.value_counts())

# SaleCondition Feature Engineering
condition_pivot = train.pivot_table(index='SaleCondition',
                                    values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar')
plt.xlabel('Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()


def encode_condition(x): return 1 if x == 'Partial' else 0


train['enc_condition'] = train.SaleCondition.apply(encode_condition)
test['enc_condition'] = test.SaleCondition.apply(encode_condition)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

# Feature Engineering 批量处理
# categories_head=categories.head()
# print(categories_head)
# for feature in categories_head:
#     condition_pivot=train.pivot_table(index=feature,values="SalePrice",aggfunc=np.median)
#     condition_pivot.plot(kind="bar")
#     plt.xlabel(feature)
#     plt.ylabel("Median Sale Price")
#     plt.xticks(rotation=0)
#     plt.show()


# Interpolation
# data = train.select_dtypes(include=[np.number])
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
print(data)
# .interpolate().dropna()
# 判断还有没有NA
print(sum(data.isnull().sum() != 0))
# print(data)
# new_nulls=pd.DataFrame(data.isnull().sum().sort_values(ascending=False)[:3])
# print(new_nulls)

# 机器学习part
y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id','Skewed_SalePrice'], axis=1)

# 把train划分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.33)
print(X_train)
print(X_test)
print(y_train)
print(y_test)

# GBR
lr = ensemble.GradientBoostingRegressor()
model = lr.fit(X_train, y_train)
print ("R^2 is: \n", model.score(X_test, y_test))
predictions = model.predict(X_test)
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
actual_values = y_test
plt.scatter(predictions, actual_values, alpha=.75,color='b') #alpha helps to show overlapping data
plt.xlabel('Predicted Price')
plt.ylabel('Actual Price')
plt.title('Linear Regression Model')
plt.show()
submission = pd.DataFrame()
submission['Id'] = test.Id
feats = test.select_dtypes(include=[np.number]).drop(['Id'], axis=1).interpolate()
# print(X_train.head())
print(feats.head())
predictions = model.predict(feats)
# final_predictions = np.exp(predictions)

# print(len(X_test))
# print(len(y_test))



#
# # 岭回归
# for i in range(-2, 3):
#     alpha = 10 ** i
#     rm = linear_model.Ridge(alpha=alpha)
#     ridge_model = rm.fit(X_train, y_train)
#     preds_ridge = ridge_model.predict(X_test)
#
#     plt.scatter(preds_ridge, actual_values, alpha=.75, color='b')
#     plt.xlabel('Predicted Price')
#     plt.ylabel('Actual Price')
#     plt.title('Ridge Regularization with alpha = {}'.format(alpha))
#     overlay = 'R^2 is: {}\nRMSE is: {}'.format(
#         ridge_model.score(X_test, y_test),
#         mean_squared_error(y_test, preds_ridge))
#     plt.annotate(text=overlay, xy=(12.1, 10.6), size='x-large')
#     plt.show()
#
# submission = pd.DataFrame()
# submission['Id'] = test.Id
# feats = test.select_dtypes(
#         include=[np.number]).drop(['Id'], axis=1).interpolate()
# print(feats)
# predictions = ridge_model.predict(feats)
#


final_predictions = np.exp(predictions)
print ("Original predictions are: \n", predictions[:5], "\n")
print ("Final predictions are: \n", final_predictions[:5])
submission['SalePrice'] = final_predictions
submission.head()
submission.to_csv('submission1.csv', index=False)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
 #regression model to predict weight based on other features in the dataset
x= df.drop('Weight',axis=1)
y= df['Weight']
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,)
reg_model = LinearRegression()
reg_model.fit(x_train,y_train)


x_cls = df.drop('NObeyesdad',axis=1)
y_cls = df['NObeyesdad']
x_train_cls,x_test_cls,y_train_cls,y_test_cls = train_test_split(x_cls,y_cls,test_size=0.2,)
clf_model = RandomForestClassifier()
clf_model.fit(x_train_cls,y_train_cls)


kmeans=KMeans(n_clusters=3)
df["Cluster"] = kmeans.fit_predict(x_cls)



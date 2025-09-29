#!/usr/bin/env python
# coding: utf-8

# In[3]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from  sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

from sklearn.datasets import load_iris
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import roc_auc_score
import pandas as pd



class checkClassifierScore():
    abc ="abc"
    
    def checkLR(self,xTrain,yTrain,xTest,yTest):
        params = [{'solver': ['liblinear'], 'penalty': ['l1', 'l2']},
                  {'solver': ['lbfgs'],     'penalty': ['l2']},
                  {'solver': ['saga'],      'penalty': ['l1', 'l2', 'elasticnet'], 'l1_ratio': [0.5]}]
        gscv =GridSearchCV(LogisticRegression(),params,n_jobs=-1, refit=True,verbose =3)
        gscv.fit(xTrain,yTrain)
        
        yPred =gscv.predict(xTest)
        print(gscv.best_params_)
        # checkConfustionMatrix(yTest,yPred)
        cm = confusion_matrix(yTest,yPred)
        print(cm)
        print(classification_report(yTest, yPred))
        
        roc = roc_auc_score(yTest,gscv.predict_proba(xTest)[:,1])
        print(roc)
        # re=gscv.cv_results_
        # pd.DataFrame.from_dict(re)
        


    def checkNaviesBayes(self,xTrain,yTrain,xTest,yTest):
        gnb = GaussianNB()
        yPred = gnb.fit(xTrain, yTrain).predict(xTest)
        # print("Number of mislabeled points out of a total %d points : %d"
              # % (xTest.shape[0], (yTest != yPred).sum()))
        # checkConfustionMatrix(yTest,yPred)
        cm = confusion_matrix(yTest,yPred)
        print(cm)
        print(classification_report(yTest, yPred))

        classifier =CategoricalNB()
        classifier.fit(xTrain, yTrain)
        y_pred = classifier.predict(xTest)
        # checkConfustionMatrix(yTest,y_pred)
        cm = confusion_matrix(yTest,y_pred)
        print(cm)
        print(classification_report(yTest, y_pred))

        

    def checkConfustionMatrix(self,yTest,yPred):
        cm = confusion_matrix(yTest,yPred)
        print(cm)
        print(classification_report(yTest, yPred))
       
        
        
    
    


# In[ ]:





# In[ ]:





# In[ ]:





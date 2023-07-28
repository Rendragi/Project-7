import os
import pickle
import pandas as pd
from modules.insurance_pre import InsurancePre

class InsurancePredict():
    def runModel(self, data, typed='multi'):
        path = pathPackages = os.getcwd()+"\\"+"modules/packages"+"\\"
        model = pickle.load(open(path + 'model_InsuranceRecommendation.pkl', 'rb'))
        col_p = pickle.load(open(path + 'columnPreparation.pkl', 'rb'))
        col_m = pickle.load(open(path + 'columnModelling.pkl', 'rb'))

        X = data[col_p]
        colEncoder, colpOneHotEncoder, colStandarScaler = InsurancePre().colPreparation()
        for col in X.columns:
            prep = pickle.load(open(path + 'prep' + col + '.pkl', 'rb'))
            if col in colpOneHotEncoder:
                dfTemp = pd.DataFrame(prep.transform(X[[col]]).toarray())
                X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
            else:
                dfTemp = pd.DataFrame(prep.transform(X[[col]]))
                X = pd.concat([X.drop(col, axis=1), dfTemp], axis=1)
        X = X.values.ravel()  # Convert X to 1-dimensional array

        # Create a new DataFrame from the 1-dimensional array and assign column names
        X = pd.DataFrame(X.reshape(1, -1), columns=col_m)

        # Convert DataFrame X back to a 2-dimensional array
        X = X.values

        if typed == 'multi':
            y = model.predict(X)
            return y
        
        elif typed == 'single':
            y = model.predict(X)[0] 
            if y == 0:
                return 0
            else:
                return 1
        else:
            return False



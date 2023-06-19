import joblib
import sklearn

loaded_model = joblib.load("parkinson_svm.pkl")


test = [-1.057164, - 0.867762, - 0.293394,	0.807620, 0.628185, -
        0.409101, - 0.227295,	0.460490,	1.862257,	1.347772,	0.206250, - 1.204049]
pred = loaded_model.predict(test)

print(pred)

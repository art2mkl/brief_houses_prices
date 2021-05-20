from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import numpy as np
import pandas as pd


class Model:
    def __init__(self, df, target, features):
        tab = [target]
        for i in features:
            tab.append(str(i))
        self.df = df[tab]
        self.x = features
        self.y = target
        self.sk_fit = ''

    def cut_df(self):
        display(self.df.head(1))
        self.df = self.df.loc[np.random.permutation(self.df.index)].reset_index(drop=True)
        display(self.df.head(1))
        cut = round(len(self.df) * 0.80)
        self.train = self.df.iloc[:cut, :]
        self.test = self.df.iloc[cut:, :]

    def sm_regression(self):
        feature = ' + '.join(self.x)
        model = smf.ols(f'{self.y} ~ {feature}', data=self.df)
        resultats = model.fit()
        return resultats.summary()

    def sk_regression(self, df):
        modeleReg=LinearRegression()
        modeleReg.fit(df[self.x],df[self.y])
        print(f'La constante est égale à : {modeleReg.intercept_}')
        print(f'Les coefficients directeurs sont respectivement de : {modeleReg.coef_}')
        Rcarre = modeleReg.score(df[self.x],df[self.y])
        print(f'le R² est de : {Rcarre}')
        self.sk_fit = modeleReg

    def sk_predict(self, feature):
        a_predir = pd.DataFrame()
        for i in range(0, len(self.x)):
            a_predir[self.x[i]] = feature[i]
        return self.sk_fit.predict(a_predir)

    def sk_predict_test(self, df):
        new_df = df.drop(self.y, axis=1)
        comparaison = pd.DataFrame({'Valeur Attendu': df[self.y], 'Prédiction': self.sk_fit.predict(new_df)})
        print(f"""R² = {round(self.sk_fit.score(new_df[self.x],df[self.y]),2)}""")
        return comparaison

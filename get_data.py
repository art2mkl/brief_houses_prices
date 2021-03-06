import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


class GetData:
    def __init__(self, csv):
        self.df = pd.read_csv(csv)

    def df_info(self):
        len_df = len(self.df)
        all_columns = len(self.df.columns)
        all_nan = self.df.isna().sum().sum()

        print(f"""
        Longueur du dataset : {len_df} enregistrements
        Nombre de colonnes : {all_columns}
        Nombre total de celulles non nulles : {self.df.notna().sum().sum()}
        Nombre total de cellules nulles : {all_nan}
        soit {round(all_nan/(len_df*all_columns) * 100,2)} %
        """)

        echantillonColonnes = []
        for i in self.df.columns:
            listcolumn = str(list(self.df[i].head(5)))
            echantillonColonnes.append(listcolumn)
            echantillonColonnes[0:5]
        obs = pd.DataFrame({'colonne': list(self.df.columns),
                            'type': list(self.df.dtypes),
                            'Echantillon': echantillonColonnes,
                            "% de valeurs nulles":
                            round(self.df.isna().sum() / len_df * 100, 2)})
        return obs

    def f_look(self, target):
        print(f'Analyse de la feature {target}')
        display(self.df[target].describe())
        plt.figure(figsize=(12, 8))
        sns.histplot(data=self.df, x=target, kde=True).set_title(f'Distribution de {target}')

    def df_corr(self, target):
        df_corr = self.df.corr()
        corr_target = df_corr.sort_values(target, ascending=False)
        plt.figure(figsize=(12, 8))
        sns.barplot(x=corr_target[target], y=corr_target.index).set_title(f'Correlation des features vis à vis de {target}')
        plt.show()

    def kompare(self, x, y):
        plot = sns.lmplot(x=x, y=y, data=self.df, height=5, aspect=2)
        plot.fig.suptitle(f'Comparaison de {x} en fonction de {y}')
        plt.show()

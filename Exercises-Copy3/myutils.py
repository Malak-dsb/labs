
def plot_transformations(df, skew_table):
    """
    Applies the recommended transformation to each column, then plots the before and after 
    distributions side by side with the skewness degree on each subplot.
    """
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import boxcox
from sklearn.preprocessing import PowerTransformer

def plot_transformations(df, skew_table):

    for _, row in skew_table.iterrows():

        feature = row['Feature']
        recommendation = row['Recommended Transformation']

        original = df[feature].dropna()

        transformed = original.copy()

        try:

            if recommendation == 'Box-Cox or Yeo-Johnson':

                if original.min() > 0:
                    transformed, _ = boxcox(original)

                else:
                    pt = PowerTransformer(method='yeo-johnson')
                    transformed = pt.fit_transform(
                        original.values.reshape(-1,1)
                    ).flatten()

            elif recommendation == 'log(x+1) or Yeo-Johnson':

                transformed = np.log1p(original)

        except:
            transformed = original

        original_skew = original.skew()
        transformed_skew = pd.Series(transformed).skew()

        fig, axes = plt.subplots(1, 2, figsize=(10,4))

        axes[0].hist(original, bins=20)
        axes[0].set_title(
            f'Original {feature}\n(Skew: {original_skew:.2f})'
        )

        axes[1].hist(transformed, bins=20)
        axes[1].set_title(
            f'Best transformation: {recommendation.split()[0]} for {feature}\n'
            f'(Skew: {transformed_skew:.2f})'
        )

        plt.tight_layout()
        plt.show()

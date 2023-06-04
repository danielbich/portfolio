import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def dateparser(date):
    """
    Konwersja ciągu znaków na obiekt daty
    """
    date_object = datetime.datetime.strptime(date, "%d.%m.%Y")
    # Konwersja daty na ciąg znaków w oczekiwanym formacie
    formatted_date = date_object.strftime("%Y-%m-%d")
    return formatted_date

def plot_cena(df, market, artykul):
    """
    Fukcja tworzy dwa scatterploty dla wybranej kombinacji 'market & artykuł': cena oraz ilość sprzedaży.
    """
    filtered_df = df[(df['MARKET_ID'] == market) & (df['ART_ID'] == artykul)]
    colors = ['#FF5733', '#33FF57', '#3357FF']

    # Set up subplots
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    # Wykres 1
    sns.scatterplot(ax=axes[0], x=filtered_df['DATA'], y=filtered_df['CENA_MARKET'], color=colors[0], s=30, label='CENA_MARKET')
    axes[0].set_title("CENY")
    axes[0].tick_params(axis='x', rotation=45)

    # Wykres 2
    sns.scatterplot(ax=axes[0], x=filtered_df['DATA'], y=filtered_df['CENA_AP'], color=colors[1], s=30, label='CENA_AP')

    # Wykres 3
    sns.scatterplot(ax=axes[0], x=filtered_df['DATA'], y=filtered_df['CENA_NP'], color=colors[2], s=30, label='CENA_NP')

    # Wykres 4
    sns.scatterplot(ax=axes[1], x=filtered_df['DATA'], y=filtered_df['y'], color=colors[2], s=30, label='ILOŚĆ')
    axes[1].set_title("ILOŚC")

    fig.suptitle('Porównanie cen vs sprzedaż dla marketu: ' + str(market) + " i artykułu: " + str(filtered_df['NAZWA'].iloc[0]))
    plt.xticks(rotation=45)

    # Wyświetl legendę
    plt.legend()

    # Wyświetl wykres
    plt.show()


def MASE(y_true, y_predict):
    """
    Oblicza wartość MASE (Mean Absolute Scaled Error) na podstawie prawdziwych wartości `y_true` i przewidywanych wartości `y_predict`.
    """
    n = len(y_true)
    error = sum(abs(y_true[i] - y_predict[i]) for i in range(n))
    mean_absolute_error = error / n

    naive_error = sum(abs(y_true[i] - y_true[i-1]) for i in range(1, n))
    mean_absolute_naive_error = naive_error / (n - 1)

    if mean_absolute_naive_error == 0:
        return 0

    mase = mean_absolute_error / mean_absolute_naive_error
    return mase

def WMAPE(y_true, y_predict):
    """
    Oblicza wartość WMAPE (Weighted Mean Absolute Percentage Error) na podstawie prawdziwych wartości `y_true` i przewidywanych wartości `y_predict`.
    """
    non_zero_mask = y_true != 0
    y_true_non_zero = y_true[non_zero_mask]
    y_predict_non_zero = y_predict[non_zero_mask]
    absolute_errors = np.abs(y_true_non_zero - y_predict_non_zero)
    weighted_errors = absolute_errors / np.abs(y_true_non_zero)
    return weighted_errors.mean()


def metryki(y_true, y_predict):
    """
    Oblicza różne metryki porównujące prawdziwe wartości `y_true` z przewidywanymi wartościami `y_predict`.

    Zwraca:
    - Tuple zawierający różne metryki: MAE, RMSE, R2, WMAPE, MASE
    """
    # Obliczenie MAE
    mae = mean_absolute_error(y_true, y_predict)

    # Obliczenie RMSE
    rmse = mean_squared_error(y_true, y_predict, squared=False)

    # Obliczenie R2
    r2 = r2_score(y_true, y_predict)

    # Obliczenie WMAPE
    wmape = WMAPE(y_true, y_predict)

    # Obliczenie MASE (o ile lepsze względem predykcji naiwnej)
    mase = MASE(y_true, y_predict)

    return mae, rmse, r2, wmape, mase

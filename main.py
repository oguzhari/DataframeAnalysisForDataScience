# Veri setinin eklenmesi
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

dataframe = sns.load_dataset('titanic')


# Veri seti için ayrı bir dosya yükleme ihtiyacı olmaması adına, seaborn kütüphanesinden bir veri veri seti alındı.
# İhtiyacınıza göre dataframe = kısmını düzenleyerek bütün programı sorunsuz bir şekilde çalıştırabilirsiniz.


def grab_col_names(df, cat_th=10, car_th=20) -> [list, list, list]:
    """
    Tasks
    ----------
    Analiz edilmek istenen dataframe için bir analiz fonksiyonu. Bu fonksiyon, almış olduğu dataframe'in numerik,
    kategorik ve kardinal sütunlarını ayrı ayrı olacak şekilde döndürecektir.

    Parameters
    ----------
    df: pandas.Dataframe
        İncelenmek istenilen pandas.Dataframe'i
    cat_th: int, float default=20
        numeric olan sütunlar için categoric varsayılma eşiği. bu eşiğin altında bulunan int64 ve float64
        veri tipindeki sütunlar numeric değil, categoric olarak işlenecek.
    car_th: int, float default=10
        categoric olan sütunlar için cardinal varsayılma eşiği. bu eşiğin altında bulunan category, object ve bool
        veri tipindeki sütunlar categoric değil, cardinal olarak işlenecek.

    Returns
    -------
    numeric_cols: list
        numeric_cols, veri tipi "int64" ve "float64" veri tiplerine sahip ancak categorical_cols içinde bulunmayan
        sütunların listesi.
    categorical_cols: list
        categorical_cols, veri tipi "category", "object" ve "bool" olan,ancak kardinal olmayan ve
        veri tipi int64 ve float64 olup cat_th'dan daha az unique veri tipine sahip sütunların listesi.
    categorical_but_cardinal: list
        categorical_but_cardinal, veri tipi "category", "object" ve "bool" olancar_th değerinden daha fazla unique
        kategori içeren sütunların listesi.
    Observations: int
        Toplam satır sayısı.
    Variable: int
        Toplam sütun sayısı.
    cat_cols: int
        kategorik sütunların sayısı.
    num_cols: int
        numerik sütunların sayısı.
    cat_but_car: int
        kategorik ama kardinal olan sütunların sayısı.
    num_but_cat: int
        numerik ama kategorik olan sütunların sayısı.
    """
    categorical_cols = [col for col in df.columns if str(df[col].dtype) in ["category", "object", "bool"]]
    numeric_but_categorical = [col for col in df.columns if df[col].nunique() < cat_th
                               and str(df[col].dtypes) in ["int64", "float64"]]
    categorical_but_cardinal = [col for col in df.columns if df[col].nunique() > car_th
                                and str(df[col].dtypes) in ["category", "object"]]
    categorical_cols = categorical_cols + numeric_but_categorical
    categorical_cols = [col for col in categorical_cols if col not in categorical_but_cardinal]

    numeric_cols = [col for col in df.columns if df[col].dtype in ["int64", "float64"]]
    numeric_cols = [col for col in numeric_cols if col not in categorical_cols]

    print(f"Observations -> {dataframe.shape[0]}")
    print(f"Variable     -> {dataframe.shape[1]}")
    print(f"cat_cols     -> {len(categorical_cols)}")
    print(f"num_cols     -> {len(numeric_cols)}")
    print(f"cat_but_car  -> {len(categorical_but_cardinal)}")
    print(f"num_but_cat  -> {len(numeric_but_categorical)}")

    return numeric_cols, categorical_cols, categorical_but_cardinal


def cat_summary(df, categorical_cols, plot=False):
    """
    Tasks
    -----
    Veri setindeki kategorik sütunlara ait verileri hakkında bilgiler veren fonksiyon.

    Parameters
    ----------
    df: pandas.DataFrame
        İncelenmek istenilen pandas.Dataframe'i
    categorical_cols: list
        Kategorik sütunların olduğu liste
    plot: bool, default=False
        true ise, bir plot döndürür.

    Returns
    -------
    col:
        Sütun içerisinde bulunan benzersiz değerlerin listesi.
    Ratio:
        Bu değerlerin tüm veri setine dağılımı, değerler toplamı 1~ olmalı.
    """
    for col in categorical_cols:
        if (df[col].dtypes == "category" or df[col].dtypes == "bool") and df[col].nunique() == 2:
            df[col] = df[col].astype(int)
        print("col type:", str(df[col].dtype))
        print(pd.DataFrame({col: df[col].value_counts(),
                            "Ratio": df[col].value_counts(normalize=True)}))
        print("#" * 23)
        if plot:
            sns.countplot(x=df[col], data=df)
            plt.show(block=True)


def num_summary(df, numerical_col, plot=False):
    """
    Tasks
    -----
    nümerik sütunlara ait describe() bilgilerini ve plot=True ise grafik bilgisini döndüren kod.

    Parameters
    ----------
    df: pandas.DataFrame
        İncelenmek istenilen pandas.Dataframe'i
    numerical_col: list
        Nümerik sütunlara ait liste.
    plot: bool, default=False
        True olması halinde söz konusu sütunlara ait histogram'ın döndürülmesini sağlar.

    Returns
    -------
    Sütunlara ait describe bilgisini ve plot=True ise histogram grafiğini döndürür.
    """
    for col in numerical_col:
        print(df[col].describe().T)
        if plot:
            dataframe[col].hist()
            plt.xlabel(col)
            plt.title(col)
            plt.show(block=True)


def check_df(df, head=5):
    print("##### Shape #####")
    print(df.shape)
    print("\n########### Types ###########")
    print(df.dtypes)
    print("\n################################ Head ################################")
    print(df.head(head))
    print("\n################################ Tail ################################")
    print(df.tail(head))
    print("\n######### NA #########")
    print(df.isnull().sum())
    print("\n################################ Quantiles ################################")
    print(df.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)


# Run
num_cols, cat_cols, car_cols = grab_col_names(dataframe)
check_df(dataframe)


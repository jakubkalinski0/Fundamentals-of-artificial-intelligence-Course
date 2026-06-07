# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Fundamentals-of-artificial-intelligence
#     language: python
#     name: python3
# ---

# %% [markdown] id="ogl0jjdkLra0" pycharm={"name": "#%% md\n"}
# # Klasyfikacja niezbalansowana, klasyfikatory zespołowe i wyjaśnialna AI

# %% [markdown] id="vcs8jkS4Lra3" pycharm={"name": "#%% md\n"}
# ## Wykorzystanie Google Colab
#
# Jeśli korzystasz z Google Colab skopiuj plik `feature_names.json` do katalogu głównego projektu.
#
# Pamiętaj o zainstalowaniu zależności z użyciem `uv` lub `pip`.

# %% [markdown] editable=true id="E4k6mouGLra4" pycharm={"name": "#%% md\n"}
# ## Ładowanie i eksploracja danych

# %% [markdown] editable=true id="tYT6eA4_Lra5" pycharm={"name": "#%% md\n"}
# W trakcie tych zajęć laboratoryjnych wykorzystamy zbiór danych [Polish companies bankruptcy](https://archive.ics.uci.edu/ml/datasets/Polish+companies+bankruptcy+data). Dotyczy on klasyfikacji, na podstawie danych z raportów finansowych, czy firma zbankrutuje w ciągu najbliższych kilku lat. Jest to zadanie szczególnie istotne dla banków, funduszy inwestycyjnych, firm ubezpieczeniowych itp. Zbiór zawiera 64 cechy obliczonych przez ekonomistów. Są one opisane na wspomnianej wcześniej stronie. Dotyczą one zysków firm, posiadanych zasobów, długów itp.
#
# Ściągnij i rozpakuj dane (`Data Folder` -> `data.zip`) do katalogu `data` obok tego notebooka. Znajduje się tam 5 plików w formacie `.arff`, wykorzystywanym głównie przez oprogramowanie Weka. Jest to program do wyposażony w graficzny interfejs użytkownika, który był często używany przez mniej techincznie obeznanych użytkowników. W Pythonie dane w tym formacie ładuje się  za pomocą bibliotek SciPy i Pandas.
#

# %% [markdown] id="mEmOG3ZELra5"
# Jeśli korzystasz z Linuksa możesz skorzystać z poniższych poleceń do pobrania i rozpakowania tych plików.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="-ejO83awLra5" outputId="14fed0eb-9f2a-4d86-a32c-6477528710ef"
# !mkdir -p data
# !wget https://archive.ics.uci.edu/static/public/365/polish+companies+bankruptcy+data.zip -O data/data.zip

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="Lz5_XXIPLra6" outputId="17e5de43-4462-422d-91ed-516688a9eccb"
# !unzip data/data.zip -d data

# %% [markdown] editable=true id="pzLskMUMLra6"
#
# W dalszej części laboratorium wykorzystamy plik `3year.arff`, w którym na podstawie danych finansowych firmy po 3 latach monitorowania chcemy przewidywać, czy firma zbankrutuje w ciągu najbliższych 3 lat. Jest to dość realistyczny horyzont czasowy.
#
# Dodatkowo w pliku `feature_names.json` znajdują się nazwy cech. Nazwy są bardzo długie, więc póki co nie będziemy z nich korzystać.

# %% editable=true id="mUT_JwgGLra6" pycharm={"name": "#%%\n"}
import json
import os

from scipy.io import arff
import pandas as pd

data = arff.loadarff(os.path.join("data", "3year.arff"))

with open("feature_names.json") as file:
    feature_names = json.load(file)

X = pd.DataFrame(data[0])

# %% [markdown] id="5CLkXIkqLra7" pycharm={"name": "#%% md\n"}
# Przyjrzyjmy się teraz naszym danym.

# %% colab={"base_uri": "https://localhost:8080/", "height": 255} editable=true id="Hd8taosQLra7" outputId="d1376e3d-283f-49dc-8ea8-b991d94c45b5" pycharm={"name": "#%%\n"}
X.head()

# %% colab={"base_uri": "https://localhost:8080/", "height": 458} editable=true id="8N4w5RvWLra7" outputId="370fdbb4-4906-4dbf-eacf-a926d01ce019" pycharm={"name": "#%%\n"}
X.dtypes

# %% colab={"base_uri": "https://localhost:8080/", "height": 349} id="mMbAC_0JLra7" outputId="6665014d-1fb4-43e2-eb1f-6efc9fbb9a6c" pycharm={"name": "#%%\n"}
X.describe()

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="DCTthGKsLra7" outputId="d2e9680b-95ae-449a-f0c9-469b8f4846aa"
feature_names

# %% [markdown] editable=true id="EDa4hFBNLra8" pycharm={"name": "#%% md\n"}
# DataFrame zawiera 64 atrybuty numeryczne o zróżnicowanych rozkładach wartości oraz kolumnę `"class"` typu `bytes` z klasami 0 i 1. Wiemy, że mamy do czynienia z klasyfikacją binarną - klasa 0 to brak bankructwa, klasa 1 to bankructwo w ciągu najbliższych 3 lat. Przyjrzyjmy się dokładniej naszym danym.

# %% [markdown] editable=true id="71yNyB8BLra8" tags=["ex"]
# ### Zadanie 1 (0.5 punktu)

# %% [markdown] editable=true id="GldGsxEBLra8" tags=["ex"]
# 1. Wyodrębnij klasy jako osobną zmienną typu `pd.Series`, usuwając je z macierzy `X`. Przekonwertuj je na liczby całkowite.
# 2. Narysuj wykres słupkowy częstotliwości obu klas w całym zbiorze. Upewnij się, że na osi X są numery lub nazwy klas, a oś Y ma wartości w procentach.
#
# **Uwaga:** sugerowane jest użycie `if` w podpunkcie 1, żeby można było tę komórkę bezpiecznie odpalić kilka razy.

# %% colab={"base_uri": "https://localhost:8080/", "height": 857} editable=true id="6cf4c_rnLra8" outputId="98bce266-5645-43e1-edfb-a09e8b7269b4" pycharm={"name": "#%%\n"} tags=["ex"]
y = ''
if 'class' in X.columns:
  y = X['class'].copy()
  X = X.drop(columns=['class'])

import matplotlib.pyplot as plt

y = y.astype('category').cat.codes

counts = y.value_counts(normalize=True) * 100
plt.figure(figsize=(6, 4))
plt.bar(counts.index, counts.values)
plt.xticks(counts.index)
plt.ylabel('Udział [%]')
plt.xlabel('Klasy')
plt.title('Rozkład klas w zbiorze danych')
plt.show()

counts.plot.bar(title='Rozkład klas w zbiorze danych')

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="V_C1P5w6Lra8" outputId="52470360-8bbd-4e9d-f282-34411dd1a4f1" tags=["ex"]
assert "class" not in X.columns

print("Solution is correct!")

# %% [markdown] editable=true id="kuBBMWzuLra8" pycharm={"name": "#%% md\n"}
# Jak widać, klasa pozytywna jest w znacznej mniejszości, stanowi poniżej 5% zbioru. Taki problem nazywamy **klasyfikacją niezbalansowaną (imbalanced classification)**. Mamy tu **klasę dominującą (majority class)** oraz **klasę mniejszościową (minority class)**. Pechowo prawie zawsze interesuje nas ta druga, bo klasa większościowa nie niesie najczęściej żadnych interesujących informacji. Przykładowo, 99% badanych jest zdrowych, a 1% ma niewykryty nowotwór - z oczywistych przyczyn chcemy wykrywać właśnie sytuację rzadką (problem diagnozy jako klasyfikacji jest zasadniczo zawsze niezbalansowany). W dalszej części laboratorium poznamy szereg konsekwencji tego zjawiska i metody na radzenie sobie z nim.
#
# Mamy sporo cech w naszym zbiorze, wszystkie są numeryczne. Ciekawe, czy mają wartości brakujące, a jeśli tak, to ile? Policzymy to z pomocą biblioteki Pandas i metody `.isna()`. Domyślnie operuje ona na kolumnach, jak większość metod w w tej bibliotece. Sumę wartości per kolumna zwróci nam metoda `.sum()`. Jeżeli podzielimy to przez liczbę wierszy `len(X)`, to otrzymamy ułamek wartości brakujących w każdej kolumnie.
#
# Biblioteka Pandas potrafi też stworzyć wykres, z pomocą funkcji np. `.plot.hist()` czy `.plot.bar()`. Przyjmują one opcje formatowania wykresu z których korzysta biblioteka `matplotlib`.

# %% colab={"base_uri": "https://localhost:8080/", "height": 514} editable=true id="Mn2CFfnBLra8" outputId="7cc0ad15-49b4-4787-dee5-0a5ee3559172" pycharm={"name": "#%%\n"}
na_perc = X.isna().sum() / len(X)
na_perc.plot.bar(title="Fraction of missing values per column", figsize=(15, 5))

# %% [markdown] editable=true id="h1iyqM7oLra8" pycharm={"name": "#%% md\n"} tags=["ex"]
# Jak widać, cecha 37 ma bardzo dużo wartości brakujących, podczas gdy pozostałe cechy mają raczej niewielką ich liczbę. W takiej sytuacji najlepiej usunąć tę cechę, a pozostałe wartości brakujące **uzupełnić** (co realizowaliśmy już poprzednio). Pamiętaj, że imputacji dokonuje się dopiero po podziale na zbiór treningowy i testowy! W przeciwnym wypadku wykorzystywalibyśmy dane ze zbioru testowego, co sztucznie zawyżyłoby wyniki. Jest to błąd metodologiczny - **wyciek danych (data leakage)**.
#
# Podział na zbiór treningowy i testowy to pierwszy moment, kiedy niezbalansowanie danych nam przeszkadza. Jeżeli zrobimy to czysto losowo, to jest spora szansa, że w zbiorze testowym będzie tylko klasa negatywna - w końcu jest jej aż >95%. Dlatego wykorzystuje się **próbkowanie ze stratyfikacją (stratified sampling)**, dzięki któremu proporcje klas w zbiorze przed podziałem oraz obu zbiorach po podziale są takie same.

# %% [markdown] editable=true id="NMlMULUfLra9" tags=["ex"]
# ### Zadanie 2 (0.75 punktu)

# %% [markdown] editable=true id="vBYbu8pmLra9" tags=["ex"]
# 1. Usuń kolumnę `"Attr37"` ze zbioru danych.
# 2. Dokonaj podziału zbioru na treningowy i testowy w proporcjach 80%-20%, z przemieszaniem (`shuffle`), ze stratyfikacją, wykorzystując funkcję `train_test_split` ze Scikit-learn'a.
# 3. Uzupełnij wartości brakujące średnią wartością cechy z pomocą klasy `SimpleImputer`.
#
# **Uwaga:**
# - jak wcześniej, sugerowane jest użycie `if` w podpunkcie 1,
# - pamiętaj o uwzględnieniu stałego ziarna `random_state=0`, aby wyniki były **reprodukowalne (reproducible)**,
# - `stratify` oczekuje wektora klas,
# - wartości do imputacji trzeba wyestymować na zbiorze treningowym (`.fit()`), a potem zastosować te nauczone wartości na obu podzbiorach (treningowym i testowym).

# %% editable=true id="CkZ1oQ1fLra9" pycharm={"name": "#%%\n"} tags=["ex"]
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

if 'Attr37' in X.columns:
    X = X.drop(columns=['Attr37'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=0,
    shuffle=True,
    stratify=y
)

imputer = SimpleImputer(strategy='mean')
imputer.fit(X_train)

X_train = pd.DataFrame(imputer.transform(X_train), columns=X_train.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)



# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="cNp5rj78Lra9" outputId="9c7e7991-de72-470f-d915-67a7b7a2c826" tags=["ex"]
import numpy as np

assert "Attr37" not in X.columns
assert not np.any(np.isnan(X_train))
assert not np.any(np.isnan(X_test))

print("Solution is correct!")

# %% [markdown] editable=true id="V2ANlZcTLra9" pycharm={"name": "#%% md\n"}
# ## Prosta klasyfikacja

# %% [markdown] editable=true id="Naxp4M0QLra9" pycharm={"name": "#%% md\n"}
# Zanim przejdzie się do modeli bardziej złożonych, trzeba najpierw wypróbować coś prostego, żeby mieć punkt odniesienia. Tworzy się dlatego **modele bazowe (baselines)**.
#
# W naszym przypadku będzie to **drzewo decyzyjne (decision tree)**. Jest to drzewo binarne z decyzjami if-else, prowadzącymi do klasyfikacji danego przykładu w liściu. Każdy podział w drzewie to pytanie postaci "Czy wartość cechy X jest większa lub równa Y?". Trening takiego drzewa to prosty algorytm zachłanny, bardzo przypomina budowę zwykłego drzewa binarnego. Ma on następujące kroki dla każdego węzła tego drzewa:
# 1. Sprawdź po kolei wszystkie możliwe punkty podziału, czyli każdą (unikalną) wartość każdej cechy, po kolei.
# 2. Dla każdego przypadku podziel zbiór na 2 części: niespełniający warunku (lewy potomek) i spełniający warunek (prawy potomek).
# 3. Oblicz jakość podziału według wybranej funkcji jakości. Im lepiej warunek rozdziela klasy od siebie (imbardziej zunifikowane są węzły-dzieci), tym wyższa jakość podziału. Innymi słowy, chcemy, żeby do jednego dziecka trafiła jedna klasa, a do drugiego druga.
# 4. Wybierz podział o najwyższej jakości.
#
# Taki algorytm wykonuje się rekurencyjnie, aż otrzymamy węzeł czysty (pure leaf), czyli taki, w którym są przykłady z tylko jednej klasy. Typowo wykorzystywaną funkcją jakości (kryterium podziału) jest entropia Shannona - im niższa entropia, tym bardziej jednolite są klasy w węźle (czyli wybieramy podział o najniższej entropii).
#
# Powyższe wytłumaczenie algorytmu jest oczywiście nieformalne i dość skrótowe. Doskonałe tłumaczenie, z interaktywnymi wizualizacjami, dostępne jest [tutaj](https://mlu-explain.github.io/decision-tree/). W formie filmów - [tutaj](https://www.youtube.com/watch?v=ZVR2Way4nwQ) oraz [tutaj](https://www.youtube.com/watch?v=_L39rN6gz7Y). Dla drzew do regresji - [ten film](https://www.youtube.com/watch?v=g9c66TUylZ4).
#
# <img src = https://miro.medium.com/max/1838/1*WyTsLwcAXivFCgNtF0OPqA.png width = "642" height = "451" >
#
# Warto zauważyć, że taka konstrukcja prowadzi zawsze do overfittingu. Otrzymanie liści czystych oznacza, że mamy 100% dokładności na zbiorze treningowym, czyli perfekcyjnie przeuczony klasyfikator. W związku z tym nasze predykcje mają bardzo niski bias, ale bardzo dużą wariancję. Pomimo tego drzewa potrafią dać bardzo przyzwoite wyniki, a w celu ich poprawy można je regularyzować, aby mieć mniej "rozrośnięte" drzewo. [Film dla zainteresowanych](https://www.youtube.com/watch?v=D0efHEJsfHo).
#

# %% [markdown] editable=true id="xRlMJezXLra9"
# Mając wytrenowany klasyfikator, trzeba oczywiście sprawdzić, jak dobrze on sobie radzi. Tu natrafiamy na kolejny problem z klasyfikacją niezbalansowaną - zwykła celność (accuracy) na pewno nie zadziała! Typowo wykorzystuje się AUC, nazywane też AUROC (Area Under Receiver Operating Characteristic), bo metryka ta uwzględnia niezbalansowanie klas.
#
# Bardzo dobre i bardziej szczegółowe wytłumaczenie, z interktywnymi wizualizacjami, można znaleć [tutaj](https://mlu-explain.github.io/roc-auc/). Dla preferujących filmy - [tutaj](https://www.youtube.com/watch?v=4jRBRDbJemM).
#
# Co ważne, z definicji AUROC, trzeba w niej użyć **prawdopodobieństw klasy pozytywnej** (klasy 1). W Scikit-learn'ie zwraca je metoda `.predict_proba()`, która w kolejnych kolumnach zwraca prawdopodobieństwa poszczególnych klas.

# %% [markdown] editable=true id="zF510ONqLra9" tags=["ex"]
# ### Zadanie 3 (0.75 punktu)

# %% [markdown] editable=true id="d5KeohHrLra9" tags=["ex"]
# 1. Wytrenuj klasyfikator drzewa decyzyjnego (klasa `DecisionTreeClassifier`). Użyj entropii jako kryterium podziału.
# 2. Oblicz i wypisz AUROC na zbiorze testowym dla drzewa decyzyjnego (funkcja `roc_auc_score`).
# 3. Skomentuj wynik - czy twoim zdaniem osiągnięty AUROC to dużo czy mało, biorąc pod uwagę możliwy zakres wartości tej metryki?
#
# **Uwaga:**
# - pamiętaj o użyciu stałego ziarna `random_state=0`,
# - jeżeli drzewo nie wyświetli się samo, użyj `plt.show()` z Matplotliba,
# - pamiętaj o tym, żeby przekazać do metryki AUROC **prawdopodobieństwa klasy pozytywnej**, a nie binarne predykcje!

# %% colab={"base_uri": "https://localhost:8080/", "height": 538} editable=true id="-1hIGJbDLra9" outputId="4db1ce07-036a-49eb-da06-71f40944a6a2" pycharm={"name": "#%%\n"} tags=["ex"]
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(criterion='entropy', random_state=0)
dt.fit(X_train, y_train)

y_proba = dt.predict_proba(X_test)[:, 1]

auroc_dt = roc_auc_score(y_test, y_proba)
print(f"AUROC: {auroc_dt:.3f}")

plt.figure(figsize=(12, 6))
plot_tree(dt, filled=True, feature_names=X_train.columns, class_names=True)
plt.title("Drzewo decyzyjne")
plt.show()


# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="0uE4dc3LLra-" outputId="498bf758-efc7-4885-afe2-da16366a0117" tags=["ex"]
assert auroc_dt > 0.7

print("Solution is correct!")

# %% [markdown] editable=true id="RtwTeKWhLra-" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Wynik 0.727 to całkiem przyzwoity rezultat jak na model bazowy.
#
# W skali od 0.5 (co oznacza losowe zgadywanie) do 1.0 (ideał), wynik w okolicy 0.73 pokazuje, że nasz klasyfikator nauczył się wykrywać istotne zależności w danych. Biorąc pod uwagę, że pojedyncze drzewo decyzyjne bez ograniczeń ma dużą skłonność do przeuczania się (overfittingu), taki poziom AUROC stanowi solidny punkt odniesienia.
#
# Mamy więc potwierdzenie, że w danych jest potencjał, ale jest też spore pole do popisu dla bardziej złożonych algorytmów, które powinny ten wynik wyraźnie poprawić.

# %% [markdown] editable=true id="2j5REA1ILra-" pycharm={"name": "#%% md\n"}
# ## Uczenie zespołowe, bagging, lasy losowe

# %% [markdown] editable=true id="gt3IGuPlLra-" pycharm={"name": "#%% md\n"}
# Bardzo często wiele klasyfikatorów działających razem daje lepsze wyniki niż pojedynczy klasyfikator. Takie podejście nazywa się **uczeniem zespołowym (ensemble learning)**. Istnieje wiele różnych podejść do tworzenia takich klasyfikatorów złożonych (ensemble classifiers).
#
# Podstawową metodą jest **bagging**:
# 1. Wylosuj N (np. 100, 500, ...) próbek boostrapowych (boostrap sample) ze zbioru treningowego. Próbka boostrapowa to po prostu losowanie ze zwracaniem, gdzie dla wejściowego zbioru z M wierszami losujemy M próbek (czyli tyle ile było w początkowym zbiorze), spośród N wylosowanych próbek. Będą tam powtórzenia, średnio nawet 1/3, ale się tym nie przejmujemy.
# 2. Wytrenuj klasyfikator bazowy (base classifier) na każdej z próbek boostrapowych.
# 3. Stwórz klasyfikator złożony poprzez uśrednienie predykcji każdego z klasyfikatorów bazowych.
#
# <img src = https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Ensemble_Bagging.svg/440px-Ensemble_Bagging.svg.png width = "440" height = "248" >
#
# Typowo klasyfikatory bazowe są bardzo proste, żeby można było szybko wytrenować ich dużą liczbę. Prawie zawsze używa się do tego drzew decyzyjnych. Dla klasyfikacji uśrednienie wyników polega na głosowaniu - dla nowej próbki każdy klasyfikator bazowy ją klasyfikuje, sumuje się głosy na każdą klasę i zwraca najbardziej popularną decyzję.
#
# Taki sposób uczenia zmniejsza wariancję klasyfikatora. Intuicyjnie, skoro coś uśredniamy, to siłą rzeczy będzie mniej rozrzucone, bo dużo ciężej będzie osiągnąć jakąś skrajność. Redukuje to też overfitting.
#
# **Lasy losowe (Random Forests)** to ulepszenie baggingu. Zaobserwowano, że pomimo losowania próbek boostrapowych, w baggingu poszczególne drzewa są do siebie bardzo podobne (są skorelowane), używają podobnych cech ze zbioru. My natomiast chcemy zróżnicowania, żeby mieć niski bias - redukcją wariancji zajmuje się uśrednianie. Dlatego używa się metody losowej podprzestrzeni (random subspace method) - przy każdym podziale drzewa losuje się tylko pewien podzbiór cech, których możemy użyć do tego podziału. Typowo jest to pierwiastek kwadratowy z ogólnej liczby cech.
#
# Zarówno bagging, jak i lasy losowe mają dodatkowo bardzo przyjemną własność - są mało czułe na hiperparametry, szczególnie na liczbę drzew. W praktyce wystarczy ustawić 500 czy 1000 drzew i klasyfikator będzie dobrze działać. Dalsze dostrajanie hiperparametrów może jeszcze trochę poprawić wyniki, ale nie tak bardzo, jak przy innych klasyfikatorach. Jest to zatem doskonały wybór domyślny, kiedy nie wiemy, jakiego klasyfikatora użyć.
#
# Dodatkowo jest to problem **embarassingly parallel** - drzewa można trenować w 100% równolegle, dzięki czemu jest to dodatkowo wydajna obliczeniowo metoda.
#
# Głębsze wytłumaczenie, z interaktywnymi wizualizacjami, można znaleźć [tutaj](https://mlu-explain.github.io/random-forest/). Dobrze tłumaczy je też [ta seria filmów](https://www.youtube.com/watch?v=J4Wdy0Wc_xQ&t=480s).

# %% [markdown] editable=true id="QMBo5fOYLra-" tags=["ex"]
# ### Zadanie 4 (0.5 punktu)

# %% [markdown] editable=true id="EcVqNfETLra_" tags=["ex"]
# 1. Wytrenuj klasyfikator Random Forest (klasa `RandomForestClassifier`). Użyj 500 drzew i entropii jako kryterium podziału.
# 2. Sprawdź AUROC na zbiorze testowym.
# 3. Skomentuj wynik w odniesieniu do drzewa decyzyjnego.
#
# **Uwaga:** pamiętaj o ustawieniu `random_state=0`. Dla przyspieszenia ustaw `n_jobs=-1` (użyje tylu procesów, ile masz dostępnych rdzeni procesora). Pamiętaj też o przekazaniu prawdopodobieństw do metryki AUROC.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="Rp0aDk6nLra_" outputId="05a08ed1-a713-464d-9f42-2614b693f254" pycharm={"name": "#%%\n"} tags=["ex"]
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    random_state=0,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_proba_rf = rf.predict_proba(X_test)[:, 1]

auroc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"AUROC: {auroc_rf:.3f}")


# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="t6bY0fo4LrbE" outputId="a68d5d78-4a6a-4f85-80c5-e3359cb6b6e7" tags=["ex"]
assert auroc_rf > 0.85

print("Solution is correct!")

# %% [markdown] editable=true id="zGngo1e0LrbE" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Wynik Lasu Losowego jest wyraźnie lepszy w porównaniu do pojedynczego drzewa decyzyjnego (skok z 0.727 na 0.899).
#
# Wzrost wartości AUROC potwierdza w praktyce zalety uczenia zespołowego (ensemble learning). O ile pojedyncze drzewo miało tendencję do zbyt mocnego dopasowywania się do danych treningowych (overfitting) i "wkuwania" ich na pamięć, to Las Losowy skutecznie ten problem zniwelował.
#
# Dzięki uśrednieniu 500 drzew i losowaniu cech przy podziałach, udało się zredukować wariancję modelu. Otrzymaliśmy w ten sposób klasyfikator, który jest znacznie bardziej stabilny i lepiej generalizuje wiedzę, co bezpośrednio przełożyło się na wyższą skuteczność na zbiorze testowym. To pokazuje, że w tym przypadku uśrednienie wielu drzew faktycznie wygrywa z pojedynczym drzewem.

# %% [markdown] editable=true id="fhROifgKLrbE" pycharm={"name": "#%% md\n"}
# Jak zobaczymy poniżej, wynik ten możemy jednak jeszcze ulepszyć!

# %% [markdown] editable=true id="966ElhdPLrbE" pycharm={"name": "#%% md\n"}
# ## Oversampling, SMOTE

# %% [markdown] editable=true id="GARHZ5hMLrbE" pycharm={"name": "#%% md\n"}
# W przypadku zbiorów niezbalansowanych można dokonać **balansowania (balancing)** zbioru. Są tutaj 2 metody:
# - **undersampling**: usunięcie przykładów z klasy dominującej
# - **oversampling**: wygenerowanie dodatkowych przykładów z klasy mniejszościowej
#
# Undersampling działa dobrze, kiedy niezbalansowanie jest niewielkie, a zbiór jest duży (możemy sobie pozwolić na usunięcie jego części). Oversampling typowo daje lepsze wyniki, istnieją dla niego bardzo efektywne algorytmy. W przypadku bardzo dużego niezbalansowania można zrobić oba.
#
# Typowym algorytmem oversamplingu jest **SMOTE (Synthetic Minority Oversampling TEchnique)**. Działa on następująco:
# 1. Idź po kolei po przykładach z klasy mniejszościowej.
# 2. Znajdź `k` najbliższych przykładów dla próbki, typowo `k=5`.
# 3. Wylosuj tylu sąsiadów, ile trzeba do oversamplingu, np. jeżeli chcemy zwiększyć klasę mniejszościową 3 razy (o 200%), to wylosuj 2 z 5 sąsiadów.
# 4. Dla każdego z wylosowanych sąsiadów wylosuj punkt na linii prostej między próbką a tym sąsiadem. Dodaj ten punkt jako nową próbkę do zbioru.
#
# <img src = https://miro.medium.com/max/734/1*yRumRhn89acByodBz0H7oA.png >
#
# Taka technika generuje przykłady bardzo podobne do prawdziwych, więc nie zaburza zbioru, a jednocześnie pomaga klasyfikatorom, bo "zagęszcza" przestrzeń, w której znajduje się klasa pozytywna.
#
# Algorytm SMOTE, jego warianty i inne algorytmy dla problemów niezbalansowanych implementuje biblioteka Imbalanced-learn.

# %% [markdown] editable=true id="67ceiH1ELrbE" tags=["ex"]
# ### Zadanie 5 (1 punkt)

# %% [markdown] editable=true id="TJh3po5zLrbE" tags=["ex"]
# Użyj SMOTE do zbalansowania zbioru treningowego (nie używa się go na zbiorze testowym!). Implementuje to klasa `SMOTE`. Wytrenuj drzewo decyzyjne oraz las losowy na zbalansowanym zbiorze, użyj tych samych argumentów co wcześniej. Pamiętaj o użyciu wszędzie stałego ziarna `random_state=0` oraz przekazaniu prawdopodobieństw do AUROC. Skomentuj wynik.
#
# Wartość ROC drzewa decyzyjnego przypisz do zmiennej `tree_roc`, a lasu do `forest_roc`.

# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="rsZysa4JLrbF" outputId="abb83f09-1ac7-4ddf-f4c1-a992e06d0675" pycharm={"name": "#%%\n"} tags=["ex"]
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=0)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

tree_smote = DecisionTreeClassifier(criterion='entropy', random_state=0)
tree_smote.fit(X_train_smote, y_train_smote)

y_proba_tree = tree_smote.predict_proba(X_test)[:, 1]
tree_roc = roc_auc_score(y_test, y_proba_tree)

rf_smote = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    random_state=0,
    n_jobs=-1
)
rf_smote.fit(X_train_smote, y_train_smote)

y_proba_rf = rf_smote.predict_proba(X_test)[:, 1]
forest_roc = roc_auc_score(y_test, y_proba_rf)

print(f"AUROC drzewa decyzyjnego: {tree_roc:.3f}")
print(f"AUROC lasu losowego:     {forest_roc:.3f}")



# %% colab={"base_uri": "https://localhost:8080/"} editable=true id="7CwNQxmELrbF" outputId="d92bcb73-7ee1-44e2-df51-51526e8d1a5d" tags=["ex"]
assert 0.6 < tree_roc < 0.8
assert 0.8 < forest_roc < 0.95

print("Solution is correct!")

# %% [markdown] editable=true id="HiTTYgyFLrbF" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Zastosowanie techniki SMOTE przyniosło zróżnicowane efekty w zależności od użytego modelu.
#
# W przypadku pojedynczego drzewa decyzyjnego, wynik AUROC spadł (do 0.710). Drzewo bez ograniczeń (przez swoją "zachłanną" naturę) próbuje idealnie dopasować się do wszystkich punktów, w tym tych syntetycznych wygenerowanych przez SMOTE. Mogło to doprowadzić do powstania sztucznych, zbyt skomplikowanych granic decyzyjnych, co pogorszyło zdolność generalizacji na prawdziwych danych testowych.
#
# Natomiast dla Lasu Losowego wynik jest znakomity (0.905). Tutaj połączenie metod zadziałało idealnie. SMOTE dostarczył więcej przykładów klasy mniejszościowej, co pozwoliło modelowi lepiej "zrozumieć", czym ta klasa się charakteryzuje. Jednocześnie natura Lasu (uśrednianie wielu drzew) zniwelowała ryzyko przeuczenia się na syntetycznych punktach. Wynik powyżej 0.90 to już poziom bardzo dobrego klasyfikatora.

# %% [markdown] editable=true id="IJTlkIsLLrbF" pycharm={"name": "#%% md\n"}
# W dalszej części laboratorium używaj zbioru po zastosowaniu SMOTE do treningu klasyfikatorów.

# %% [markdown] editable=true id="5m3zNrQvLrbF" pycharm={"name": "#%% md\n"}
# ## Dostrajanie (tuning) hiperparametrów

# %% [markdown] editable=true id="IqVroejcLrbF" pycharm={"name": "#%% md\n"}
# Lasy losowe są stosunkowo mało czułe na dobór hiperparametrów - i dobrze, bo mają ich dość dużo. Można zawsze jednak spróbować to zrobić, a w szczególności najważniejszy jest parametr `max_features`, oznaczający, ile cech losować przy każdym podziale drzewa. Typowo sprawdza się wartości z zakresu `[0.1, 0.5]`.
#
# W kwestii szybkości, kiedy dostrajamy hiperparametry, to mniej oczywiste jest, jakiego `n_jobs` użyć. Z jednej strony klasyfikator może być trenowany na wielu procesach, a z drugiej można trenować wiele klasyfikatorów na różnych zestawach hiperparametrów równolegle. Jeżeli nasz klasyfikator bardzo dobrze się uwspółbieżnia (jak Random Forest), to można dać mu nawet wszystkie rdzenie, a za to wypróbowywać kolejne zestawy hiperparametrów sekwencyjnie. Warto ustawić parametr `verbose` na 2 lub więcej, żeby dostać logi podczas długiego treningu i mierzyć czas wykonania. W praktyce ustawia się to metodą prób i błędów.

# %% [markdown] editable=true id="LRcmTNoGLrbG" tags=["ex"]
# ### Zadanie 6 (1 punkt)

# %% [markdown] editable=true id="zQKF9_SSLrbG" tags=["ex"]
# 1. Dobierz wartość hiperparametru `max_features`:
#    - użyj grid search z 5 foldami,
#    - wypróbuj wartości `[0.1, 0.2, 0.3, 0.4, 0.5]`,
#    - wybierz model o najwyższym AUROC (argument `scoring`).
# 2. Sprawdź, jaka była optymalna wartość `max_features`. Jest to atrybut wytrenowanego `GridSearchCV`.
# 3. Skomentuj wynik. Czy warto było poświęcić czas i zasoby na tę procedurę?
# 4. Wynik przypisz do zmiennej `auroc`.
#
# **Uwaga:**
# - pamiętaj, żeby jako estymatora przekazanego do grid search'a użyć instancji Random Forest, która ma już ustawione `random_state=0` i `n_jobs`

# %% editable=true id="ZLF4kenRLrbG" pycharm={"is_executing": true, "name": "#%%\n"} tags=["ex"]
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

rf = RandomForestClassifier(
    n_estimators=500,
    criterion='entropy',
    random_state=0,
    n_jobs=-1
)

param_grid = {'max_features': [0.1, 0.2, 0.3, 0.4, 0.5]}

grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    scoring='roc_auc',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train_smote, y_train_smote)

best_rf = grid_search.best_estimator_
best_param = grid_search.best_params_['max_features']

y_proba_best = best_rf.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_proba_best)

print(f"Najlepsza wartość max_features: {best_param}")
print(f"AUROC: {auroc:.3f}")


# %% editable=true id="KQuERFxWLrbG" tags=["ex"]
assert 0.9 <= auroc <= 0.95

print("Solution is correct!")

# %% [markdown] editable=true id="GoYTDXc4LrbH" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Wynik poprawił się z 0.905 na 0.912.
#
# Odpowiedź na pytanie "czy było warto" jest niejednoznaczna. Z jednej strony w najbardziej zaawansowanych modelach walczy się o każdy ułamek skuteczności, gdzie taki przyrost jest cenny. Z drugiej strony, osiągnięcie go wymagało aż 25-krotnego wytrenowania dużego modelu (5 kandydatów × 5 foldów), co jest ogromnym kosztem obliczeniowym i czasowym w stosunku do marginalnego zysku.
#
# Eksperyment ten doskonale potwierdza teorię, że Lasy Losowe są wyjątkowo odporne na dobór hiperparametrów i działają świetnie już na ustawieniach domyślnych. Znaleziona optymalna wartość 0.2 (czyli używanie 20% cech przy podziale) jest zazwyczaj bardzo bliska domyślnej heurystyce (pierwiastek z liczby cech), co tłumaczy, dlaczego zysk z tuningu był niewielki.

# %% [markdown] editable=true id="gB2CVbHiLrbH" pycharm={"name": "#%% md\n"}
# W praktycznych zastosowaniach osoba trenująca model wedle własnego uznana, doświadczenia, dostępnego czasu i zasobów wybiera, czy dostrajać hiperparametry i w jak szerokim zakresie. Dla Random Forest na szczęście często może nie być znaczącej potrzeby i za to go lubimy :)

# %% [markdown] editable=true id="a5sLNV39LrbH" pycharm={"name": "#%% md\n"}
# **Random Forest - podsumowanie**
#
# 1. Model oparty o uczenie zespołowe.
# 2. Kluczowe elementy:
#    - bagging: uczenie wielu klasyfikatorów na próbkach boostrapowych,
#    - metoda losowej podprzestrzeni: losujemy podzbiór cech do każdego podziału drzewa,
#    - uśredniamy głosy klasyfikatorów.
# 3. Dość odporny na overfitting, zmniejsza wariancję błędu dzięki uśrednianiu.
# 4. Mało czuły na hiperparametry.
# 5. Przeciętnie daje bardzo dobre wyniki, doskonały wybór domyślny przy wybieraniu algorytmu klasyfikacji.

# %% [markdown] editable=true id="rwUOh3XILrbH" pycharm={"name": "#%% md\n"}
# ## Boosting

# %% [markdown] editable=true id="vtxlBCfuLrbI" pycharm={"name": "#%% md\n"}
# Drugą bardzo ważną grupą algorytmów ensemblingu jest **boosting**, też oparty o drzewa decyzyjne. O ile Random Forest trenował wszystkie klasyfikatory bazowe równolegle i je uśredniał, o tyle boosting robi to sekwencyjnie. Drzewa te uczą się na całym zbiorze, nie na próbkach boostrapowych. Idea jest następująca: trenujemy drzewo decyzyjne, radzi sobie przeciętnie i popełnia błędy na częsci przykładów treningowych. Dokładamy kolejne, ale znające błędy swojego poprzednika, dzięki czemu może to uwzględnić i je poprawić. W związku z tym "boostuje" się dzięki wiedzy od poprzednika. Dokładamy kolejne drzewa zgodnie z tą samą zasadą.
#
# Jak uczyć się na błędach poprzednika? Jest to pewna **funkcja kosztu** (błędu), którą chcemy zminimalizować. Zakłada się jakąś jej konkretną postać, np. squared error dla regresji, albo logistic loss dla klasyfikacji. Później wykorzystuje się spadek wzdłuż gradientu (gradient descent), aby nauczyć się, w jakim kierunku powinny optymalizować kolejne drzewa, żeby zminimalizować błędy poprzednika. Jest to konkretnie **gradient boosting**, absolutnie najpopularniejsza forma boostingu, i jeden z najpopularniejszych i osiągających najlepsze wyniki algorytmów ML.
#
# Tyle co do intuicji. Ogólny algorytm gradient boostingu jest trochę bardziej skomplikowany. Bardzo dobrze i krok po kroku tłumaczy go [ta seria filmów na YT](https://www.youtube.com/watch?v=3CC4N4z3GJc). Szczególnie ważne implementacje gradient boostingu to **XGBoost (Extreme Gradient Boosting)** oraz **LightGBM (Light Gradient Boosting Machine)**. XGBoost był prawdziwym przełomem w ML, uzyskując doskonałe wyniki i bardzo dobrze się skalując - był wykorzystany w CERNie do wykrywania cząstki Higgsa w zbiorze z pomiarów LHC mającym 10 milionów próbek. Jego implementacja jest dość złożona, ale dobrze tłumaczy ją [inna seria filmików na YT](https://www.youtube.com/watch?v=OtD8wVaFm6E).
#
# ![](xgboost.png)
#
# Obecnie najczęściej wykorzystuje się LightGBM. Został stworzony przez Microsoft na podstawie doświadczeń z XGBoostem. Został jeszcze bardziej ulepszony i przyspieszony, ale różnice są głównie implementacyjne. Różnice dobrze tłumaczy [ta prezentacja z konferencji PyData](https://www.youtube.com/watch?v=5CWwwtEM2TA) oraz [prezentacja Microsoftu](https://www.youtube.com/watch?v=5nKSMXBFhes). Dla zainteresowanych - [praktyczne aspekty LightGBM](https://www.kaggle.com/code/prashant111/lightgbm-classifier-in-python/notebook).

# %% [markdown] editable=true id="WWmD5QWALrbI" tags=["ex"]
# ### Zadanie 7 (0.5 punktu)

# %% [markdown] editable=true id="HXky0_KpLrbI" tags=["ex"]
# 1. Wytrenuj klasyfikator LightGBM (klasa `LGBMClassifier`). Przekaż `importance_type="gain"` - przyda nam się to za chwilę.
# 2. Sprawdź AUROC na zbiorze testowym.
# 3. Skomentuj wynik w odniesieniu do wcześniejszych algorytmów.
#
# Pamiętaj o `random_state`, `n_jobs` i prawdopodobieństwach dla AUROC.

# %% editable=true id="OomrvENyLrbI" pycharm={"is_executing": true, "name": "#%%\n"} tags=["ex"]
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score

lgbm = LGBMClassifier(
    random_state=0,
    n_jobs=-1,
    importance_type='gain'
)

lgbm.fit(X_train_smote, y_train_smote)

y_proba_lgbm = lgbm.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_proba_lgbm)

print(f"AUROC: {auroc:.3f}")



# %% editable=true id="_JxPs1R_LrbI" tags=["ex"]
assert 0.9 <= auroc <= 0.97

print("Solution is correct!")

# %% [markdown] editable=true id="xDR6LCAtLrbI" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Wynik AUROC 0.943 to najlepszy dotąd wynik i znaczący skok jakościowy.
#
# Przeskok o ponad 3 punkty procentowe względem najlepiej dostrojonego Lasu Losowego (0.912) to mimo wszystko duży przeskok. Widać tutaj wyraźną przewagę podejścia boostingu nad baggingiem. Podczas gdy Las Losowy "tylko" uśredniał decyzje niezależnych drzew, LightGBM aktywnie uczył się na błędach poprzedników, precyzyjnie korygując model tam, gdzie radził sobie gorzej.
#
# To doskonale pokazuje, że algorytmy oparte na Gradient Boostingu (jak XGBoost czy LightGBM) potrafią wycisnąć z danych znacznie więcej informacji niż klasyczne metody.

# %% [markdown] editable=true id="HMDO_Tz5LrbI" pycharm={"name": "#%% md\n"}
# Boosting dzięki uczeniu na poprzednich drzewach redukuje nie tylko wariancję, ale też bias w błędzie, dzięki czemu może w wielu przypadkach osiągnąć lepsze rezultaty od lasu losowego. Do tego dzięki znakomitej implementacji LightGBM jest szybszy.
#
# Boosting jest jednak o wiele bardziej czuły na hiperparametry niż Random Forest. W szczególności bardzo łatwo go przeuczyć, a większość hiperparametrów, których jest dużo, wiąże się z regularyzacją modelu. To, że teraz poszło nam lepiej z domyślnymi, jest rzadkim przypadkiem.
#
# W związku z tym, że przestrzeń hiperparametrów jest duża, przeszukanie wszystkich kombinacji nie wchodzi w grę. Zamiast tego można wylosować zadaną liczbę zestawów hiperparametrów i tylko je sprawdzić - chociaż im więcej, tym lepsze wyniki powinniśmy dostać. Służy do tego `RandomizedSearchCV`. Co więcej, klasa ta potrafi próbkować rozkłady prawdopodobieństwa, a nie tylko sztywne listy wartości, co jest bardzo przydatne przy parametrach ciągłych.
#
# Hiperparametry LightGBMa są dobrze opisane w oficjalnej dokumentacji: [wersja krótsza](https://lightgbm.readthedocs.io/en/latest/pythonapi/lightgbm.LGBMClassifier.html#lightgbm.LGBMClassifier) i [wersja dłuższa](https://lightgbm.readthedocs.io/en/latest/Parameters.html). Jest ich dużo, więc nie będziemy ich tutaj omawiać. Jeżeli chodzi o ich dostrajanie w praktyce, to przydatny jest [oficjalny przewodnik](https://lightgbm.readthedocs.io/en/latest/Parameters-Tuning.html) oraz dyskusje na Kaggle.

# %% [markdown] editable=true id="EvK-R2UFLrbI" tags=["ex"]
# ### Zadanie 8 (1.5 punktu)

# %% [markdown] editable=true id="uDhpCeGuLrbJ" tags=["ex"]
# 1. Zaimplementuj random search dla LightGBMa (klasa `RandomizedSearchCV`):
#    - użyj tylu prób, na ile pozwalają twoje zasoby obliczeniowe, ale przynajmniej 30,
#    - przeszukaj przestrzeń hiperparametrów:
#     ```
#     param_grid = {
#         "n_estimators": [100, 250, 500],
#         "learning_rate": [0.05, 0.1, 0.2],
#         "num_leaves": [31, 48, 64],
#         "colsample_bytree": [0.8, 0.9, 1.0],
#         "subsample": [0.8, 0.9, 1.0],
#     }
#     ```
# 2. Wypisz znalezione optymalne hiperparametry.
# 3. Wypisz raporty z klasyfikacji (funkcja `classification_report`), dla modelu LightGBM bez i z dostrajaniem hiperparametrów.
# 4. Skomentuj różnicę precyzji (precision) i czułości (recall) między modelami bez i z dostrajaniem hiperparametrów. Czy jest to pożądane zjawisko w tym przypadku?
# 5. Wartość ROC przypisz do zmiennej `auroc`.
#
# **Uwaga:**
# - koniecznie ustaw `verbose=-1` przy tworzeniu `LGBMClassifier`, żeby uniknąć kolosalnej ilości logów, która potrafi też wyłączyć Jupytera
# - pamiętaj o ustawieniu `importance_type`, `random_state=0` i `n_jobs`, oraz ewentualnie `verbose` w `RandomizedSearchCV` dla śledzenia przebiegu
# - istnieje możliwość, że ustawienie `n_jobs` dla grid searcha będzie szybsze niż dla samego LightGBM; odpowiada to tuningowi wielu klasyfikatorów równolegle, przy wolniejszym treningu pojedynczych klasyfikatorów
# - nie ustawiaj wszędzie `n_jobs=-1`, bo wtedy stworzysz więcej procesów niż rdzeni i spowodujesz thread contention

# %% editable=true id="hYiwjN-zLrbJ" pycharm={"is_executing": true, "name": "#%%\n"} tags=["ex"]
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

lgbm_base = LGBMClassifier(
    random_state=0,
    n_jobs=4,
    importance_type='gain',
    verbose=-1
)

param_grid = {
    "n_estimators": [100, 250, 500],
    "learning_rate": [0.05, 0.1, 0.2],
    "num_leaves": [31, 48, 64],
    "colsample_bytree": [0.8, 0.9, 1.0],
    "subsample": [0.8, 0.9, 1.0],
}

random_search = RandomizedSearchCV(
    estimator=lgbm_base,
    param_distributions=param_grid,
    n_iter=30,
    scoring='roc_auc',
    cv=5,
    random_state=0,
    n_jobs=-1,
    verbose=1
)

random_search.fit(X_train_smote, y_train_smote)

print("Najlepsze parametry LightGBM:")
print(random_search.best_params_)

lgbm_base.fit(X_train_smote, y_train_smote)
y_pred_base = lgbm_base.predict(X_test)
y_proba_base = lgbm_base.predict_proba(X_test)[:, 1]
auroc_base = roc_auc_score(y_test, y_proba_base)

lgbm_best = random_search.best_estimator_
y_pred_best = lgbm_best.predict(X_test)
y_proba_best = lgbm_best.predict_proba(X_test)[:, 1]
auroc = roc_auc_score(y_test, y_proba_best)

# fajny raport z chatGPT
print("\n=== Raport LightGBM (bez tuningu) ===")
print(classification_report(y_test, y_pred_base))

print("\n=== Raport LightGBM (po tuningu) ===")
print(classification_report(y_test, y_pred_best))

print(f"\nAUROC (bazowy): {auroc_base:.3f}")
print(f"AUROC (po tuningu): {auroc:.3f}")



# %% editable=true id="89t2JdPBLrbJ" tags=["ex"]
assert 0.9 <= auroc <= 0.99

print("Solution is correct!")

# %% [markdown] editable=true id="rV2EXZgxLrbJ" pycharm={"name": "#%% md\n"} tags=["ex"]
# ##### Komentarz
#
# Dostrojony model stał się znacznie bardziej konserwatywny. Precyzja dla klasy mniejszościowej (1) wzrosła drastycznie (z 60% na 80%), ale stało się to kosztem spadku czułości (z 60% na 54%). Oznacza to, że model po tuningu rzadziej "bije na alarm", ale gdy już to robi, ma znacznie większą rację.
#
# Mimo że AUROC minimalnie wzrósł (co oznacza, że model ogólnie lepiej sortuje przypadki), przy domyślnym progu odcięcia (0.5) model "boi się" strzelać w klasę 1. Aby odzyskać wyższą czułość, należałoby w tym przypadku ręcznie obniżyć próg decyzyjny (np. klasyfikować jako 1 wszystko powyżej 0.3, a nie 0.5).

# %% [markdown] editable=true id="d_8H-YnULrbJ" pycharm={"name": "#%% md\n"}
# **Boosting - podsumowanie**
#
# 1. Model oparty o uczenie zespołowe.
# 2. Kolejne modele są dodawane sekwencyjnie i uczą się na błędach poprzedników.
# 3. Nauka typowo jest oparta o minimalizację funkcji kosztu (błędu), z użyciem spadku wzdłuż gradientu.
# 4. Wiodący model klasyfikacji dla danych tabelarycznych, z 2 głównymi implementacjami: XGBoost i LightGBM.
# 5. Liczne hiperparametry, wymagające odpowiednich metod dostrajania.

# %% [markdown] editable=true id="kigwIyRdLrbJ" pycharm={"name": "#%% md\n"}
# ## Wyjaśnialna AI

# %% [markdown] editable=true id="xfXUureCLrbK" pycharm={"name": "#%% md\n"}
# W ostatnich latach zaczęto zwracać coraz większą uwagę na wpływ sztucznej inteligencji na społeczeństwo, a na niektórych czołowych konferencjach ML nawet obowiązkowa jest sekcja "Social impact" w artykułach naukowych. Typowo im lepszy model, tym bardziej złożony, a najpopularniejsze modele boostingu są z natury skomplikowane. Kiedy mają podejmować krytyczne decyzje, to musimy wiedzieć, czemu predykcja jest taka, a nie inna. Jest to poddziedzina uczenia maszynowego - **wyjaśnialna AI (explainable AI, XAI)**.
#
# Taka informacja jest cenna, bo dzięki temu lepiej wiemy, co robi model. Jest to ważne z kilku powodów:
# 1. Wymogi prawne - wdrażanie algorytmów w ekonomii, prawie etc. ma coraz częściej konkretne wymagania prawne co do wyjaśnialności predykcji.
# 2. Dodatkowa wiedza dla użytkowników - często dodatkowe obserwacje co do próbek są ciekawe same w sobie i dają wiedzę użytkownikowi (często posiadającemu specjalistyczną wiedzę z dziedziny), czasem nawet bardziej niż sam model predykcyjny.
# 3. Analiza modelu - dodatkowa wiedza o wewnętrznym działaniu algorytmu pozwala go lepiej zrozumieć i ulepszyć wyniki, np. przez lepszy preprocessing danych.
#
# W szczególności można ją podzielić na **globalną** oraz **lokalną interpretowalność (global / local interpretability)**. Ta pierwsza próbuje wyjaśnić, czemu ogólnie model działa tak, jak działa. Analizuje strukturę modelu oraz trendy w jego predykcjach, aby podsumować w prostszy sposób jego tok myślenia. Interpretowalność lokalna z kolei dotyczy predykcji dla konkretnych próbek - czemu dla danego przykładu model podejmuje dla niego taką, a nie inną decyzję o klasyfikacji.
#
# W szczególności podstawowym sposobem interpretowalności jest **ważność cech (feature importance)**. Wyznacza ona, jak ważne są poszczególne cechy:
# - w wariancie globalnym, jak mocno model opiera się na poszczególnych cechach,
# - w wariancie lokalnym, jak mocno konkretne wartości cech wpłynęły na predykcję, i w jaki sposób.
#
# Teraz będzie nas interesować globalna ważność cech. Dla modeli drzewiastych definiuje się ją bardzo prosto. Każdy podział w drzewie decyzyjnym wykorzystuje jakąś cechę i redukuje z pomocą podziału funkcję kosztu (np. entropię) o określoną ilość. Dla drzewa decyzyjnego ważność to sumaryczna redukcja entropii, jaką udało się uzyskać za pomocą danej cechy. Dla lasów losowych i boostingu sumujemy te wartości dla wszystkich drzew. Alternatywnie można też użyć liczby splitów, w jakiej została użyta dana cecha, ale jest to mniej standardowe.
#
# Warto zauważyć, że taka ważność cech jest **względna**:
# - nie mówimy, jak bardzo ogólnie ważna jest jakaś cecha, tylko jak bardzo przydatna była dla naszego modelu w celu jego wytrenowania,
# - ważność cech można tylko porównywać ze sobą, np. jedna jest 2 razy ważniejsza od drugiej; nie ma ogólnych progów ważności.
#
# Ze względu na powyższe, ważności cech normalizuje się często do zakresu [0, 1] dla łatwiejszego porównywania.

# %% [markdown] editable=true id="n3rtNBc_LrbK" tags=["ex"]
# ### Zadanie 9 (0.5 punktu)

# %% [markdown] editable=true id="t8AEpaVZLrbK" tags=["ex"]
# 1. Wybierz 5 najważniejszych cech dla drzewa decyzyjnego. Przedstaw wyniki na poziomym wykresie słupkowym. Użyj czytelnych nazw cech ze zmiennej `feature_names`.
# 2. Powtórz powyższe dla lasu losowego, oraz dla boostingu (tutaj znormalizuj wyniki - patrz uwaga niżej). Wybierz te hiperparametry, które dały wcześniej najlepsze wyniki.
# 3. Skomentuj, czy wybrane cechy twoim zdaniem mają sens jako najważniejsze cechy.
#
# **Uwaga:** Scikit-learn normalizuje ważności do zakresu [0, 1], natomiast LightGBM nie. Musisz to znormalizować samodzielnie, dzieląc przez sumę.

# %% editable=true id="OIHv2JiHLrbK" tags=["ex"]
import matplotlib.pyplot as plt
import numpy as np

importances_tree = tree_smote.feature_importances_
indices_tree = np.argsort(importances_tree)[::-1][:5]

plt.figure(figsize=(8, 4))
plt.barh(range(5), importances_tree[indices_tree][::-1], color='skyblue')
plt.yticks(range(5), [feature_names[i] for i in indices_tree[::-1]])
plt.xlabel('Ważność cechy')
plt.title('Top 5 cech — Drzewo decyzyjne')
plt.tight_layout()
plt.show()

importances_rf = rf_smote.feature_importances_
indices_rf = np.argsort(importances_rf)[::-1][:5]

plt.figure(figsize=(8, 4))
plt.barh(range(5), importances_rf[indices_rf][::-1], color='lightgreen')
plt.yticks(range(5), [feature_names[i] for i in indices_rf[::-1]])
plt.xlabel('Ważność cechy')
plt.title('Top 5 cech — Las losowy')
plt.tight_layout()
plt.show()

importances_lgbm = lgbm_best.feature_importances_.astype(float)
importances_lgbm /= importances_lgbm.sum()
indices_lgbm = np.argsort(importances_lgbm)[::-1][:5]

plt.figure(figsize=(8, 4))
plt.barh(range(5), importances_lgbm[indices_lgbm][::-1], color='orange')
plt.yticks(range(5), [feature_names[i] for i in indices_lgbm[::-1]])
plt.xlabel('Znormalizowana ważność cechy')
plt.title('Top 5 cech — LightGBM')
plt.tight_layout()
plt.show()



# %% [markdown] editable=true id="7_za7MGdLrbL" tags=["ex"]
# ##### Komentarz
#
# Wybrane cechy mają głęboki sens ekonomiczny i idealnie pasują do problemu przewidywania bankructwa. Modele niezależnie od siebie wskazały na kluczowe filary kondycji firmy:
#
#  - Obsługa długu (np. profit... / financial expenses) – czy firmę stać na spłatę odsetek? To często być albo nie być dla przedsiębiorstwa.
#
#  - Stabilność (retained earnings / total assets) – klasyczny wskaźnik (część modelu Altmana) pokazujący, czy firma ma "poduszkę finansową" z lat ubiegłych.
#
#  - Trend sprzedaży (sales n / sales n-1) – spadek przychodów to często pierwszy sygnał ostrzegawczy przed utratą płynności.
#
# Fakt, że różne algorytmy (drzewo, las, boosting) wytypowały bardzo podobny zestaw logicznych wskaźników, potwierdza wiarygodność naszych modeli – nauczyły się zwracać uwagę na istotne cechy przedsiębiorstwa, a nie na przypadkowy szum.

# %% [markdown] id="068HWOQMLrbL"
# ### Dla zainteresowanych
#
# Najpopularniejszym podejściem do interpretowalności lokalnych jest **SHAP (SHapley Additive exPlanations)**, metoda oparta o kooperatywną teorię gier. Traktuje się cechy modelu jak zbiór graczy, podzielonych na dwie drużyny (koalicje): jedna chce zaklasyfikować próbkę jako negatywną, a druga jako pozytywną. O ostatecznej decyzji decyduje model, który wykorzystuje te wartości cech. Powstaje pytanie - w jakim stopniu wartości cech przyczyniły się do wyniku swojej drużyny? Można to obliczyć jako wartości Shapleya (Shapley values), które dla modeli ML oblicza algorytm SHAP. Ma on bardzo znaczące, udowodnione matematycznie zalety, a dodatkowo posiada wyjątkowo efektywną implementację dla modeli drzewiastych oraz dobre wizualizacje.
#
# Bardzo intuicyjnie, na prostym przykładzie, SHAPa wyjaśnia [pierwsza część tego artykułu](https://iancovert.com/blog/understanding-shap-sage/). Dobrze i dość szczegółówo SHAPa wyjaśnia jego autor [w tym filmie](https://www.youtube.com/watch?v=-taOhqkiuIo).

# %% [markdown] id="w9NGkT2DLrbL" pycharm={"name": "#%% md\n"}
# **Wyjaśnialna AI - podsumowanie**
#
# 1. Problem zrozumienia, jak wnioskuje model i czemu podejmuje określone decyzje.
# 2. Ważne zarówno z perspektywy data badaczy danych, jak i użytkowników systemu.
# 3. Można wyjaśniać model lokalnie (konkretne predykcje) lub globalnie (wpływ poszczególnych cech).

# %% [markdown] editable=true id="BLA917TfLrbL" tags=["ex"]
# ## Zadanie dodatkowe (3 punkty)

# %% [markdown] editable=true id="EWqObRklLrbL" pycharm={"name": "#%% md\n"} tags=["ex"]
# Dokonaj selekcji cech, usuwając 20% najsłabszych cech. Może się tu przydać klasa `SelectPercentile`. Czy Random Forest i LightGBM (bez dostrajania hiperparametrów, dla uproszczenia) wytrenowane bez najsłabszych cech dają lepszy wynik (AUROC lub innej metryki)?
#
# Wykorzystaj po 1 algorytmie z 3 grup algorytmów selekcji cech:
# 1. Filter methods - mierzymy ważność każdej cechy niezależnie, za pomocą pewnej miary (typowo ze statystyki lub teorii informacji), a potem odrzucamy (filtrujemy) te o najniższej ważności. Są to np. `chi2` i `mutual_info_classif` z pakietu `sklearn.feature_selection`.
# 2. Embedded methods - klasyfikator sam zwraca ważność cech, jest jego wbudowaną cechą (stąd nazwa). Jest to w szczególności właściwość wszystkich zespołowych klasyfikatorów drzewiastych. Mają po wytrenowaniu atrybut `feature_importances_`.
# 2. Wrapper methods - algorytmy wykorzystujące w środku używany model (stąd nazwa), mierzące ważność cech za pomocą ich wpływu na jakość klasyfikatora. Jest to np. recursive feature elimination (klasa `RFE`). W tym algorytmie trenujemy klasyfikator na wszystkich cechach, wyrzucamy najsłabszą, trenujemy znowu i tak dalej.
#
# Typowo metody filter są najszybsze, ale dają najsłabszy wynik, natomiast metody wrapper są najwolniejsze i dają najlepszy wynik. Metody embedded są gdzieś pośrodku.
#
# Dla zainteresowanych, inne znane i bardzo dobre algorytmy:
# - Relief (filter method) oraz warianty, szczególnie ReliefF, SURF i MultiSURF (biblioteka `ReBATE`): [Wikipedia](https://en.wikipedia.org/wiki/Relief_(feature_selection)), [artykuł "Benchmarking Relief-Based Feature Selection Methods"](https://www.researchgate.net/publication/321307194_Benchmarking_Relief-Based_Feature_Selection_Methods)
# - Boruta (wrapper method), stworzony na Uniwersytecie Warszawskim, łączący Random Forest oraz testy statystyczne (biblioteka `boruta_py`): [link 1](https://towardsdatascience.com/boruta-explained-the-way-i-wish-someone-explained-it-to-me-4489d70e154a), [link 2](https://danielhomola.com/feature%20selection/phd/borutapy-an-all-relevant-feature-selection-method/)

# %% editable=true id="DsS9ndAGLrbL" pycharm={"name": "#%%\n"} tags=["ex"]
from sklearn.feature_selection import SelectPercentile, mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
import numpy as np

feature_cols = X_train.columns
if not isinstance(X_train_smote, pd.DataFrame):
    X_train_smote = pd.DataFrame(X_train_smote, columns=feature_cols)
if not isinstance(X_test, pd.DataFrame):
    X_test = pd.DataFrame(X_test, columns=feature_cols)

n_features = X_train_smote.shape[1]
n_keep = int(n_features * 0.8)

def evaluate_classifiers(X_tr, X_te, label):
    rf_clf = RandomForestClassifier(
        n_estimators=500,
        criterion="entropy",
        random_state=0,
        n_jobs=-1,
    )
    rf_clf.fit(X_tr, y_train_smote)
    rf_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_te)[:, 1])

    lgbm_clf = LGBMClassifier(random_state=0, n_jobs=-1, verbose=-1)
    lgbm_clf.fit(X_tr, y_train_smote)
    lgbm_auc = roc_auc_score(y_test, lgbm_clf.predict_proba(X_te)[:, 1])

    print(f"\n{label}")
    print(f"  Random Forest AUROC: {rf_auc:.3f}")
    print(f"  LightGBM AUROC:      {lgbm_auc:.3f}")
    return rf_auc, lgbm_auc

baseline_rf, baseline_lgbm = evaluate_classifiers(
    X_train_smote, X_test, "Wszystkie cechy (bazowo)"
)

# 1. Filter method - mutual information
mi_scores = mutual_info_classif(X_train_smote, y_train_smote, random_state=0)
mi_threshold = np.sort(mi_scores)[n_features - n_keep]
mi_mask = mi_scores >= mi_threshold
X_train_mi = X_train_smote.loc[:, mi_mask]
X_test_mi = X_test.loc[:, mi_mask]
filter_rf, filter_lgbm = evaluate_classifiers(
    X_train_mi, X_test_mi, f"Filter (mutual_info) - {X_train_mi.shape[1]} cech"
)

# 2. Embedded method - Random Forest importances
rf_embedded = RandomForestClassifier(
    n_estimators=500, criterion="entropy", random_state=0, n_jobs=-1
)
rf_embedded.fit(X_train_smote, y_train_smote)
importances = rf_embedded.feature_importances_
imp_threshold = np.sort(importances)[n_features - n_keep]
emb_mask = importances >= imp_threshold
X_train_emb = X_train_smote.loc[:, emb_mask]
X_test_emb = X_test.loc[:, emb_mask]
embedded_rf, embedded_lgbm = evaluate_classifiers(
    X_train_emb, X_test_emb, f"Embedded (RF importances) - {X_train_emb.shape[1]} cech"
)

# 3. Wrapper method - RFE with Random Forest
rfe_estimator = RandomForestClassifier(
    n_estimators=100, criterion="entropy", random_state=0, n_jobs=-1
)
rfe = RFE(rfe_estimator, n_features_to_select=n_keep, step=5)
rfe.fit(X_train_smote, y_train_smote)
X_train_rfe = rfe.transform(X_train_smote)
X_test_rfe = rfe.transform(X_test)
wrapper_rf, wrapper_lgbm = evaluate_classifiers(
    X_train_rfe, X_test_rfe, f"Wrapper (RFE) - {n_keep} cech"
)

print("\n=== Podsumowanie ===")
print(f"Baseline RF:  {baseline_rf:.3f}  ->  Filter: {filter_rf:.3f}, Embedded: {embedded_rf:.3f}, Wrapper: {wrapper_rf:.3f}")
print(f"Baseline LGBM: {baseline_lgbm:.3f}  ->  Filter: {filter_lgbm:.3f}, Embedded: {embedded_lgbm:.3f}, Wrapper: {wrapper_lgbm:.3f}")
print(
    "\nSelekcja cech nie zawsze poprawia wynik AUROC - czasem usunięcie słabszych cech "
    "pomaga Random Forest (mniej szumu), ale dla LightGBM efekt bywa neutralny lub lekko negatywny, "
    "bo boosting sam radzi sobie z mniej istotnymi atrybutami."
)

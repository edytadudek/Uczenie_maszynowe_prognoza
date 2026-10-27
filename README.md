# Prognozowanie popytu na produkty przy użyciu algorytmu XGBoost
Celem projektu jest prognozowanie popytu na produkty w hurtowni mięsa z wykorzystaniem modeli XGBoost i Random Forest na danych historycznych (2021–2025).

## Stack Technologiczny
* **Język:** Python (Pandas, NumPy, Scikit-learn)
* **Modele ML:** XGBoost Regressor, Random Forest Regressor
* **Wizualizacja & BI:** Power BI, Matplotlib, Seaborn

## Struktura repozytorium
```
meat_sales/
├── dane/
│   ├── surowe/          # Sprzedaz.csv, ProduktyKategorie.csv (niepublikowane)
│   ├── przetworzone/    # Dane_sklepy.csv, Dane_restauracje.csv (niepublikowane, generowane)
│   └── kalendarz/
│       └── Kalendarz.csv          # wygenerowany kalendarz świąt i cech czasowych
├── notebooki/
│   ├── 01_analiza_eksploracyjna.ipynb      # EDA surowych danych
│   ├── 02_kalendarz.ipynb                  # święta, ferie, majówka, cechy cykliczne
│   ├── 03_przygotowanie_sklepy.ipynb       # szereg czasowy Data × Klient × Podkategoria (sklepy)
│   ├── 04_przygotowanie_restauracje.ipynb  # j.w. dla HoReCa + cechy okazji
│   ├── 05_model_sklepy_rf.ipynb
│   ├── 06_model_sklepy_xgb.ipynb           # + porównanie RF vs XGB (sklepy)
│   ├── 07_model_restauracje_rf.ipynb
│   └── 08_model_restauracje_xgb.ipynb      # + porównanie RF vs XGB (restauracje)
├── wyniki/
│   ├── prognozy/        # CSV z prognozami na 2025 (niepublikowane, generowane)
│   └── wykresy/         # wykresy wyników
├── requirements.txt
└── README.md
```

## Dane
Surowe dane transakcyjne pochodzą z wewnętrznego systemu hurtowni i **nie są publikowane** w repozytorium (poufność danych firmowych oraz rozmiar plików > 50 MB). Aby uruchomić notebooki, umieść w folderze `dane/surowe/` pliki:
* `Sprzedaz.csv` – transakcje: `Klient;data_sprzedaży;ID_artykułu;Cena_zakupu;Cena_sprzedaży;Ilość;Wartość_netto;Stawka_VAT`
* `ProduktyKategorie.csv` – słownik produktów: `ID_artykułu;Nazwa;grupa_towarów;producent;rodzaj_towarów;pierwsze_nazwa;rodzaj_towaru`

Separator `;`, kodowanie UTF-8, przecinek jako separator dziesiętny.

## Uruchomienie
```bash
pip install -r requirements.txt
jupyter notebook
```
Notebooki z folderu `notebooki/` uruchamiaj w kolejności numerów 01 → 08 (notebooki XGB porównują wyniki z prognozami RF, więc RF musi być policzony wcześniej). Ścieżki są względne względem folderu `notebooki/` (`../dane/...`); Jupyter domyślnie uruchamia notebook w jego własnym folderze. Jest to tylko wersja do podglądu ze względu na fakt, że nie udostępniam orginalnych danych.

---------------------------------------------------------------------------------------------------------------
## 1. Opis i przygotowanie danych (Data Engineering)
Projekt opiera się na rzeczywistym, 5-letnim zbiorze danych (2021-2025) pochodzącym z wewnętrznego systemu hurtowni mięs i wędlin. Dane obejmują sprzedaż B2B dla dwóch głównych segmentów: sklepów detalicznych oraz klientów z sektora HoReCa (restauracje).

### Charakterystyka oryginalnego zbioru danych
* **Rozmiar:** Surowy zbiór (**196 045 transakcji**) przekształcono w ciągłe szeregi czasowe gotowe dla modeli ML (**483 890 rekordów dla sklepów** i **465 630 dla restauracji**).
* **Typ danych:** Transakcyjne (jeden wiersz = jedna transakcja sprzedażowa konkretnego dnia dla danego klienta).
* **Atrybuty:** m.in. data sprzedaży, ID artykułu, rodzaj klienta (S - sklep, R - restauracja), ilość (kg), wartość netto, cena zakupu i sprzedaży.

### Feature Engineering
Dane charakteryzowały zawiłością oraz elementami do skorygowania.
* **Data Cleaning:** Usunięto transakcje ujemne oraz mikro-zamówienia (< 0.5 kg). Skorygowano błędy wprowadzania danych (Przykładowo: outliery cenowe typu 954 zł/kg zastąpiono średnią rynkową). Skrajne anomalie wolumenowe (np. jednorazowe zamówienie 1.3 tony, przy czym średnia zamówień klienta wynosła 14 kg) zastąpiono średnią ilością sprzedaży w danym okresie klienta. Dane zawierały pozycję z ilością i wartością netto równą zero w praktyce rekordy występowały w bazie sprzedaży. W bazie również można było zauważyć cenę_zakupu na zero, która przypuszczalnie mogła oznaczać promocję od producenta.
* **Rekategoryzacja asortymentu:** Przeprowadzono klasyfikację 3029 ogólnych produktów operacyjnych do **11 kategorii głównych** oraz **53 podkategorii**.
* **Ciągłość Szeregu Czasowego:** Zbudowano pełną siatkę kombinacji `Data × Klient × Podkategoria` (1826 dni), a brakujące dni nieoperacyjne (np. święta) uzupełniono wartością 0.
* **Feature Engineering (Inżynieria Cech):**
  * *Cykliczność:* Transformacja trygonometryczna (sinus/cosinus) dla miesięcy i dni tygodnia.
  * *Kalendarz:* Flagi dni wolnych, świąt ruchomych, majówki, wakacji oraz ferii zimowych.
  * *Specyfika HoReCa:* Flagi okazji generujących ruch w gastronomii (m.in. Walentynki, Dzień Kobiet, Dzień Matki, Andrzejki) wraz z 3-dniowym oknem wyprzedzającym, flaga czwartku oraz sezonu restauracyjnego (wykorzystane w modelu XGBoost).
  * *Autokorelacja:* Generowanie zmiennych opóźnionych czasowo (**`lag_364`**), mapujących popyt w ten sam dzień tygodnia rok wcześniej.

---------------------------------------------------------------------------------------------------------------
## 2. Modelowanie i Predykcja (Machine Learning)

W celu prognozowania ilości zapotrzebowania (w kilogramach) zaimplementowano i porównano dwa podejścia zespołowe oparte na drzewach decyzyjnych: **XGBoost Regressor** oraz **Random Forest Regressor**. Algorytmy te radzą sobie z nieliniowością danych tabelarycznych oraz szumem informacyjnym.
###  Strategia walidacji szeregów czasowych (Time-Based Splitting)
Aby uniknąć wycieku danych z przyszłości (data leakage) zastosowano podział chronologiczny:
* **Zbiór treningowy:** 01.01.2021 – 31.12.2023 (Trening modeli bazowych)
* **Zbiór walidacyjny:** 01.01.2024 – 31.12.2024 (Strojenie hiperparametrów i Early Stopping)
* **Zbiór testowy:** 01.01.2025 – 31.12.2025 (Ocena końcowa na danych testowych)

### Architektura Modelu
#### XGBoost (Gradient Boosting)
* **XGBoost:** Budowana sekwencyjna z optymalizacją błędów poprzednich drzew. Wykorzystano funkcję celu **Tweedie** (`reg:tweedie`), przystosowaną do rozkładów z dużą gęstością zer i wartościami dodatnimi. Wydajność zoptymalizowano metodą histogramową (`tree_method='hist'`) oraz regularyzacją L1/L2.

#### Random Forest (Bagging)
* **Random Forest:** Zastosowano podejście Baggingu na 500 drzewach z narzuconą restrykcją gęstości liści (`min_samples_leaf`). Eksperymentalnie wykazano wyższą stabilność modelu przy zastosowaniu **transformacji logarytmicznej** zmiennej docelowej.

--------------------------------------------------------------------------------------------------------------
##  3. Wyniki i Porównanie Skuteczności (Model Evaluation)
Jakość prognoz została zweryfikowana na zbiorze testowym (rok 2025) za pomocą metryk: **MAE**, **RMSE**, **R²**, **Bias** oraz **WMAPE**.
####  Segment A: Sklepy Detaliczne (Porównanie na zbiorze testowym 2025)
| Metryka jakości | XGBoost Regressor | Random Forest Regressor | Średnia wartość rzeczywista |
| :--- | :---: | :---: | :---: |
| **MAE** [kg] | 1.6500 | **1.6098** | 1.8155 kg |
| **RMSE** [kg] | **7.6519** | 9.2796 | — |
| **Współczynnik R²** | **0.4928** | 0.2540 | — |
| **Bias** [kg] | **-0.3548** | -1.0220 | — |
| **WMAPE** [%] | **63.40%** | 76.50% | — |
**Wnioski (Sklepy):**
* **Zwycięzca: XGBoost.** Wyjaśnia blisko **50% zmienności popytu** ($R^2 = 0.4928$) w porównaniu do zaledwie 25% w przypadku Random Forest.
* Random Forest charakteryzuje się znacznie wyższym błędem **RMSE (9.28 kg)**, co oznacza, że generuje duże, krytyczne błędy predykcji. Posiada również silny negatywny **Bias (-1.02 kg)** – model drastycznie niedoszacowuje zamówień (w skali całego asortymentu niedoszacował popyt aż o 98 ton, podczas gdy XGBoost tylko o 39 ton).

#### Segment B: HoReCa / Restauracje (Porównanie na zbiorze testowym 2025)
| Metryka jakości | XGBoost Regressor | Random Forest Regressor | Średnia wartość rzeczywista |
| :--- | :---: | :---: | :---: |
| **MAE** [kg] | 1.5184 | **1.3298** | 1.7761 kg |
| **RMSE** [kg] | **13.1218** | 13.6942 | — |
| **Współczynnik R²** | **0.5706** | 0.5323 | — |
| **Bias** [kg] | **-0.2782** | -0.6206 | — |
| **WMAPE** [%] | **64.70%** | 68.30% | — |

 **Kluczowe wnioski (Restauracje):**
* Segment gastronomiczny okazał się trudniejszy ze względu na potężną asymetrię (ogromne, skokowe zamówienia i długie okresy braku aktywności).
* **XGBoost** wykazuje lepszą ogólną stabilność predykcji i zdolność do odzwierciedlania zmienności rynku ($R^2 = 0.57$, niższy WMAPE). Ponownie wykazuje mniejszą tendencję do zaniżania prognoz (**Bias: -0.28 kg** vs -0.62 kg w RF).
* **Random Forest** zanotował jednak niższy błąd bezwzględny MAE (1.33 kg), co oznacza, że jego prognozy w standardowe, nieświąteczne dni były bliższe rzeczywistości, mimo słabszego radzenia sobie z ekstremalnymi pikami.
---------------------------------------------------------------------------------------------------------------
###  4. Istotność Cech (Feature Importance Insight)
Analiza wag zmiennych objaśniających ujawniła różnice w zachowaniu algorytmów:
* **XGBoost** silnie sparametryzował się pod zmienne kalendarzowe i restrykcje operacyjne. Dla sklepów kluczową cechą okazała się flaga `czy_niedziela` (waga 0.38) oraz zmienna historyczna `lag_364` (waga 0.21). Model doskonale zmapował cykliczny rytm pracy hurtowni i sezonowość.
* **Random Forest** w obu segmentach uzależnił się głównie od zmiennej historycznej `lag_364` (waga do 0.42) oraz danych strukturalnych (`podkategoria`, `klient`).
* **Wniosek:** Wyższa skuteczność XGBoost wynika z faktu, że w predykcji popytu branży mięsnej uwarunkowania kalendarzowe (dni wolne, ruchome okna świąteczne) mają silniejsze znaczenie biznesowe niż prosta kontynuacja trendu z zeszłego roku.

###  5. Wnioski z realizowanego projektu 
---------------------------------------------------------------------------------------------------------------
Zastosowanie uczenia maszynowego w predykcji procesów biznesowych jest możliwy, lecz wymaga dostosowania się do danych sprzedażowych i bilższego poznania konkekstu danych.

Ograniczenia:
* Oba modele wykazują tendencję do niedoszacowania rzeczywistej sprzedaży.
* Modele mają trudności z przewidywaniem nagłych zmian i skoków popytu.
* Silny wpływ zmiennej lag_364 wskazuje na dużą zależność prognoz od wzorca sprzedaży z poprzedniego roku.
* Badanie obejmuje tylko sprzedaż kilku konsumentów, nie została dostarczona informacja jakiej natury są dane sklepy, restaurację. 
* Brak dokładnych informacji o charakterze danych, jej strukturze, zmiennych które wynikają ze specyfiki sprzedaży.  

Kierunki dalszego rozwoju projektu:
* Dodanie do zbioru danych dodatkowych zmiennych (np. promocje)
* Rozszerzenie Feature Engineering o dodatkowe opóźnienia, np. lag_7, lag_30, lag_90.
* Opracowanie prognozy tygodniowej, która może być bardziej stabilna i lepiej odpowiadać potrzebom planowania zakupów oraz dostaw.
* Dalsza optymalizacja modelu XGBoost pod kątem ograniczenia systematycznego niedoszacowania.
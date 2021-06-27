## header H2 Projekt SI papier, kamień, nożyce

W tym projekcie użyłem do klasyfikacji obrazu pochodzącego z kamerki (przy użyciu opencv) biblioteki keras. Projekt składa się z pliku train.py, gdzie jest uczony model, a w play.py znajduje sie logika gry.


## Technologia
Do projektu użyłem:
* keras
* opencv
* numpy
* oraz innych bibliotek, które znajdują się w pliku requirements.txt


W projekcie użyłem następujących materiałów:
https://keras.io/examples/vision/image_classification_from_scratch/
https://www.kaggle.com/drgfreeman/rockpaperscissors

## instalacja
Najprościej jest zainstalować przy użyciu anaconda
```
$ conda create --name <env> --file requirements.txt
```
Następnie wytrenować model
```
$ python train.py
```
A później uruchomić grę
```
$ python play.py
```

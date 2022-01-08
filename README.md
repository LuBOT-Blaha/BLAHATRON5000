# BLAHATRON5000

Toto je rekurentná neurálna sieť trénovaná na facebookových statusoch Ľuboša Blahu. (Všetky statusy ktoré sme použili sú v súbore model.txt) Je z veľkej časti založená na kóde od NeuralNine https://www.neuralnine.com/generating-texts-with-recurrent-neural-networks-in-python/

## Generovanie vlastných statusov

Ak si chceš vygenerovať vlastné posty, naklonuj repozitár a edituj blahatron_out.py. Tam si vieš nastaviť dĺžku vygenerovaných postov aj teplotu (v istom zmysle je to miera náhodnosti výberu písmen počas generovania --- môže viesť ku skomoleným slovám a všeliakým iným "preklepom") Keď budeš mať blahatron_out.py nastavený ako chceš, spusti ho príkazom "python blahatron_out.py". Tak, ako je teraz napísaný vytvorí súbor output.txt a vygeneruje v ňom 100 statusov. Ak vieš python, vieš si ho jednoducho upraviť aby napríklad písal do konzoly a podobne. Hlavná funkcia o ktorej potrebuješ vedieť je generate_text( , ). Tá berie dva vstupy -- prvý je počet znakov v statuse ktorý má generovať a druhý vstup je teplota.

Časť kódu ktorá generuje statusy je 
```
f = open("output.txt", "a")


for iterator in range(1, 101):
    count = int(abs(random.gauss(2721, 1506) - 300)) + 300
    temper = random.uniform(0.15, 0.35)
    f.write("----------------------")
    f.write("\n")
    f.write(generate_text(count, temper))
    f.write("\n")

f.close()
```
tento kód generuje s náhodnou teplotou z rozsahu (0.15 - 0.35) a s dĺžkou statusov nastavenou tak, aby sa zhodovala s distribúciou dĺžky statusov skutočného BLAHATRONA. Minimálne tých, ktoré sa nám podarilo získať.

Ak chceš vygenerovať status s presne určenou dĺžkou a teplotou, celý ten blok vymaž alebo zakomentuj a nahraď
```
print(generate_text(dĺžka, teplota))
```
## Requirements

Na spustenie kódu potrebuješ python, tensorfow a numpy


## Acknowledgements

Tento kód je z veľkej časti založený na kóde od NeuralNine, preto sme aj zachovali pôvodnú hlavičku s atribúciou v súboroch *.py

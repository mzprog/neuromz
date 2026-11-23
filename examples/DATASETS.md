# Example datasets: sources and licenses

| file | what | source | license |
|---|---|---|---|
| `xor.csv`, `xor_and.csv` | truth tables | written by hand | public domain (CC0) |
| `iris.csv` (+ `iris_train.csv`, `iris_test.csv`) | Fisher's Iris data, 150 flowers | the copy shipped with scikit-learn (identical to R's `iris` and Fisher 1936); train/test = every 5th row | public domain (historical data set) |
| `names_en.txt` | 5163 English first names, one per line | US Census Bureau, *Frequently Occurring Names*, 1990 census: `dist.female.first` (4275) and `dist.male.first` (1219) from www2.census.gov/topics/genealogy/1990surnames/ | public domain (work of the US federal government) |
| `names_ar.txt` | 504 common Arabic given names, one per line | compiled for this project (see below) | CC0 |
| `data/train-*`, `data/t10k-*` (`make data`, not committed) | MNIST | LeCun, Cortes, Burges; via the Google / AWS mirrors, MD5-checked | CC BY-SA 3.0 (as distributed by the authors' site) |
| `data/alice.txt` (`make data-text`, not committed) | Lewis Carroll, *Alice's Adventures in Wonderland* (1865), 151 KB, 3382 lines | Project Gutenberg eBook #11 via official mirrors, SHA-256-checked, header/footer removed | public domain in the USA |

## How `names_en.txt` was made

The two Census files list one name per line in upper case with frequency
columns. The names were taken in file order (female, then male), converted to
title case (`MARY` → `Mary`), de-duplicated (331 names occur in both lists),
and shuffled with Python's `random.Random(1990)` so that the last 10 % (the
validation split) is not just the end of the alphabet.

## How `names_ar.txt` was made

The list was written by hand for this project from general knowledge of
common given names across the Arabic-speaking world — 286 male and 227 female
names, including compound names written as one word (عبدالله, عبدالرحمن)
and the hamza forms أ إ آ ؤ ئ ء, as is usual in name lists. It is not
copied from any database. Each name was NFC-normalised and checked to contain
only Arabic letters (U+0600–U+06FF, no diacritics); duplicates (9) were
dropped and the list was shuffled with `random.Random(1447)`. It uses 36
distinct letters. It is a small, informal list (no claim of completeness or
frequency ranking) meant for demonstrating a character-level model on
right-to-left, multi-byte text.

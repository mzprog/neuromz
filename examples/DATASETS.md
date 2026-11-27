# Example datasets: sources and licenses

| file | what | source | license |
|---|---|---|---|
| `xor.csv`, `xor_and.csv` | truth tables | written by hand | public domain (CC0) |
| `iris.csv` (+ `iris_train.csv`, `iris_test.csv`) | Fisher's Iris data, 150 flowers | the copy shipped with scikit-learn (identical to R's `iris` and Fisher 1936); train/test = every 5th row | public domain (historical data set) |
| `names_en.txt` | 5163 English first names, one per line | US Census Bureau, *Frequently Occurring Names*, 1990 census: `dist.female.first` (4275) and `dist.male.first` (1219) from www2.census.gov/topics/genealogy/1990surnames/ | public domain (work of the US federal government) |
| `names_ar.txt` | 504 common Arabic given names, one per line | compiled for this project (see below) | CC0 |
| `data/train-*`, `data/t10k-*` (`cmake --build build --target data`, not committed) | MNIST | LeCun, Cortes, Burges; via the Google / AWS mirrors, MD5- and SHA-256-checked | CC BY-SA 3.0 (as distributed by the authors' site) |
| `data/alice.txt` (`cmake --build build --target data-text`, not committed) | Lewis Carroll, *Alice's Adventures in Wonderland* (1865), 151 KB, 3382 lines | Project Gutenberg eBook #11 via official mirrors, SHA-256-checked, header/footer removed | public domain in the USA |
| `data/kalila.txt` (`cmake --build build --target data-text`, not committed) | Ibn al-Muqaffaʿ, *Kalila wa-Dimna* (كليلة ودمنة, Arabic, 8th century), 431 KB, 240 376 characters, 126 paragraphs | Arabic Wikisource, the 19 chapters of [كليلة ودمنة](https://ar.wikisource.org/wiki/كليلة_ودمنة) at fixed page revisions, each SHA-256-checked, wiki markup removed (see below) | the work: public domain (author died c. 759); the Wikisource transcription: CC BY-SA 4.0 (Wikimedia terms of use) |

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

## How `data/kalila.txt` is made

The book is one of the classics of Arabic prose: Ibn al-Muqaffaʿ's 8th-
century translation and adaptation of the Indian *Panchatantra* fables. It
is far out of copyright. Wikisource's transcription is licensed CC BY-SA 4.0,
which allows downloading and processing it with attribution (this file is
the attribution: Arabic Wikisource contributors, page history linked from
each chapter). Nothing of it is committed to this repository.

`scripts/get_text.sh` (`get_text.ps1` on Windows) fetches the raw wikitext of each chapter at a fixed
revision id (`index.php?oldid=N&action=raw`; a revision never changes, so its
SHA-256 is stable even when the page is edited later), in the order of the
book's index page, and checks every checksum. `scripts/strip_wikitext.awk`
then removes the markup: the `{{ترويسة}}` header template, the one image, the
category link, link targets (`[[target|text]]` → `text`) and heading `=`
signs, and writes one paragraph per line. The text keeps its partial
vocalisation (about 2,500 tashkeel marks, 1 % of the characters: mostly
tanwin ً ٌ ٍ, some shadda ّ) and its hamza/alef spellings — what the optional
normalisation (`--strip-diacritics`, `--unify-alef`) is for. One chapter
begins with the name of the volunteer who transcribed it; it is kept, as
written on the page.

Other sources considered: Project Gutenberg has almost no Arabic books;
the Tanzil Qur'an text forbids modification (and generating imitation
scripture would be inappropriate); OpenITI's corpus is CC BY-NC-SA, not a
permissive licence.

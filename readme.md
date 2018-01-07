to be:

### Corpus fun facts

Language distribution
    
    defaultdict(<class 'int'>, {'en': 7356, 'vo': 5, 'fr': 7, 'cy': 12, 'gd': 2, 'sco': 12, 'la': 20, 'ny': 5, 'st': 1, 'rw': 6, 'un': 56, 'sm': 7, 'uz': 2, 'id': 3, 'wo': 2, 'is': 2, 'ha': 7, 'no': 9, 'so': 4, 'ga': 5, 'sa': 1, 'crs': 2, 'ia': 17, 'gv': 2, 'kha': 4, 'de': 6, 'tl': 1, 'haw': 4, 'es': 2, 'it': 1, 'lb': 1, 'da': 3245, 'co': 7, 'zzp': 4, 'fy': 1, 'war': 8, 'tt': 1, 'mg': 4, 'br': 2, 'gn': 2, 'eo': 2, 'xh': 1, 'tn': 1, 'tlh': 1, 'sv': 4, 'nl': 1, 'sk': 1, 'af': 1, 'nn': 3, 'pl': 1, 'rn': 1, 'ms': 1, 'fi': 1})

Classes distribution
    
    defaultdict(<class 'int'>, {'is': 2926, 'was': 8053, 'were': 2255, 'been': 1408, 'be': 2499, 'are': 1074, 'am': 257, 'being': 336, '----': 18})



### Installation

    brew install icu4c
    [...]
    $ ls /usr/local/Cellar/icu4c/
    58.2
    $ export ICU_VERSION=58
    $ export PYICU_INCLUDES=/usr/local/Cellar/icu4c/58.2/include
    $ export PYICU_LFLAGS=-L/usr/local/Cellar/icu4c/58.2/lib
    $ pip install pyicu
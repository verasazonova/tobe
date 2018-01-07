to be:

### Corpus fun facts

Language distribution
    
    [('English', 7356), ('Danish', 3245), ('un', 56), ('Latin', 20), ('Interlingua', 17), ('Welsh', 12), ('Scots', 12), ('Norwegian', 9), ('Waray', 8), ('French', 7), ('Samoan', 7), ('Hausa', 7), ('Corsican', 7), ('Kinyarwanda', 6), ('German', 6), ('Volapük', 5), ('Nyanja', 5), ('Irish', 5), ('Somali', 4), ('Khasi', 4), ('Hawaiian', 4), ('zzp', 4), ('Malagasy', 4), ('Swedish', 4), ('Indonesian', 3), ('Norwegian Nynorsk', 3), ('Scottish Gaelic', 2), ('Uzbek', 2), ('Wolof', 2), ('Icelandic', 2), ('Seselwa Creole French', 2), ('Manx', 2), ('Spanish', 2), ('Breton', 2), ('Guarani', 2), ('Esperanto', 2), ('Southern Sotho', 1), ('Sanskrit', 1), ('Tagalog', 1), ('Italian', 1), ('Luxembourgish', 1), ('Western Frisian', 1), ('Tatar', 1), ('Xhosa', 1), ('Tswana', 1), ('Klingon', 1), ('Dutch', 1), ('Slovak', 1), ('Afrikaans', 1), ('Polish', 1), ('Rundi', 1), ('Malay', 1), ('Finnish', 1)]

Classes distribution
    
    [('was', 8053), ('is', 2926), ('be', 2499), ('were', 2255), ('been', 1408), ('are', 1074), ('being', 336), ('am', 257), ('----', 18)]


### Installation

    brew install icu4c
    [...]
    $ ls /usr/local/Cellar/icu4c/
    58.2
    $ export ICU_VERSION=58
    $ export PYICU_INCLUDES=/usr/local/Cellar/icu4c/58.2/include
    $ export PYICU_LFLAGS=-L/usr/local/Cellar/icu4c/58.2/lib
    $ pip install pyicu
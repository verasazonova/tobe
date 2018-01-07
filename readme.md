to be:





### Installation

    brew install icu4c
    [...]
    $ ls /usr/local/Cellar/icu4c/
    58.2
    $ export ICU_VERSION=58
    $ export PYICU_INCLUDES=/usr/local/Cellar/icu4c/58.2/include
    $ export PYICU_LFLAGS=-L/usr/local/Cellar/icu4c/58.2/lib
    $ pip install pyicu
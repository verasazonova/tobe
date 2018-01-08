to be:

### Corpus fun facts

Language distribution
    
    [('English', 7356), ('Danish', 3245), ('un', 56), ('Latin', 20), ('Interlingua', 17), ('Welsh', 12), ('Scots', 12), ('Norwegian', 9), ('Waray', 8), ('French', 7), ('Samoan', 7), ('Hausa', 7), ('Corsican', 7), ('Kinyarwanda', 6), ('German', 6), ('Volapük', 5), ('Nyanja', 5), ('Irish', 5), ('Somali', 4), ('Khasi', 4), ('Hawaiian', 4), ('zzp', 4), ('Malagasy', 4), ('Swedish', 4), ('Indonesian', 3), ('Norwegian Nynorsk', 3), ('Scottish Gaelic', 2), ('Uzbek', 2), ('Wolof', 2), ('Icelandic', 2), ('Seselwa Creole French', 2), ('Manx', 2), ('Spanish', 2), ('Breton', 2), ('Guarani', 2), ('Esperanto', 2), ('Southern Sotho', 1), ('Sanskrit', 1), ('Tagalog', 1), ('Italian', 1), ('Luxembourgish', 1), ('Western Frisian', 1), ('Tatar', 1), ('Xhosa', 1), ('Tswana', 1), ('Klingon', 1), ('Dutch', 1), ('Slovak', 1), ('Afrikaans', 1), ('Polish', 1), ('Rundi', 1), ('Malay', 1), ('Finnish', 1)]

Classes distribution
    
    [('was', 8053), ('is', 2926), ('be', 2499), ('were', 2255), ('been', 1408), ('are', 1074), ('being', 336), ('am', 257), ('----', 18)]



### Training

#### Context = 5

    Evaluating on dev
                 precision    recall  f1-score   support
    
           ----       1.00      1.00      1.00         2
             am       0.71      0.83      0.77        24
            are       0.88      0.92      0.90        97
           were       0.87      0.88      0.88       205
            was       0.90      0.90      0.90       690
             is       0.83      0.85      0.84       272
           been       0.99      1.00      1.00       126
          being       1.00      0.64      0.78        33
             be       1.00      0.98      0.99       224
    
    avg / total       0.90      0.90      0.90      1673
    
    ['----', 'am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
    [[  2   0   0   0   0   0   0   0   0]
     [  0  20   0   0   3   1   0   0   0]
     [  0   0  89   6   2   0   0   0   0]
     [  0   0   8 180  14   3   0   0   0]
     [  0   8   2  18 622  38   1   0   1]
     [  0   0   1   0  40 231   0   0   0]
     [  0   0   0   0   0   0 126   0   0]
     [  0   0   1   1   9   1   0  21   0]
     [  0   0   0   1   1   3   0   0 219]]
    
    Evaluating on test
                 precision    recall  f1-score   support
    
           ----       1.00      1.00      1.00         2
             am       0.85      0.96      0.90        24
            are       0.86      0.93      0.89        98
           were       0.88      0.92      0.90       204
            was       0.92      0.93      0.93       689
             is       0.84      0.83      0.83       271
           been       1.00      0.99      1.00       127
          being       1.00      0.61      0.75        33
             be       1.00      0.96      0.98       224
    
    avg / total       0.92      0.92      0.91      1672
    
    ['----', 'am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
    [[  2   0   0   0   0   0   0   0   0]
     [  0  23   0   0   1   0   0   0   0]
     [  0   0  91   5   1   1   0   0   0]
     [  0   0   7 188   6   3   0   0   0]
     [  0   4   0  13 639  33   0   0   0]
     [  0   0   6   5  35 225   0   0   0]
     [  0   0   0   1   0   0 126   0   0]
     [  0   0   1   1   6   4   0  20   1]
     [  0   0   1   1   4   2   0   0 216]]

Context = 10

    Evaluating on dev
                 precision    recall  f1-score   support
    
           ----       1.00      1.00      1.00         2
             am       0.71      0.83      0.77        24
            are       0.88      0.92      0.90        97
           were       0.87      0.88      0.88       205
            was       0.90      0.90      0.90       690
             is       0.83      0.85      0.84       272
           been       0.99      1.00      1.00       126
          being       1.00      0.64      0.78        33
             be       1.00      0.98      0.99       224
    
    avg / total       0.90      0.90      0.90      1673
    
    ['----', 'am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
    [[  2   0   0   0   0   0   0   0   0]
     [  0  20   0   0   3   1   0   0   0]
     [  0   0  89   6   2   0   0   0   0]
     [  0   0   8 180  14   3   0   0   0]
     [  0   8   2  18 622  38   1   0   1]
     [  0   0   1   0  40 231   0   0   0]
     [  0   0   0   0   0   0 126   0   0]
     [  0   0   1   1   9   1   0  21   0]
     [  0   0   0   1   1   3   0   0 219]]
    
    Evaluating on test
                 precision    recall  f1-score   support
    
           ----       1.00      1.00      1.00         2
             am       0.85      0.96      0.90        24
            are       0.86      0.93      0.89        98
           were       0.88      0.92      0.90       204
            was       0.92      0.93      0.93       689
             is       0.84      0.83      0.83       271
           been       1.00      0.99      1.00       127
          being       1.00      0.61      0.75        33
             be       1.00      0.96      0.98       224
    
    avg / total       0.92      0.92      0.91      1672
    
    ['----', 'am', 'are', 'were', 'was', 'is', 'been', 'being', 'be']
    [[  2   0   0   0   0   0   0   0   0]
     [  0  23   0   0   1   0   0   0   0]
     [  0   0  91   5   1   1   0   0   0]
     [  0   0   7 188   6   3   0   0   0]
     [  0   4   0  13 639  33   0   0   0]
     [  0   0   6   5  35 225   0   0   0]
     [  0   0   0   1   0   0 126   0   0]
     [  0   0   1   1   6   4   0  20   1]
     [  0   0   1   1   4   2   0   0 216]]

### Installation

    brew install icu4c
    [...]
    $ ls /usr/local/Cellar/icu4c/
    58.2
    $ export ICU_VERSION=58
    $ export PYICU_INCLUDES=/usr/local/Cellar/icu4c/58.2/include
    $ export PYICU_LFLAGS=-L/usr/local/Cellar/icu4c/58.2/lib
    $ pip install pyicu
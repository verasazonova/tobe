#!/usr/bin/env bash

echo "This projects has been written for python > 3.0"
echo "It is highly recommended to start a clean virtual environment"
echo ""
echo "If yo have pyenv-virtualenv install you can do:"
echo "    pyenv virtualenv 3.6.2 tobe36"
echo "    pyenv activate tobe36"

echo

read -p "Do you want to install? y/N" response

case "$response" in
        [yY][eE][sS]|[yY])

            echo "Installing requirements"
            pip install -r requirements.txt

            echo "Installing spacy models"
            python -m spacy download en
            python -m spacy download en_vectors_web_lg

            if [ ! -d models ]; then
                mkdir models
            fi

            if [ ! -e models/weights.hdf5 ]; then
                echo "Please download the model (weights.hdf5) and place it in the models directory"
            fi

            ;;
        *)
            echo "Installation aborted.  Printing usage"
            ;;
esac

echo "To execute the program run:"
echo "    python tobe/main.py --filenames masked_text_1.txt masked_text_2.txt"
echo ""
echo "The program will write outputs in files result_masked_text_1.txt and result_masked_text_2.txt respectively"
echo ""
echo "Running ./compare_models.sh will recreate the experiments described in the report. "
echo "Beware it can take up to a few hours"


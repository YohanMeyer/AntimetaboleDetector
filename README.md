# Context

This repository contains Yohan Meyer's work concerning antimetabole detection for his master's thesis.
This work completes the chiasmi extraction pipeline developed together with Guillaume Berthomet that you can find here: <https://github.com/YohanMeyer/ChiasmusData>

You can also find Guillaume's own antimetabole detector here: <https://github.com/Dironiil/AntimetaboleDetectionModels>

# How-to

The main detection script is `antimetabole-rating.py`, which should be run in the `rating-src` environment. Some examples about how to use the `AntimetaboleRatingEngine` class are given in the same file.
All requirements for this project can be found in the `requirements.txt` file and can be installed with `pip install -r requirements.txt` or `conda install --file requirements.txt`.

Some features require the English fastText models. You can download it at the following link: <https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.de.zip>
Unzip it and put the `fasttext_models/wiki.en.bin` file in the `fasttext_models` folder.

For more information about the data, features and models, please refer to the thesis.

# Credits

This work was directly inspired by the work of Dubremetz and Nivre (2017) and Schneider et al. (2021).
You can find their projects and references at the following links:

Dubremetz and Nivre (2017) : <https://github.com/mardub1635/chiasmusDetector>

Schneider et al. (2021) : <https://github.com/cvjena/chiasmus-detector>

# Citation

This original work can be cited as follow:

```@mastersthesis{meyer2023_chiasmi,
  author={Yohan Meyer},
  title={Application of comprehensive and responsible Machine Learning methods to the detection of chiasmi's rhetorical salience},
  school={University of Passau \& INSA Lyon},
  year={2023}
}```

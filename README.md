# Automatic Speech Recognition (ASR) with PyTorch

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Этот репозиторий содержит реализацию модель Conformer-CTC для задачи ASR

Логи обучения: https://www.comet.com/progphys/asr-hw/view/new/panels

Демо: https://colab.research.google.com/drive/1-gxxBJGX5qQa9hHNlaCQCuPMuOF7XuFx?usp=sharing 

## Installation



0. Создайте окружение для работы с проектом

   b. `venv` (`+pyenv`) version:

   ```bash
   # create env
   ~/.pyenv/versions/PYTHON_VERSION/bin/python3 -m venv project_env

   # alternatively, using default python version
   python3 -m venv project_env

   # activate env
   source project_env/bin/activate
   ```

1. Установить все необходимые зависимости

   ```bash
   pip install -r requirements.txt
   ```


## How To Use

Для обучения модели сделать следующее

```bash
python3 train.py -cn=CONFIG_NAME HYDRA_CONFIG_ARGUMENTS
```

Где `CONFIG_NAME` — это конфигурация из `src/configs`, а `HYDRA_CONFIG_ARGUMENTS` — необязательные аргументы.

Чтобы запустить инференс (оценить модель или сохранить предсказания):
```bash
python3 inference.py HYDRA_CONFIG_ARGUMENTS
```
## Credits
This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).
## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

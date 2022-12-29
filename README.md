# pytlsd
Python transparent bindings for LSD (Line Segment Detector)

Bindings over the original C implementation of LSD, that allows to change the different thresholds involved and to provide custom image gradientes in order to compute the method with stronger features.

![](resources/example.jpg)

## Install

```
git clone --recursive https://github.com/ibaiGorordo/pytlsd.git
cd pytlsd
pip3 install -r requirements.txt
pip3 install .
```

## Execution

```
python3 tests/test.py
```

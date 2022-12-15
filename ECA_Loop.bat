@echo off

rem =前后不能有空格

set snr_array=20 18 16 14 12 10 8 6 4 2 0 -2 -4 -6 -8 -10
set "a=./dataset/rayleigh/dataset-rayleigh-ldpc-"
set "b=db.csv"

for %%c in (%snr_array%) do (
    python main.py -a eca_resnet18 --ksize 3557 --epochs 100 %a%%%c%b%
)

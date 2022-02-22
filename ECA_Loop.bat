@echo off

rem =前后不能有空格

set snr_array=20 18 16 14 12 10 8 6 4 2 0 -2 -4 -6 -8 -10
set epoch_array=100 200 300 400 500 600 700 800 900 1000

for %%a in (%snr_array%) do (
    for  %%b in (%epoch_array%) do (
        python main.py %%a %%b
    )

)


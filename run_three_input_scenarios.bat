@echo off
REM Multi-scenario training + fixed test: EQ19/EQ20 peak_center 15%% missing
REM epochs=50, nsample=50 fixed

python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_dxta_str.yaml --device cuda:0 --epochs 50 --nsample 50
python exe_strain_multiscenario.py --config base_input_accg_daccg_xta_str.yaml --device cuda:0 --epochs 50 --nsample 50
python exe_strain_multiscenario.py --config base_input_accg_daccg_ddaccg_xta_dxta_ddxta_str.yaml --device cuda:0 --epochs 50 --nsample 50

pause

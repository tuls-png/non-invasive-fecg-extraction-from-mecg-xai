@echo off

echo Running ADFECGDB experiment...
python run_experiment_new.py --dataset adfecgdb --mode full --method all

echo.

echo Running CinC2013 experiment...
python run_experiment_new.py --dataset cinc2013 --mode full --method all

echo.

echo Running NIFECGDB experiment...
python run_experiment_new.py --dataset nifecgdb --mode full --method all

echo.
echo All experiments completed.
pause
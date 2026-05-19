@echo off

call .venv\Scripts\activate

echo Running ADFECGDB...
python run_experiment_new.py --dataset adfecgdb --mode full || goto :error

echo Running NIFECGDB...
python run_experiment_new.py --dataset nifecgdb --mode full || goto :error

echo Running CinC2013...
python run_experiment_new.py --dataset cinc2013 --mode full || goto :error

echo.
echo All experiments completed successfully.
pause
exit /b

:error
echo.
echo Experiment failed.
pause
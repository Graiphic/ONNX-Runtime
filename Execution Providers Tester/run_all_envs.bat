@echo off
REM ------------------------------------------------------------
REM This script assumes that 'conda activate' works after
REM running 'conda init cmd.exe' once.
REM ------------------------------------------------------------

REM Change directory to the "Source" folder (where main.py is located)
cd /d "%~dp0\Source"

REM List of conda environments to test
SET ENV_LIST=^
    onnxruntime-dml-1-22-0 ^
    onnxruntime-onednn-1-22-0 ^
    onnxruntime-openvino-1-22-0 ^
    onnxruntime-trt-1-22-0

FOR %%E IN (%ENV_LIST%) DO (
    echo.
    echo --------------------------------------------
    echo Activating conda env: %%E
    echo --------------------------------------------
    CALL conda activate %%E

    IF ERRORLEVEL 1 (
        echo [ERROR] Could not activate conda env %%E.
        GOTO End
    )

    echo Running "python main.py" in env %%E...
    python main.py

    IF ERRORLEVEL 1 (
        echo [WARNING] "python main.py" returned an error in env %%E.
    ) ELSE (
        echo [OK] "python main.py" completed successfully in env %%E.
    )

    CALL conda deactivate
    echo --------------------------------------------
    echo Finished with env: %%E
    echo --------------------------------------------
)

:End
echo.
echo All done. Press any key to exit...
pause >nul

@echo off
REM Script de instalação e execução do Pipeline de Classificação Astronômica SDSS
REM Inteligência Computacional - CEFET-MG

echo ==================================
echo   Pipeline SDSS - Setup ^& Run
echo ==================================
echo.

REM Verificar se Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo X Python nao encontrado. Por favor, instale Python 3.8 ou superior.
    pause
    exit /b 1
)

for /f "tokens=*" %%i in ('python --version') do set PYTHON_VERSION=%%i
echo + Python encontrado: %PYTHON_VERSION%
echo.

REM Criar ambiente virtual
if not exist "venv" (
    echo Criando ambiente virtual...
    python -m venv venv
    echo + Ambiente virtual criado.
) else (
    echo + Ambiente virtual ja existe.
)

echo.
echo Ativando ambiente virtual...
call venv\Scripts\activate.bat

echo.
echo Instalando dependencias...
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo + Dependencias instaladas com sucesso!
echo.

echo Executando pipeline...
echo ==================================
cd src
python main.py
cd ..

echo.
echo ==================================
echo + Pipeline concluido!
echo Resultados salvos em 'output/'
echo ==================================
echo.
pause

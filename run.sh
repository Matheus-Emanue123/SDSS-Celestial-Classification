#!/bin/bash

# Script de instalação e execução do Pipeline de Classificação Astronômica SDSS
# Inteligência Computacional - CEFET-MG

set -e  # Exit on error

echo "=================================="
echo "  Pipeline SDSS - Setup & Run"
echo "=================================="
echo ""

# Verificar se Python está instalado
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 não encontrado. Por favor, instale Python 3.8 ou superior."
    exit 1
fi

echo "✓ Python encontrado: $(python3 --version)"
echo ""

# Criar ambiente virtual (opcional, mas recomendado)
if [ ! -d "venv" ]; then
    echo "📦 Criando ambiente virtual..."
    python3 -m venv venv
    echo "✓ Ambiente virtual criado."
else
    echo "✓ Ambiente virtual já existe."
fi

echo ""
echo "🔧 Ativando ambiente virtual..."
source venv/bin/activate

echo ""
echo "📥 Instalando dependências..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✓ Dependências instaladas com sucesso!"
echo ""

echo "🚀 Executando pipeline..."
echo "=================================="
cd src/
python main.py
cd ..

echo ""
echo "=================================="
echo "✅ Pipeline concluído!"
echo "📊 Resultados salvos em 'output/'"
echo "=================================="

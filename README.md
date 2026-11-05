# LLM Explorer — Plataforma Educacional de Transformers
**Link da Landing:https://frontend-sepia-six-41.vercel.app/**


Plataforma interativa para aprender **tudo** sobre Large Language Models: do treinamento a inferencia, com visualizacoes 3D, formulas matematicas e explicacoes passo a passo.

## Modulos

| # | Modulo | O que ensina |
|---|--------|-------------|
| 1 | O que sao LLMs | Historia, timeline RNN→LSTM→Transformer, fundamentos |
| 2 | Tokenizacao | BPE interativo, comparacao de tokenizadores |
| 3 | Embeddings | Espaco vetorial 3D, positional encoding, distancia coseno |
| 4 | Arquitetura | Transformer completo: FFN, LayerNorm, residual connections |
| 5 | Atencao | Self-Attention passo a passo, Multi-Head, heatmaps 3D |
| 6 | Treinamento | Loss surface 3D, gradient descent, mini-modelo treinavel |
| 7 | Inferencia | Temperature, top-k, top-p, beam search, geracao ao vivo |
| 8 | Fine-tuning | RLHF, LoRA, transfer learning |
| 9 | Laboratorio | Pipeline completo interativo, comparador de modelos |

## Stack

- **Backend**: Python 3.11+, FastAPI, NumPy, SciPy, scikit-learn
- **Frontend**: React 18, TypeScript, Vite, Tailwind CSS, Three.js (R3F), Plotly.js, KaTeX, Framer Motion

## Quick Start

```bash
# Tudo de uma vez (cria venv, instala deps, inicia ambos os servidores)
git clone https://github.com/Appa019/llm-visualizer.git
cd llm-visualizer
./start.sh
```

Ou separadamente:

```bash
# Backend
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

# Frontend
cd frontend
npm install
npm run dev
```

Acesse:
- **Frontend**: http://localhost:5173 (a landing page fica em `/home`)
- **API Docs**: http://localhost:8000/docs

## Deploy da landing

O site publicado na Vercel não tem backend, então ele serve só a landing page. O build com
`VITE_LANDING_ONLY=1` (variável já configurada no projeto Vercel de produção) monta apenas
`src/pages/LandingPage.tsx`, sem carregar os módulos 3D:

```bash
cd frontend
VITE_LANDING_ONLY=1 npm run build
```

## Estrutura

```
backend/
  main.py           # FastAPI + CORS + routers
  core/             # Logica de dominio (simulador, trainer, gerador)
  routers/          # Endpoints REST (tokenization, embeddings, attention, training, inference)
  utils/            # Tokenizador, math helpers
frontend/
  src/
    pages/          # 10 paginas educacionais
    components/
      viz/          # EmbeddingSpace 3D, Heatmap3D, LossSurface, AttentionFlow, NetworkGraph
      education/    # EducationalViz, FormulaBlock (KaTeX), StepByStep, WhyItMatters
      layout/       # Sidebar, Header, Layout
      ui/           # Card, Toggle, Slider, StepIndicator
    api/client.ts   # Axios client
```

## Principio Educativo

Toda visualizacao segue o padrao **"Veja, Entenda, Aplique"**:

1. **Legenda**: explica o que o grafico mostra em linguagem simples
2. **Formula**: renderizada com KaTeX, variaveis clicaveis com cores consistentes (Q=vermelho, K=verde, V=azul)
3. **"Por que importa?"**: conexao com uso pratico real
4. **Interatividade**: rotacao 3D, hover com detalhes, sliders para parametros

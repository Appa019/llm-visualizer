import { useEffect, useRef, useState } from 'react'
import './landing.css'

const GITHUB_URL = 'https://github.com/Appa019/llm-visualizer'
const CLONE_COMMANDS = [
  'git clone https://github.com/Appa019/llm-visualizer.git',
  'cd llm-visualizer',
  './start.sh',
].join('\n')

// Real screenshots of the app, cropped from the running modules.
const telas = [
  {
    nome: 'Tokenização',
    imagem: '/landing/tokenizacao.webp',
    alt: 'Tokenizador interativo: a frase "O gato dormiu no tapete" dividida em cinco tokens, com o índice de cada um numa tabela.',
    texto: 'O texto vira tokens e IDs na frente de você. O módulo tem BPE interativo e uma comparação entre tokenizadores.',
  },
  {
    nome: 'Embeddings',
    imagem: '/landing/embeddings.webp',
    alt: 'Espaço de embeddings em 3D com palavras como rei, rainha, homem, mulher, feliz e triste posicionadas como pontos.',
    texto: 'Palavras como pontos num espaço 3D que dá para girar. Também mostra positional encoding e distância cosseno.',
  },
  {
    nome: 'Atenção',
    imagem: '/landing/atencao-3d.webp',
    alt: 'Pesos de atenção em 3D: os tokens da frase "o gato pulou no jardim" em círculo, ligados por arcos.',
    texto: 'Self-attention passo a passo, multi-head e os pesos em 3D: cada arco liga dois tokens da frase.',
  },
  {
    nome: 'Treinamento',
    imagem: '/landing/treinamento.webp',
    alt: 'Superfície de loss em 3D com um slider de learning rate acima.',
    texto: 'Superfície de loss em 3D com gradient descent descendo até o mínimo. O learning rate fica num slider.',
  },
]

const modulos = [
  ['O que são LLMs', 'História, a linha do tempo RNN → LSTM → Transformer e os fundamentos.'],
  ['Tokenização', 'BPE interativo e comparação de tokenizadores.'],
  ['Embeddings', 'Espaço vetorial 3D, positional encoding e distância cosseno.'],
  ['Arquitetura', 'O transformer completo: FFN, LayerNorm e conexões residuais.'],
  ['Atenção', 'Self-attention passo a passo, multi-head e heatmaps 3D.'],
  ['Treinamento', 'Superfície de loss 3D, gradient descent e um mini-modelo que treina na hora.'],
  ['Inferência', 'Temperatura, top-k, top-p, beam search e geração ao vivo.'],
  ['Fine-tuning', 'RLHF, LoRA e transfer learning.'],
  ['Laboratório', 'O pipeline completo, interativo, e um comparador de modelos.'],
]

const roteiro = [
  ['Legenda', 'O que o gráfico mostra, em linguagem simples.'],
  ['Fórmula', 'Renderizada com KaTeX, com variáveis clicáveis e cores fixas: Q em vermelho, K em verde, V em azul.'],
  ['Por que importa', 'A ligação entre o conceito e o uso real dos modelos.'],
  ['Interação', 'Girar em 3D, passar o mouse para ver detalhes e mudar parâmetros em sliders.'],
]

// Fixed logits used by the backend temperature demo (TextGenerator.demonstrar_temperatura,
// rng seed 0, 126-word vocabulary), sorted high to low. Only the first 10 are shown.
const TOKENS = ['deles', 'depois', 'qual', 'sem', 'essa', 'lhes', 'meu', 'minhas', 'pela', 'só']
const LOGITS = [
  2.002, 1.96, 1.822, 1.802, 1.739, 1.689, 1.583, 1.574, 1.514, 1.493, 1.458, 1.421, 1.366, 1.346, 1.32, 1.315,
  1.304, 1.295, 1.165, 1.043, 1.031, 1.004, 0.99, 0.947, 0.935, 0.903, 0.873, 0.844, 0.844, 0.788, 0.784, 0.781,
  0.726, 0.696, 0.694, 0.684, 0.656, 0.64, 0.633, 0.63, 0.621, 0.581, 0.541, 0.503, 0.43, 0.412, 0.395, 0.386,
  0.362, 0.357, 0.355, 0.352, 0.329, 0.264, 0.25, 0.22, 0.215, 0.202, 0.189, 0.161, 0.126, 0.109, 0.105, 0.094,
  0.076, 0.052, 0.049, 0.041, -0.004, -0.076, -0.129, -0.13, -0.132, -0.135, -0.159, -0.164, -0.209, -0.219,
  -0.259, -0.287, -0.314, -0.316, -0.378, -0.433, -0.436, -0.458, -0.496, -0.536, -0.544, -0.582, -0.586, -0.618,
  -0.623, -0.633, -0.654, -0.662, -0.662, -0.665, -0.704, -0.713, -0.732, -0.735, -0.743, -0.755, -0.758, -0.77,
  -0.922, -1.01, -1.074, -1.091, -1.17, -1.184, -1.208, -1.246, -1.259, -1.265, -1.278, -1.28, -1.288, -1.32,
  -1.341, -1.402, -1.427, -2.204, -2.25, -2.325,
]

// Same softmax(logits / T) as the backend; LOGITS[0] is the max, subtracted for stability.
function softmax(logits: number[], temperatura: number): number[] {
  const exps = logits.map((l) => Math.exp((l - logits[0]) / temperatura))
  const soma = exps.reduce((a, b) => a + b, 0)
  return exps.map((e) => e / soma)
}

const pct = new Intl.NumberFormat('pt-BR', { style: 'percent', minimumFractionDigits: 1, maximumFractionDigits: 1 })
const dec = new Intl.NumberFormat('pt-BR', { minimumFractionDigits: 1 })

function Temperatura() {
  const [t, setT] = useState(1)
  const probs = softmax(LOGITS, t).slice(0, TOKENS.length)
  // Bars are scaled to the most likely token so the shape stays readable at high T; the % is absolute.
  const maior = probs[0]

  return (
    <div>
      <label htmlFor="controle-temperatura" className="flex items-baseline justify-between text-lg font-semibold">
        Temperatura
        <output htmlFor="controle-temperatura" className="text-[1.75rem] font-extrabold tabular-nums">
          {dec.format(t)}
        </output>
      </label>
      <input
        id="controle-temperatura"
        type="range"
        min={0.1}
        max={2}
        step={0.1}
        value={t}
        onChange={(e) => setT(Number(e.target.value))}
        className="mt-3 h-11 w-full cursor-pointer"
      />
      <ol className="mt-6 space-y-2" aria-label="Probabilidade de cada token">
        {TOKENS.map((token, i) => (
          <li key={token} className="grid grid-cols-[5.5rem_1fr_4rem] items-center gap-3">
            <span className="font-semibold">{token}</span>
            <span className="h-5 w-full" aria-hidden="true">
              <span className="bar block h-full bg-[var(--ink)]" style={{ transform: `scaleX(${probs[i] / maior})` }} />
            </span>
            <span className="text-right tabular-nums">{pct.format(probs[i])}</span>
          </li>
        ))}
      </ol>
    </div>
  )
}

function Telas() {
  const [ativa, setAtiva] = useState(0)
  const tela = telas[ativa]

  return (
    <>
      <div className="lg:col-span-5" data-reveal>
        <h2 id="telas-titulo" className="display text-[clamp(2.25rem,4vw,3.75rem)]">
          As telas da plataforma
        </h2>
      </div>
      <div className="mt-6 flex flex-wrap gap-x-6 gap-y-1 lg:col-span-7 lg:mt-0 lg:self-end" data-reveal style={{ '--d': '80ms' } as React.CSSProperties}>
        {telas.map((item, i) => (
          <button
            key={item.nome}
            type="button"
            aria-pressed={i === ativa}
            aria-controls="tela-atual"
            onClick={() => setAtiva(i)}
            className={`min-h-11 text-xl font-bold transition-colors lg:text-[1.75rem] ${
              i === ativa ? 'text-[var(--ink)] underline decoration-2 underline-offset-8' : 'text-[var(--muted)] hover:text-[var(--ink)]'
            }`}
          >
            {item.nome}
          </button>
        ))}
      </div>
      <figure id="tela-atual" className="mt-10 lg:col-span-12 lg:mt-14 lg:grid lg:grid-cols-12 lg:gap-x-6" aria-live="polite" data-reveal style={{ '--d': '160ms' } as React.CSSProperties}>
        <img
          key={tela.imagem}
          src={tela.imagem}
          alt={tela.alt}
          width={1520}
          height={950}
          className="swap h-[280px] w-full object-cover object-left-top sm:h-auto lg:col-span-9"
        />
        <figcaption className="mt-4 lg:col-span-3 lg:mt-0 lg:self-end">
          <span className="block text-[1.75rem] font-bold">{tela.nome}</span>
          <span className="mt-2 block">{tela.texto}</span>
        </figcaption>
      </figure>
    </>
  )
}

export default function LandingPage() {
  const raiz = useRef<HTMLDivElement>(null)
  const [copiado, setCopiado] = useState(false)

  // Arm the entrance animation only when motion is allowed; content stays visible otherwise.
  useEffect(() => {
    const el = raiz.current
    if (!el || window.matchMedia('(prefers-reduced-motion: reduce)').matches) return
    el.classList.add('reveal-on')
    const observador = new IntersectionObserver(
      (entradas) => {
        for (const entrada of entradas) {
          if (entrada.isIntersecting) {
            entrada.target.classList.add('is-in')
            observador.unobserve(entrada.target)
          }
        }
      },
      { rootMargin: '0px 0px -10% 0px' },
    )
    el.querySelectorAll('[data-reveal]').forEach((n) => observador.observe(n))
    return () => observador.disconnect()
  }, [])

  async function copiar() {
    try {
      await navigator.clipboard.writeText(CLONE_COMMANDS)
      setCopiado(true)
      setTimeout(() => setCopiado(false), 2000)
    } catch {
      // Clipboard can be blocked (insecure context, permissions); the commands stay selectable.
    }
  }

  const container = 'mx-auto w-full max-w-[1440px] px-5 sm:px-8 lg:px-12'
  const grade = `${container} lg:grid lg:grid-cols-12 lg:gap-x-6`

  return (
    <div ref={raiz} className="landing min-h-svh">
      <header className={`${grade} flex items-center justify-between py-6`}>
        <a href="#topo" className="text-lg font-extrabold lg:col-span-5">
          LLM Explorer
        </a>
        <nav aria-label="Seções" className="hidden gap-8 md:flex lg:col-span-4">
          <a href="#telas" className="hover:underline">Telas</a>
          <a href="#modulos" className="hover:underline">Módulos</a>
          <a href="#rodar" className="hover:underline">Rodar</a>
        </nav>
        <a href={GITHUB_URL} className="link text-right lg:col-span-3" target="_blank" rel="noopener noreferrer">
          Código no GitHub
        </a>
      </header>

      <main id="topo">
        <section className={`${grade} pt-4 lg:pt-6`} aria-labelledby="titulo">
          <figure className="lg:col-span-12 lg:row-start-1 lg:grid lg:grid-cols-12 lg:gap-x-6">
            <img
              src="/landing/atencao-mapa.webp"
              alt="Mapa de atenção da frase “o gato pulou no jardim”: uma grade 5 por 5 em tons de vermelho a amarelo, com o peso de cada par de tokens escrito na célula."
              width={1660}
              height={690}
              className="h-auto max-h-[46vh] w-full object-contain object-left lg:col-span-9"
            />
            <figcaption className="mt-3 text-base text-[var(--muted)] lg:col-span-3 lg:mt-0 lg:self-end">
              Mapa de atenção, como aparece na plataforma. Cada célula é o peso que um token da frase “o gato pulou no jardim” dá a outro.
            </figcaption>
          </figure>

          <h1
            id="titulo"
            className="display mt-10 text-[clamp(2.25rem,10.6vw,5.25rem)] lg:col-span-7 lg:col-start-6 lg:row-start-2 lg:mt-16 lg:text-[clamp(2.75rem,5.6vw,5.25rem)]"
          >
            <span className="mask-line"><span>Transformers</span></span>
            <span className="mask-line"><span>vistos por dentro</span></span>
          </h1>
          <p className="mt-6 max-w-[34ch] text-xl lg:col-span-5 lg:col-start-1 lg:row-start-2 lg:mt-0 lg:self-end">
            Uma plataforma aberta, em português, para estudar LLMs do texto à geração, com visualizações 3D, fórmulas e um simulador que roda sem GPU.
          </p>
          <div className="mt-8 flex flex-wrap items-center gap-x-8 gap-y-4 lg:col-span-7 lg:col-start-6 lg:row-start-3">
            <a href={GITHUB_URL} className="btn" target="_blank" rel="noopener noreferrer">
              Ver o código no GitHub
            </a>
            <a href="#rodar" className="link">
              Como rodar na sua máquina
            </a>
          </div>
        </section>

        <section id="telas" className={`${grade} mt-32 lg:mt-48`} aria-labelledby="telas-titulo">
          <Telas />
        </section>

        <section id="modulos" className={`${grade} mt-32 lg:mt-48`} aria-labelledby="modulos-titulo">
          <div className="lg:col-span-5">
            <div className="lg:sticky lg:top-10" data-reveal>
              <h2 id="modulos-titulo" className="display text-[clamp(2.25rem,4vw,3.75rem)]">
                Nove módulos, do texto à geração
              </h2>
              <figure className="mt-10 hidden lg:block">
                <img
                  src="/landing/arquitetura.webp"
                  alt="Diagrama de um bloco decoder do módulo de arquitetura: Layer Norm, Multi-Head Attention, Add & Norm, Feed Forward e Add & Norm empilhados, com as conexões residuais ao lado."
                  width={500}
                  height={994}
                  loading="lazy"
                  className="h-auto w-[250px]"
                />
                <figcaption className="mt-3 max-w-[30ch] text-base text-[var(--muted)]">
                  Um bloco decoder, como aparece no módulo de arquitetura.
                </figcaption>
              </figure>
            </div>
          </div>
          <ol className="mt-10 space-y-10 lg:col-span-7 lg:mt-0">
            {modulos.map(([titulo, texto], i) => (
              <li key={titulo} data-reveal style={{ '--d': `${(i % 3) * 70}ms` } as React.CSSProperties}>
                <h3 className="text-[1.75rem] font-bold leading-tight">{titulo}</h3>
                <p className="mt-2 max-w-[52ch]">{texto}</p>
              </li>
            ))}
          </ol>
        </section>

        <section id="temperatura" className={`${grade} mt-32 lg:mt-48`} aria-labelledby="temperatura-titulo">
          <div className="lg:col-span-5 lg:col-start-8 lg:row-start-1" data-reveal>
            <h2 id="temperatura-titulo" className="display text-[clamp(2.25rem,4vw,3.75rem)]">
              Mexa na temperatura
            </h2>
            <p className="mt-6 max-w-[42ch]">
              Estes são os logits fixos que o simulador usa no módulo de inferência. A probabilidade de cada token é
              softmax(logits / T): com T baixo ela se concentra nos primeiros, com T alto se espalha pelo vocabulário de 126 palavras.
            </p>
          </div>
          <div className="mt-10 lg:col-span-6 lg:col-start-1 lg:row-start-1 lg:mt-0" data-reveal style={{ '--d': '100ms' } as React.CSSProperties}>
            <Temperatura />
          </div>
        </section>

        <section id="metodo" className={`${grade} mt-32 lg:mt-48`} aria-labelledby="metodo-titulo">
          <div className="lg:col-span-5" data-reveal>
            <h2 id="metodo-titulo" className="display text-[clamp(2.25rem,4vw,3.75rem)]">
              Veja, entenda, aplique
            </h2>
            <figure className="mt-10 lg:mt-14">
              <img
                src="/landing/atencao-formula.webp"
                alt="Fórmula do peso de atenção: w_ij = softmax(q_i · k_j / raiz de d_k), com q em vermelho e k em verde."
                width={460}
                height={140}
                loading="lazy"
                className="h-auto w-full max-w-[460px]"
              />
              <figcaption className="mt-3 max-w-[36ch] text-base text-[var(--muted)]">
                A fórmula do mapa de atenção, como aparece no módulo, com q e k nas cores fixas.
              </figcaption>
            </figure>
          </div>
          <div className="mt-6 lg:col-span-7 lg:mt-0" data-reveal style={{ '--d': '80ms' } as React.CSSProperties}>
            <p className="max-w-[46ch] text-xl">Toda visualização da plataforma vem com as mesmas quatro camadas.</p>
            <dl className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-2">
              {roteiro.map(([termo, definicao]) => (
                <div key={termo}>
                  <dt className="text-xl font-bold">{termo}</dt>
                  <dd className="mt-1">{definicao}</dd>
                </div>
              ))}
            </dl>
          </div>
        </section>

        <section id="rodar" className={`${grade} mt-32 lg:mt-48`} aria-labelledby="rodar-titulo">
          <div className="lg:col-span-5" data-reveal>
            <h2 id="rodar-titulo" className="display text-[clamp(2.25rem,4vw,3.75rem)]">
              Rode na sua máquina
            </h2>
            <p className="mt-6 max-w-[42ch]">
              O backend em FastAPI traz um simulador de transformer feito com NumPy, então tudo roda sem GPU. O script{' '}
              <code>start.sh</code> cria o ambiente Python, instala as dependências e sobe os dois servidores.
            </p>
          </div>
          <div className="mt-10 lg:col-span-7 lg:mt-0" data-reveal style={{ '--d': '100ms' } as React.CSSProperties}>
            <pre className="overflow-x-auto bg-[#f4f4f5] p-6 text-base leading-relaxed">
              <code>{CLONE_COMMANDS}</code>
            </pre>
            <div className="mt-4 flex flex-wrap items-center justify-between gap-4">
              <p>Depois, abra localhost:5173. Precisa de Python 3.11+ e Node 18+.</p>
              <button type="button" onClick={copiar} className="link min-h-11" aria-live="polite">
                {copiado ? 'Comandos copiados' : 'Copiar comandos'}
              </button>
            </div>
            <p className="mt-10 max-w-[56ch] text-[var(--muted)]">
              Feito com React 18, TypeScript, Vite, Tailwind CSS, Three.js, Plotly.js, KaTeX e Framer Motion no frontend;
              Python, FastAPI, NumPy, SciPy e scikit-learn no backend.
            </p>
          </div>
        </section>
      </main>

      <footer className={`${container} mt-32 pb-8 lg:mt-48`}>
        <div className="flex flex-wrap gap-x-8 gap-y-3">
          <a href={GITHUB_URL} className="link" target="_blank" rel="noopener noreferrer">Código no GitHub</a>
          <a href={`${GITHUB_URL}/blob/main/LICENSE`} className="link" target="_blank" rel="noopener noreferrer">Licença MIT</a>
          <a href="https://pedropestana.dev" className="link" target="_blank" rel="noopener noreferrer">Feito por Pedro Pestana</a>
        </div>
        <p className="display mt-16 whitespace-nowrap text-[13.6vw] min-[1440px]:text-[196px] leading-none" aria-hidden="true">
          LLM Explorer
        </p>
      </footer>
    </div>
  )
}

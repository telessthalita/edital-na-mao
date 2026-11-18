import { FormEvent, useEffect, useRef, useState } from "react";
import { api, Edital, RespostaAgente, Saude } from "./api";
import { Resposta } from "./components/Resposta";

type Msg = { id: number; role: "user" } & { texto: string } | { id: number; role: "assistant"; r: RespostaAgente };

const SUGESTOES = [
  "Quais matérias caem pra Analista de TI e quantas questões cada uma tem?",
  "Quantos dias faltam pra prova?",
  "Monta um cronograma de estudo com 12 horas por semana",
  "Compara o salário de analista entre os dois editais",
  "Tem isenção da taxa de inscrição? Quem pode pedir?",
];

let proximoId = 1;

export default function App() {
  const [editais, setEditais] = useState<Edital[]>([]);
  const [saude, setSaude] = useState<Saude | null>(null);
  const [editalId, setEditalId] = useState<string | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>([]);
  const [entrada, setEntrada] = useState("");
  const [pensando, setPensando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);
  const fim = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.saude().then(setSaude).catch((e) => setErro(e.message));
    api.editais().then(setEditais).catch(() => {});
  }, []);

  useEffect(() => {
    fim.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [msgs, pensando]);

  const historico = () =>
    msgs.slice(-8).map((m) => (m.role === "user" ? { role: "user", content: m.texto } : { role: "assistant", content: m.r.resposta }));

  async function enviar(texto: string) {
    const pergunta = texto.trim();
    if (!pergunta || pensando) return;
    setErro(null);
    setEntrada("");
    const hist = historico();
    setMsgs((m) => [...m, { id: proximoId++, role: "user", texto: pergunta }]);
    setPensando(true);
    try {
      const r = await api.conversar(pergunta, hist, editalId);
      setMsgs((m) => [...m, { id: proximoId++, role: "assistant", r }]);
    } catch (e) {
      setErro((e as Error).message);
    } finally {
      setPensando(false);
    }
  }

  const atual = editais.find((e) => e.id === editalId);

  return (
    <div className="app">
      <aside className="lateral">
        <div className="marca">
          <span className="marca-texto">Edital</span> na mão
        </div>
        <p className="lema">Pergunta pro edital. A resposta vem com a página.</p>

        <nav className="editais" aria-label="Editais">
          <h2>Editais</h2>
          <button className={editalId === null ? "edital ativo" : "edital"} onClick={() => setEditalId(null)}>
            <span>Todos os editais</span>
            <small>{editais.length} carregados</small>
          </button>
          {editais.map((e) => (
            <button key={e.id} className={editalId === e.id ? "edital ativo" : "edital"} onClick={() => setEditalId(e.id)}>
              <span>{e.titulo}</span>
              <small>{e.paginas} páginas · {e.total_trechos} trechos</small>
            </button>
          ))}
        </nav>

        {saude?.upload && (
          <Upload
            onEnviado={(e) => {
              setEditais((l) => [...l.filter((x) => x.id !== e.id), e]);
              setEditalId(e.id);
            }}
          />
        )}

        <footer className="rodape">
          {saude && !saude.ia && <p className="aviso">IA desligada: falta configurar a GEMINI_API_KEY no servidor.</p>}
          <p>
            Os editais de exemplo são fictícios. Confere sempre no edital oficial antes de decidir qualquer coisa.
          </p>
          <p>
            Feito por <a href="https://telessthalita.com.br" target="_blank" rel="noreferrer">Thalita Teles</a>
          </p>
        </footer>
      </aside>

      <main className="conversa">
        <header className="topo">
          <h1>{atual ? atual.titulo : "Todos os editais"}</h1>
          {msgs.length > 0 && (
            <button className="limpar" onClick={() => setMsgs([])}>
              Nova conversa
            </button>
          )}
        </header>

        <div className="mensagens" aria-live="polite">
          {msgs.length === 0 && (
            <div className="vazio-inicial">
              <p className="convite">
                Edital tem 80 páginas e a informação que você precisa está no anexo III. Pergunta aqui que eu acho, e te
                mostro de onde tirei.
              </p>
              <div className="sugestoes">
                {SUGESTOES.map((s) => (
                  <button key={s} onClick={() => enviar(s)} disabled={pensando || !saude?.ia}>
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}
          {msgs.map((m) =>
            m.role === "user" ? (
              <div key={m.id} className="msg usuario">
                {m.texto}
              </div>
            ) : (
              <div key={m.id} className="msg assistente">
                <Resposta r={m.r} />
              </div>
            ),
          )}
          {pensando && (
            <div className="msg assistente pensando">
              <span className="ponto" />
              <span className="ponto" />
              <span className="ponto" />
              <span className="pensando-texto">Lendo o edital… pode levar uns 20 segundos</span>
            </div>
          )}
          {erro && <div className="erro-msg">{erro}</div>}
          <div ref={fim} />
        </div>

        <form
          className="entrada"
          onSubmit={(e: FormEvent) => {
            e.preventDefault();
            enviar(entrada);
          }}
        >
          <textarea
            value={entrada}
            onChange={(e) => setEntrada(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                enviar(entrada);
              }
            }}
            placeholder={atual ? `Pergunta sobre ${atual.titulo}` : "Pergunta sobre qualquer edital carregado"}
            rows={2}
            maxLength={2000}
            aria-label="Sua pergunta"
          />
          <button type="submit" disabled={pensando || !entrada.trim()}>
            Perguntar
          </button>
        </form>
      </main>
    </div>
  );
}

function Upload({ onEnviado }: { onEnviado: (e: Edital) => void }) {
  const [aberto, setAberto] = useState(false);
  const [arquivo, setArquivo] = useState<File | null>(null);
  const [titulo, setTitulo] = useState("");
  const [fonte, setFonte] = useState("");
  const [enviando, setEnviando] = useState(false);
  const [erro, setErro] = useState<string | null>(null);

  if (!aberto) {
    return (
      <button className="abrir-upload" onClick={() => setAberto(true)}>
        + Subir um edital (PDF)
      </button>
    );
  }

  async function enviar(e: FormEvent) {
    e.preventDefault();
    if (!arquivo) return;
    setEnviando(true);
    setErro(null);
    try {
      const ed = await api.enviarEdital(arquivo, titulo, fonte);
      onEnviado(ed);
      setAberto(false);
      setArquivo(null);
      setTitulo("");
      setFonte("");
    } catch (err) {
      setErro((err as Error).message);
    } finally {
      setEnviando(false);
    }
  }

  return (
    <form className="upload" onSubmit={enviar}>
      <label>
        PDF do edital
        <input
          type="file"
          accept="application/pdf"
          required
          onChange={(e) => {
            const f = e.target.files?.[0] ?? null;
            setArquivo(f);
            if (f && !titulo) setTitulo(f.name.replace(/\.pdf$/i, ""));
          }}
        />
      </label>
      <label>
        Nome
        <input value={titulo} onChange={(e) => setTitulo(e.target.value)} required minLength={3} maxLength={120} />
      </label>
      <label>
        Link oficial (opcional)
        <input value={fonte} onChange={(e) => setFonte(e.target.value)} type="url" placeholder="https://..." />
      </label>
      {erro && <p className="erro-msg">{erro}</p>}
      <div className="upload-acoes">
        <button type="button" onClick={() => setAberto(false)} disabled={enviando}>
          Cancelar
        </button>
        <button type="submit" disabled={enviando || !arquivo}>
          {enviando ? "Lendo o PDF…" : "Enviar"}
        </button>
      </div>
    </form>
  );
}

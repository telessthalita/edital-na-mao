import { useRef, useState } from "react";
import type { RespostaAgente } from "../api";
import { Texto } from "./Texto";

const NOMES: Record<string, string> = {
  buscar_trecho: "buscou no edital",
  calcular_dias_ate_prova: "calculou os dias",
  montar_cronograma: "montou o cronograma",
  comparar_editais: "comparou editais",
  listar_editais: "listou os editais",
};

export function Resposta({ r }: { r: RespostaAgente }) {
  const [ativa, setAtiva] = useState<number | null>(null);
  const fontes = useRef<HTMLOListElement>(null);
  const validas = new Set(r.citacoes.map((c) => c.n));

  const mostrar = (n: number) => {
    setAtiva(n);
    fontes.current?.querySelector(`[data-n="${n}"]`)?.scrollIntoView({ behavior: "smooth", block: "nearest" });
  };

  return (
    <div className="resposta">
      <Texto texto={r.resposta} onRef={mostrar} refsValidas={validas} />

      {r.citacoes.length > 0 && (
        <section className="fontes" aria-label="Fontes">
          <h3>De onde saiu</h3>
          <ol ref={fontes}>
            {r.citacoes.map((c) => (
              <li key={c.n} data-n={c.n} className={ativa === c.n ? "ativa" : undefined}>
                <div className="fonte-topo">
                  <span className="ref ref-estatica">{c.n}</span>
                  <span className="pagina">pág. {c.pagina}</span>
                  <span className="edital-nome">{c.edital}</span>
                  {c.fonte_url && (
                    <a href={`${c.fonte_url}#page=${c.pagina}`} target="_blank" rel="noreferrer">
                      abrir PDF
                    </a>
                  )}
                </div>
                {c.secao && !c.trecho.startsWith(c.secao) && <div className="secao">{c.secao}</div>}
                <blockquote>{c.trecho}</blockquote>
              </li>
            ))}
          </ol>
        </section>
      )}

      <details className="trace">
        <summary>
          Como cheguei nisso <span className="meta">· {r.trace.length} {r.trace.length === 1 ? "ferramenta" : "ferramentas"} · {(r.ms / 1000).toFixed(1)} s · {r.modelo}</span>
        </summary>
        {r.trace.length === 0 ? (
          <p className="vazio">Respondi sem chamar ferramenta nenhuma.</p>
        ) : (
          <ol>
            {r.trace.map((t, i) => (
              <li key={i} className={t.erro ? "erro" : undefined}>
                <div>
                  <code>{t.ferramenta}</code> <span className="acao">{NOMES[t.ferramenta] ?? ""}</span>
                  <span className="ms">{t.ms} ms</span>
                </div>
                <pre>{JSON.stringify(t.argumentos, null, 1)}</pre>
                <div className="resumo">→ {t.resumo}</div>
              </li>
            ))}
          </ol>
        )}
        {r.limite_atingido && <p className="vazio">Bati o limite de passos e respondi com o que tinha.</p>}
      </details>
    </div>
  );
}

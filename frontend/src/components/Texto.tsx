import type { ReactNode } from "react";

/** Markdown mínimo (parágrafo, lista, negrito) + refs [n] viram botão. Sem innerHTML. */
type Props = { texto: string; onRef: (n: number) => void; refsValidas: Set<number> };

function inline(linha: string, onRef: Props["onRef"], validas: Set<number>, chave: string): ReactNode[] {
  const partes = linha.split(/(\*\*[^*]+\*\*|\[\d+\])/g);
  return partes.map((p, i) => {
    const k = `${chave}-${i}`;
    const ref = p.match(/^\[(\d+)\]$/);
    if (ref) {
      const n = Number(ref[1]);
      if (!validas.has(n)) return null;
      return (
        <button key={k} type="button" className="ref" onClick={() => onRef(n)} aria-label={`Ver fonte ${n}`}>
          {n}
        </button>
      );
    }
    if (p.startsWith("**") && p.endsWith("**")) return <strong key={k}>{p.slice(2, -2)}</strong>;
    return p;
  });
}

export function Texto({ texto, onRef, refsValidas }: Props) {
  const blocos: ReactNode[] = [];
  let lista: { ordenada: boolean; itens: string[] } | null = null;

  const fechaLista = () => {
    if (!lista) return;
    const itens = lista.itens.map((it, i) => <li key={i}>{inline(it, onRef, refsValidas, `li${blocos.length}-${i}`)}</li>);
    blocos.push(lista.ordenada ? <ol key={blocos.length}>{itens}</ol> : <ul key={blocos.length}>{itens}</ul>);
    lista = null;
  };

  for (const bruta of texto.split("\n")) {
    const linha = bruta.trim();
    const item = linha.match(/^([-*•]|\d+[.)])\s+(.*)$/);
    if (item) {
      const ordenada = /\d/.test(item[1]);
      if (!lista || lista.ordenada !== ordenada) {
        fechaLista();
        lista = { ordenada, itens: [] };
      }
      lista.itens.push(item[2]);
      continue;
    }
    fechaLista();
    if (linha) blocos.push(<p key={blocos.length}>{inline(linha.replace(/^#+\s*/, ""), onRef, refsValidas, `p${blocos.length}`)}</p>);
  }
  fechaLista();
  return <div className="texto">{blocos}</div>;
}

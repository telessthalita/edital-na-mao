export type Edital = {
  id: string;
  titulo: string;
  paginas: number;
  total_trechos: number;
  fonte_url: string | null;
  criado_em: string;
};

export type Citacao = {
  n: number;
  edital_id: string;
  edital: string;
  pagina: number;
  secao: string | null;
  trecho: string;
  fonte_url: string | null;
};

export type PassoTrace = {
  passo: number;
  ferramenta: string;
  argumentos: Record<string, unknown>;
  resumo: string;
  ms: number;
  erro: boolean;
};

export type RespostaAgente = {
  resposta: string;
  citacoes: Citacao[];
  trace: PassoTrace[];
  passos: number;
  modelo: string;
  ms: number;
  limite_atingido: boolean;
};

export type Saude = { ok: boolean; banco: string; ia: boolean; modelo: string | null; upload: boolean };

const BASE = import.meta.env.VITE_API_URL ?? "";

async function pedir<T>(caminho: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${caminho}`, init);
  } catch {
    throw new Error("Não consegui falar com o servidor. Se for a demo gratuita, ela pode estar acordando: tenta de novo em 30 segundos.");
  }
  if (!res.ok) {
    let msg = `Erro ${res.status}`;
    try {
      const corpo = await res.json();
      if (typeof corpo.detail === "string") msg = corpo.detail;
      else if (Array.isArray(corpo.detail)) msg = "Confere os campos e tenta de novo.";
    } catch {
      /* resposta sem json */
    }
    throw new Error(msg);
  }
  return res.json() as Promise<T>;
}

export const api = {
  saude: () => pedir<Saude>("/api/health"),
  editais: () => pedir<Edital[]>("/api/editais"),
  enviarEdital: (arquivo: File, titulo: string, fonteUrl: string) => {
    const form = new FormData();
    form.append("arquivo", arquivo);
    form.append("titulo", titulo);
    if (fonteUrl) form.append("fonte_url", fonteUrl);
    return pedir<Edital>("/api/editais", { method: "POST", body: form });
  },
  conversar: (mensagem: string, historico: { role: string; content: string }[], editalId: string | null) =>
    pedir<RespostaAgente>("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mensagem, historico, edital_id: editalId }),
    }),
};

# Spec: Edital na mão

## O problema

Edital de concurso tem 60, 80, 150 páginas. A pessoa precisa saber coisas bem objetivas (quais matérias caem, quando é a prova, qual a escolaridade exigida, quanto custa a inscrição) e acaba lendo resumo de terceiro em vez da fonte. Quando lê a fonte, perde informação que está num anexo lá no fim.

O Edital na mão deixa a pessoa conversar com o edital. Toda resposta aponta a página de onde saiu, então dá pra conferir em vez de confiar.

## O que precisa acontecer

### Ingestão (o R do RAG)
1. Recebe um PDF de edital (upload ou script de seed).
2. Extrai o texto página por página, guardando o número da página.
3. Quebra em pedaços (chunks) de ~900 caracteres com sobreposição, sem atravessar página, e tenta marcar a seção (ex: "6. DAS INSCRIÇÕES", "ANEXO II").
4. Gera embedding de cada chunk (Gemini, `RETRIEVAL_DOCUMENT`) e grava texto, página, seção e vetor no banco.

### Busca
5. Busca híbrida: vetor (similaridade de cosseno no pgvector) + texto (full text search do Postgres em português), combinadas com Reciprocal Rank Fusion. Motivo: edital é cheio de termo exato ("nível superior", "R$ 85,00", "Anexo II") que a busca vetorial às vezes perde, e de pergunta vaga que a busca por palavra perde.

### Agente (tool calling)
6. Um loop de agente escrito na mão (sem LangChain) que manda a conversa e a lista de ferramentas pro Gemini, executa as ferramentas que ele pedir e devolve o resultado, até ele responder em texto ou bater o limite de passos.
7. Ferramentas:
   - `listar_editais()`: quais editais estão carregados.
   - `buscar_trecho(pergunta, edital_id?)`: busca híbrida, devolve trechos com página.
   - `calcular_dias_ate_prova(data_prova)`: dias e semanas até a data, a partir de hoje.
   - `montar_cronograma(materias, horas_por_semana, data_prova)`: divide as horas por matéria proporcional ao peso/quantidade de questões, semana a semana. Conta determinística, sem LLM.
   - `comparar_editais(edital_a, edital_b, aspecto)`: busca o mesmo aspecto nos dois e devolve lado a lado.
8. Regras do agente (prompt de sistema): só afirmar o que veio de `buscar_trecho`, sempre citar a página, dizer "não achei no edital" quando não achar, não dar palpite sobre banca nem resultado.
9. A resposta da API traz: texto, citações (edital, página, trecho) e o trace (cada ferramenta chamada, argumentos, resumo do resultado, tempo).

### Interface
10. Página React com: lista de editais e upload, chat, citações clicáveis na resposta e um painel "como o agente chegou nisso" mostrando o trace.

### Avaliação
11. Um conjunto de perguntas-gabarito (pergunta + página esperada) e um script que mede hit@5 e MRR da busca em três modos: só vetor, só texto e híbrida. O resultado vai pro README.

### Proteção da demo pública
12. Limite de requisições por IP, limite de tamanho/páginas no upload e upload desligável por variável de ambiente (pra cota gratuita do Gemini não estourar).

## Fora do escopo (por enquanto)
- Login e editais por usuário.
- Streaming da resposta.
- OCR de PDF escaneado (só PDF com texto).

## Como saber que ficou pronto
- [x] `pytest` passa: chunking, fusão RRF, ferramentas, loop do agente (com LLM falso) e API.
- [x] Com chave do Gemini e um edital carregado, a pergunta "quais as matérias de conhecimentos específicos e quantos dias faltam pra prova?" faz o agente chamar `buscar_trecho` e depois `calcular_dias_ate_prova`, e a resposta cita página certa.
- [x] Pergunta sobre algo que não está no edital volta "não achei no edital", sem inventar.
- [x] Script de avaliação roda e mostra a tabela vetor x texto x híbrida.
- [x] Frontend mostra resposta, citação e trace, rodando local de ponta a ponta.
- [x] Caminho de deploy combinado com a Thalita (ver abaixo) e documentado.
- [x] README em português com versão em inglês recolhida, explicando arquitetura do agente, do RAG e as decisões.

## Stack e decisões
| Parte | Escolha | Por quê |
| --- | --- | --- |
| API | Python + FastAPI | Ecossistema de IA é Python; já usado no AI Corporate Assistant |
| LLM e embeddings | Gemini via REST (`httpx`) | Tier gratuito, tool calling e embedding no mesmo lugar. REST direto em vez de SDK pra deixar o protocolo visível e não depender de versão de SDK |
| Camada de provedor | Interface `ChatModel` / `Embedder` | Trocar pra Groq, Ollama ou Claude sem mexer no agente |
| Banco | Postgres + pgvector (Supabase) | Vetor e full text no mesmo banco, uma query só |
| Banco local/testes | `MemoryStore` (numpy + BM25) | Roda sem Docker e sem Supabase; testes não dependem de rede |
| Agente | Loop próprio | Dá pra explicar cada linha na entrevista |
| Frontend | React + Vite + TypeScript | Stack do dia a dia |

## Deploy (combinado em 18/11/2026)
- Só Render, web service gratuito via Docker. A API serve o frontend buildado: um serviço só.
- Sem Supabase por enquanto: store em memória, e os editais de exemplo são carregados a cada vez que o serviço sobe.
- Upload desligado e limite de perguntas por IP na demo.
- Publicação fica pra depois, quando a Thalita decidir. Nada pago.
- Supabase (pgvector no ar) fica como evolução; o `PgStore` ainda não rodou contra um Postgres real.

## Plano
1. Esqueleto do repo, config e camada de provedor (Gemini + fakes de teste).
2. Ingestão: PDF, chunking com página e seção. Gerador de edital fictício pra testes.
3. Stores (memória e Postgres) e busca híbrida com RRF.
4. Ferramentas e loop do agente, com testes usando LLM falso.
5. API FastAPI com rate limit e upload.
6. Frontend.
7. Avaliação de retrieval.
8. Teste ponta a ponta com chave real e edital real.
9. README PT/EN e deploy combinado.

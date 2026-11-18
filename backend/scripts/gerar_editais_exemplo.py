"""Gera dois editais FICTÍCIOS em PDF pra testes e pra demo.

Órgãos, bancas e números são inventados. Uso:
    python -m scripts.gerar_editais_exemplo
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "editais"

BOILER = [
    "O candidato deverá acompanhar todas as publicações referentes a este concurso no endereço eletrônico da "
    "organizadora, sendo de sua inteira responsabilidade a verificação de datas, horários e locais.",
    "A inscrição implica o conhecimento e a tácita aceitação das condições estabelecidas neste edital, das quais o "
    "candidato não poderá alegar desconhecimento.",
    "Não serão aceitas solicitações de alteração de cargo após a efetivação da inscrição.",
]

VILA_SERENA = {
    "titulo": "PREFEITURA MUNICIPAL DE VILA SERENA\nCONCURSO PÚBLICO Nº 01/2026\nEDITAL DE ABERTURA",
    "secoes": [
        ("1. DAS DISPOSIÇÕES PRELIMINARES", [
            "1.1 O concurso público será regido por este edital e executado pelo Instituto Exemplo de Seleção (IES), "
            "destinando-se ao provimento de cargos efetivos do quadro de pessoal da Prefeitura Municipal de Vila Serena.",
            "1.2 O prazo de validade do concurso será de 2 (dois) anos, contados da data da homologação do resultado "
            "final, podendo ser prorrogado uma única vez, por igual período.",
            "1.3 Os candidatos nomeados estarão subordinados ao Regime Jurídico Único dos Servidores do Município.",
            *BOILER,
        ]),
        ("2. DOS CARGOS, VAGAS E REMUNERAÇÃO", [
            "2.1 Analista de Tecnologia da Informação: 4 (quatro) vagas de ampla concorrência e cadastro reserva. "
            "Requisito: diploma de curso superior em Ciência da Computação, Sistemas de Informação, Engenharia de "
            "Computação ou Análise e Desenvolvimento de Sistemas. Vencimento inicial: R$ 6.850,00. "
            "Jornada: 40 (quarenta) horas semanais.",
            "2.2 Técnico em Informática: 6 (seis) vagas. Requisito: ensino médio completo e curso técnico em "
            "Informática. Vencimento inicial: R$ 3.420,00. Jornada: 40 (quarenta) horas semanais.",
            "2.3 Assistente Administrativo: 12 (doze) vagas. Requisito: ensino médio completo. Vencimento inicial: "
            "R$ 2.580,00. Jornada: 40 (quarenta) horas semanais.",
            "2.4 Além do vencimento, os servidores farão jus a auxílio-alimentação no valor de R$ 750,00 mensais.",
        ]),
        ("3. DAS INSCRIÇÕES", [
            "3.1 As inscrições serão realizadas exclusivamente pela internet, no período de 5 de janeiro de 2027 a "
            "4 de fevereiro de 2027, no endereço eletrônico da organizadora.",
            "3.2 Valor da inscrição: R$ 120,00 (cento e vinte reais) para cargos de nível superior e R$ 85,00 "
            "(oitenta e cinco reais) para cargos de nível médio.",
            "3.3 O boleto bancário poderá ser pago até o dia 5 de fevereiro de 2027. Pagamentos após essa data não "
            "serão considerados.",
            "3.4 O candidato poderá se inscrever para apenas um cargo, uma vez que as provas serão aplicadas no mesmo "
            "dia e horário.",
            *BOILER,
        ]),
        ("4. DA ISENÇÃO DO VALOR DE INSCRIÇÃO", [
            "4.1 Terá direito à isenção o candidato inscrito no Cadastro Único para Programas Sociais do Governo "
            "Federal (CadÚnico) e o candidato doador de medula óssea cadastrado no REDOME.",
            "4.2 O pedido de isenção deverá ser feito entre 5 e 12 de janeiro de 2027, com o envio da documentação "
            "comprobatória.",
            "4.3 O resultado dos pedidos de isenção será divulgado em 20 de janeiro de 2027.",
        ]),
        ("5. DAS VAGAS RESERVADAS", [
            "5.1 Do total de vagas de cada cargo, 5% (cinco por cento) serão reservadas a pessoas com deficiência e "
            "20% (vinte por cento) a pessoas pretas e pardas, na forma da legislação vigente.",
            "5.2 O candidato que se declarar pessoa com deficiência deverá enviar laudo médico emitido nos últimos "
            "12 (doze) meses.",
        ]),
        ("6. DAS PROVAS", [
            "6.1 A prova objetiva será aplicada no dia 14 de março de 2027, das 9h às 13h, horário de Brasília, "
            "na cidade de Vila Serena. Os locais de prova serão divulgados em 5 de março de 2027.",
            "6.2 O candidato deverá comparecer com antecedência mínima de 60 minutos, munido de caneta esferográfica "
            "de tinta preta e de documento oficial de identificação com foto.",
            "6.3 Analista de Tecnologia da Informação, total de 60 questões: Língua Portuguesa: 10 questões; "
            "Raciocínio Lógico: 5 questões; Legislação Municipal: 5 questões; Conhecimentos Específicos: 40 questões. "
            "As questões de conhecimentos básicos têm peso 1 e as de conhecimentos específicos, peso 2.",
            "6.4 Técnico em Informática, total de 50 questões: Língua Portuguesa: 15 questões; Matemática: 10 "
            "questões; Conhecimentos Específicos: 25 questões.",
            "6.5 Assistente Administrativo, total de 50 questões: Língua Portuguesa: 20 questões; Matemática: 10 "
            "questões; Noções de Informática: 10 questões; Legislação Municipal: 10 questões.",
            "6.6 É proibido o uso de celular, relógio digital ou qualquer equipamento eletrônico durante a prova.",
        ]),
        ("7. DOS CRITÉRIOS DE APROVAÇÃO E DESEMPATE", [
            "7.1 Será considerado aprovado o candidato que obtiver, no mínimo, 50% (cinquenta por cento) do total "
            "de pontos da prova objetiva e não zerar nenhuma das disciplinas.",
            "7.2 Em caso de empate, terá preferência, sucessivamente, o candidato: a) com idade igual ou superior a "
            "60 anos; b) com maior nota em Conhecimentos Específicos; c) que tiver exercido a função de jurado.",
        ]),
        ("8. DA PROVA DE TÍTULOS", [
            "8.1 Haverá prova de títulos apenas para o cargo de Analista de Tecnologia da Informação, de caráter "
            "classificatório.",
            "8.2 Pontuação: especialização (pós-graduação lato sensu), 0,5 (meio) ponto; mestrado, 1,0 (um) ponto; "
            "doutorado, 1,5 (um e meio) ponto. A pontuação máxima na prova de títulos é de 2,0 (dois) pontos.",
        ]),
        ("9. DOS RECURSOS", [
            "9.1 O candidato poderá interpor recurso contra o gabarito preliminar no prazo de 2 (dois) dias úteis "
            "contados da sua divulgação, exclusivamente pelo site da organizadora.",
            "9.2 Não serão aceitos recursos enviados por e-mail, correio ou fora do prazo.",
        ]),
        ("10. DO CRONOGRAMA", [
            "Inscrições: 5/1/2027 a 4/2/2027. Pedido de isenção: 5/1/2027 a 12/1/2027. Divulgação dos locais de "
            "prova: 5/3/2027. Prova objetiva: 14/3/2027. Gabarito preliminar: 15/3/2027. Resultado final: 30/4/2027.",
        ]),
        ("ANEXO I - CONTEÚDO PROGRAMÁTICO", [
            "LÍNGUA PORTUGUESA (todos os cargos): interpretação de texto, ortografia, acentuação, concordância "
            "verbal e nominal, regência, crase, pontuação.",
            "RACIOCÍNIO LÓGICO: proposições, conectivos, equivalências lógicas, sequências, problemas de contagem.",
            "LEGISLAÇÃO MUNICIPAL: Lei Orgânica do Município de Vila Serena; Estatuto dos Servidores Municipais.",
            "CONHECIMENTOS ESPECÍFICOS - ANALISTA DE TECNOLOGIA DA INFORMAÇÃO: engenharia de software (requisitos, "
            "metodologias ágeis, Scrum, testes); banco de dados (modelo relacional, SQL, normalização, índices, "
            "transações); redes de computadores (TCP/IP, DNS, HTTP); segurança da informação e LGPD; desenvolvimento "
            "web (APIs REST, HTML, JavaScript); computação em nuvem; programação em Python e Java.",
            "CONHECIMENTOS ESPECÍFICOS - TÉCNICO EM INFORMÁTICA: montagem e manutenção de computadores, sistemas "
            "operacionais Windows e Linux, redes locais, impressoras, pacote de escritório, atendimento ao usuário.",
            "NOÇÕES DE INFORMÁTICA (Assistente Administrativo): editor de texto, planilhas, correio eletrônico, "
            "navegadores, cuidados com senhas e golpes digitais.",
        ]),
        ("ANEXO II - ATRIBUIÇÕES DOS CARGOS", [
            "ANALISTA DE TECNOLOGIA DA INFORMAÇÃO: planejar, desenvolver e manter sistemas da Prefeitura; administrar "
            "bancos de dados; definir padrões de segurança; apoiar projetos de transformação digital.",
            "TÉCNICO EM INFORMÁTICA: prestar suporte técnico aos usuários; instalar e configurar equipamentos e "
            "programas; manter o inventário de hardware; apoiar a manutenção da rede local.",
            "ASSISTENTE ADMINISTRATIVO: executar atividades de apoio administrativo, protocolo, arquivo e atendimento "
            "ao público.",
        ]),
    ],
}

VALE_AZUL = {
    "titulo": "COMPANHIA DE SANEAMENTO DO VALE AZUL\nPROCESSO SELETIVO PÚBLICO Nº 02/2026\nEDITAL DE ABERTURA",
    "secoes": [
        ("1. DAS DISPOSIÇÕES PRELIMINARES", [
            "1.1 O processo seletivo será executado pela Fundação Exemplo de Concursos (FEC) e destina-se ao "
            "preenchimento de empregos públicos da Companhia de Saneamento do Vale Azul.",
            "1.2 Os aprovados serão contratados pelo regime da Consolidação das Leis do Trabalho (CLT), com período "
            "de experiência de 90 (noventa) dias.",
            *BOILER,
        ]),
        ("2. DOS EMPREGOS", [
            "2.1 Analista de Sistemas Júnior: 3 (três) vagas. Requisito: curso superior completo em qualquer área de "
            "Tecnologia da Informação. Salário: R$ 7.400,00. Jornada: 40 (quarenta) horas semanais, em regime "
            "híbrido.",
            "2.2 Técnico de Operação: 10 (dez) vagas. Requisito: ensino médio técnico em Química ou Saneamento. "
            "Salário: R$ 4.100,00. Jornada em escala de revezamento.",
            "2.3 Benefícios: vale-refeição de R$ 1.100,00, plano de saúde e participação nos lucros.",
        ]),
        ("3. DAS INSCRIÇÕES", [
            "3.1 Inscrições de 10 de dezembro de 2026 a 15 de janeiro de 2027, pela internet.",
            "3.2 Taxa de inscrição: R$ 98,00 para nível superior e R$ 72,00 para nível médio técnico.",
            "3.3 Haverá isenção para candidatos inscritos no CadÚnico, mediante pedido até 17 de dezembro de 2026.",
        ]),
        ("4. DAS PROVAS", [
            "4.1 A prova objetiva será realizada em 28 de fevereiro de 2027, no período da tarde, das 14h às 18h.",
            "4.2 Analista de Sistemas Júnior, 70 questões: Língua Portuguesa: 15 questões; Língua Inglesa: 10 "
            "questões; Conhecimentos Específicos: 45 questões.",
            "4.3 Técnico de Operação, 50 questões: Língua Portuguesa: 15 questões; Matemática: 10 questões; "
            "Conhecimentos Específicos: 25 questões.",
            "4.4 Será eliminado o candidato que obtiver menos de 60% (sessenta por cento) de acertos na prova.",
        ]),
        ("5. DO CRONOGRAMA", [
            "Inscrições: 10/12/2026 a 15/1/2027. Prova objetiva: 28/2/2027. Resultado final: 10/4/2027.",
        ]),
        ("ANEXO I - CONTEÚDO PROGRAMÁTICO", [
            "CONHECIMENTOS ESPECÍFICOS - ANALISTA DE SISTEMAS JÚNIOR: lógica de programação, estruturas de dados, "
            "banco de dados relacional e SQL, desenvolvimento web, testes automatizados, controle de versão com Git, "
            "noções de DevOps e containers, inglês técnico.",
        ]),
    ],
}


class EditalPDF(FPDF):
    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", size=8)
        self.cell(0, 10, f"Página {self.page_no()} - documento fictício gerado para testes", align="C")


def gerar(edital: dict, destino: Path) -> None:
    pdf = EditalPDF(format="A4")
    pdf.set_auto_page_break(True, margin=20)
    pdf.set_margins(20, 20, 20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 13)
    pdf.multi_cell(0, 7, edital["titulo"], align="C")
    pdf.ln(6)
    for titulo, paragrafos in edital["secoes"]:
        if titulo.startswith("ANEXO"):
            pdf.add_page()
        pdf.set_font("Helvetica", "B", 11)
        pdf.multi_cell(0, 6, titulo)
        pdf.ln(1)
        pdf.set_font("Helvetica", size=10)
        for p in paragrafos:
            pdf.multi_cell(0, 5.5, p, align="J")
            pdf.ln(2.5)
        pdf.ln(3)
    destino.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(destino))


MANIFESTO = [
    {
        "id": "vila-serena-01-2026",
        "arquivo": "exemplo-vila-serena.pdf",
        "titulo": "Prefeitura de Vila Serena 01/2026 (fictício)",
        "fonte_url": None,
    },
    {
        "id": "vale-azul-02-2026",
        "arquivo": "exemplo-vale-azul.pdf",
        "titulo": "Saneamento Vale Azul 02/2026 (fictício)",
        "fonte_url": None,
    },
]


def main(out: Path = OUT) -> list[Path]:
    caminhos = [out / "exemplo-vila-serena.pdf", out / "exemplo-vale-azul.pdf"]
    gerar(VILA_SERENA, caminhos[0])
    gerar(VALE_AZUL, caminhos[1])
    manifesto = out / "editais.json"
    if not manifesto.exists():
        manifesto.write_text(json.dumps(MANIFESTO, ensure_ascii=False, indent=2), encoding="utf-8")
    return caminhos


if __name__ == "__main__":
    for c in main(Path(sys.argv[1]) if len(sys.argv) > 1 else OUT):
        print("gerado:", c)

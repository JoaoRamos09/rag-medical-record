from app.ai.summary_medical_record.models.MedicalRecordSummaryState import MedicalRecordSummaryState
from app.ai.utils.openai_utils import invoking_model_with_few_shot_prompt
from langsmith import traceable
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate


@traceable(run_type="llm", metadata={"ls_provider": "openai", "ls_model": "gpt-4o-mini", "module": "summary_medical_record"})
async def create_summary_node(state: MedicalRecordSummaryState) -> MedicalRecordSummaryState:
    summary = await invoking_model_with_few_shot_prompt(
        user_input= state["medical_record_markdown"], 
        prompt_system= get_system_prompt(state["summary_max_words"]), 
        few_shot_prompt= create_few_shot_prompt())
    
    state["summary"] = summary.content
    return state

def get_input(medical_record_data: str) -> str:
    return f"""
    O texto da consulta médica: 
    {medical_record_data}
    """

def get_system_prompt(summary_max_words: int) -> str:
    return f"""Você é um experiente médico com anos de experiência de análise de prontuários médicos. Resuma esse prontuário em até {str(summary_max_words)} palavras, sem perder nenhuma informação importante, de forma a ser entendido por um médico que não conhece o paciente e precisa dar continuidade ao tratamento. Ignore informações que não são relevantes para o diagnóstico, como nome. Retorne em texto normal, sem markdown."""

def examples_few_shot_prompt():
    return [
        {
            "input": """LUIZ CARLOS OLIVEIRA, 71 ANOS
                # HAS
                EM USO DE BENICARD ANLO 40/5 (1 CP DE MANHÃ), ATENOLOL 50MG DE MANHÃ, SINVASTATINA 40MG NOITE
                HDA: Paciente com alteração de hábito intestinal há aproximadamente 1 mês, com fezes esbranquiçadas, 
                evoluindo com diarreia líquida há 9 dias, de coloração esverdeada, alguma dor abdominal e náuseas. 
                Associadamente, exames prévios mostraram aumento de bilirrubinas. Realizados novos exames neste atendimento, 
                com laudo preliminar evidenciando aumento de densidade de conteúdo de vesícula biliar, aumento de bilirrubina,
                aumento de PCR, aumento de creatinina e potássio. Subjetivo: Avalio paciente em leito de enfermaria, 
                acompanhado por familiar. Refere ainda um episódio de diarreia durante a noite.
                Objetivo:
                BEG, lúcido e orientado, MUC
                Eupneico em AA, anictérico, Tax 36,6 em 28/08
                Sinais vitais estáveis
                Abdome normotenso, depressível, indolor à palpação, sem sinais de peritonismo, Murphy e Blumberg negativos
                Extremidades aquecidas e perfundidas, sem edema
                TC abdome 20/08: Ausência de pneumoperitônio ou líquido livre na cavidade abdominal. Vesícula biliar moderadamente 
                repleta, com conteúdo tenuemente mais denso que o usual, inespecífico. Não há sinais inflamatórios em tecidos adiposos
                pericolecistócitos. Não há dilatação do sistema biliar intra ou extra-hepático. Fígado, baço e pâncreas com dimensões e 
                densidades normais. Adrenais anatômicas. Formação ovalada e hipodensa na cortical do polo superior do rim esquerdo, com 
                calcificações periféricas, medindo 2,7 cm, inespecífica e nódulo em segmento posterior. Microcálculo em cálice do polo 
                superior do rim direito, medindo 0,2 cm. Os rins possuem contornos regulares e dimensões normais, não havendo hidronefrose. 
                Bexiga pouco repleta, sem evidência de conteúdo anômalo. Próstata com dimensões levemente aumentadas e apresentando pequena 
                calcificação inespecífica. Pequena hérnia gástrica de hiato. Proeminentes linfonodos mesentéricos no quadrante superior 
                esquerdo, o maior ovalado, com dimensões levemente aumentadas, medindo 1,1 cm no menor diâmetro e com região central adiposa, 
                sugerindo benignidade. Aorta com trajeto e diâmetro normais. Mínima aterosclerose aórtica. Não há evidência de linfonodomegalias 
                intra ou retroperitoneais.
                USG de abdome: Fígado de forma, contornos e dimensões usuais, apresentando leve aumento difuso da ecogenicidade do seu parênquima, 
                sugerindo depósito, mais provavelmente esteatose grau I. Baço com dimensões usuais e ecoestrutura preservada. Pâncreas sem evidência 
                de alterações nas porções identificadas. Vesícula biliar normodistendida de paredes finas e lisas, sem evidência de litíase. Não há 
                sinais de dilatação das vias biliares intra e extra-hepáticas. Rins de forma, contornos e dimensões usuais, apresentando espessura e 
                textura do parênquima preservada. Não há evidência de dilatação dos sistemas coletores renais e/ou imagens de cálculos. Bexiga vazia. 
                Não há evidência de líquido livre na cavidade peritoneal.TC abdome 28/08: Impressão: Vesícula biliar de boa repleção, com paredes finas,
                contendo material tenuamente mais denso que o usual, inespecífico e ausência de litíase (revisão por Dr. Nilmar). Nefrolitíase não 
                obstrutiva à direita. Cistos renais bilaterais (Bosniak II). Pequena calcificação justapareital anterior da bexiga.
                Aumento do volume da próstata. Ateromatose aórtica. Persiste a proeminência de linfonodos intermesentéricos de raiz do mesentério,
                alguns com dimensões levemente aumentadas, inespecíficos.
                Culturas/Sorologias
                Clostridium 21/08 - CLOSTRIDIUM DIFFICILE - TOXINA A + B
                Resultado: 0,34 - Valores de referência: Não reagente: Inferior a 0,90
                Laboratoriais
                20/08: Hb 14,7 | Leuco 9700 | b0 | Plaq 255000 | Cr 2,08 | Ur 57 | Na 143 | K 5,7 | TGO 40 | TGO 63 | FA 113 | GGT 59 | Amilase 101 | Lipase 23 | Lactato 3,3 | PCR 1,76 | BT 1,56 | BD 0,62
                21/08: FA 113 | Hb 12,1 | Ht 36,6 | RDW 11,2 | Leuco 9700 | Plaq 255000 | Hb 12,7 | K 5,7 | PCR 2,7 | Equ: 63 | FA: 59
                28/08: FA: 38 | Hb 12 | VCM: 88 | RDW: 14 | Plaq: 220400 | Cr: 2,11 | Ur: 19 | Na: 142 | K: 3,9
                30/08: Equ: 3/8 | RDW: 14,2 | BT: 0,79 | BD: 0,33 | AST (TGO): 22 | ALT (TGP): 14 | TGO: 39 | FA: 48
                31/08: Erit: 4,4 | Hb 13 | Ht 37,9 | RDW 15,5 | Leuco 9800 | Plaq 311000 | Cr 0,55 | Ur 11 | Na 139 | K 4,1 | Tropt < 0,5 | BT 2,3""",
        "output": """ 71 anos, hipertenso, apresenta diarreia líquida há 9 dias com alterações de bilirrubinas, PCR, creatinina e potássio. Exames indicam alterações renais e hepáticas leves, vesícula biliar sem litíase e clostridium difficile positivo. Clínico estável, investigando cistos renais, linfonodos mesentéricos proeminentes e esteatose hepática grau I."""
        },
        {
            "input": """Em 05/05/2025, às 15h40, atendimento sob prontuário 345678. Paciente feminina, 65 anos, trazida por familiares por quadro de confusão mental, febre e tosse produtiva há 5 dias. Evoluiu nas últimas 24h com desorientação e sonolência.
            Portadora de Alzheimer em estágio inicial, hipertensão e osteoporose. Faz uso regular de donepezila, losartana e suplementação de cálcio com vitamina D.
            Exame físico: PA 135/85 mmHg, FC 92 bpm, FR 24 rpm, Tax 38.5°C, SpO2 92% em ar ambiente. Ausculta pulmonar com estertores crepitantes em base direita. Glasgow 13.
            Radiografia de tórax evidenciando consolidação em lobo inferior direito. Exames laboratoriais com leucocitose e proteína C reativa elevada.
            Diagnosticada pneumonia comunitária grave. Iniciada antibioticoterapia com ceftriaxona e claritromicina, suporte ventilatório com oxigenioterapia e hidratação venosa. Internada em unidade de terapia intensiva para monitorização.""",
            "output": """Idosa de 65 anos com Alzheimer apresenta quadro de pneumonia comunitária grave, manifestando confusão mental, febre e sintomas respiratórios. Exames confirmam infecção pulmonar. Iniciado tratamento com antibióticos e suporte intensivo. Necessária internação em UTI para monitorização."""
        },
        {
            "input": """No dia 12/06/2025, às 11h30, consulta sob prontuário 567890. Paciente masculino, 40 anos, atleta, queixa-se de dor aguda em joelho direito após trauma durante partida de futebol há 2 horas. Refere ter ouvido um estalo no momento do trauma.
            Nega comorbidades ou cirurgias prévias. Pratica futebol regularmente 3 vezes por semana.
            Ao exame: marcha claudicante, joelho direito com edema +2/+4, derrame articular moderado, teste de gaveta anterior positivo, Lachman positivo. Pulsos distais presentes e simétricos, sem alterações neurovasculares.
            Realizada radiografia que não evidenciou fraturas. Hipótese diagnóstica de lesão do ligamento cruzado anterior.
            Prescrito analgésicos, anti-inflamatórios, crioterapia e repouso relativo com uso de muletas. Solicitada ressonância magnética do joelho e encaminhado para avaliação ortopédica especializada.""",
            "output": """Atleta de 40 anos sofreu trauma em joelho direito durante prática esportiva. Exame físico sugestivo de lesão ligamentar. Radiografia sem fraturas. Suspeita de ruptura do ligamento cruzado anterior. Iniciado tratamento conservador e encaminhado para investigação complementar com especialista."""
        },
        {
            "input": """Histórico familiar significativo para doenças autoimunes. Paciente Maria Silva, 28 anos, prontuário 789012, comparece à consulta em 15/07/2025 relatando fadiga intensa, dores articulares e manchas avermelhadas na pele há 6 meses. Exame físico revela eritema malar característico, artrite em punhos e joelhos, fotossensibilidade.
            Exames laboratoriais mostram: FAN positivo 1:640 padrão homogêneo, Anti-DNA positivo, complemento baixo, hemograma com anemia leve.
            Diagnóstico estabelecido de Lúpus Eritematoso Sistêmico. Iniciado hidroxicloroquina 400mg/dia, prednisona 20mg/dia e protetor solar FPS 70.
            Paciente orientada sobre a doença, necessidade de acompanhamento regular e medidas de fotoproteção. Retorno agendado em 30 dias com exames de controle.""",
            "output": """28 anos, histórico familiar de doenças autoimunes, com fadiga, artrite e eritema malar. Diagnóstico: Lúpus Eritematoso Sistêmico (FAN 1:640, anti-DNA positivo, complemento baixo, anemia). Tratamento: hidroxicloroquina, prednisona e fotoproteção. Retorno em 30 dias."""
        },
        {
            "input": """Avaliação pediátrica de emergência - 20/08/2025
            Dados vitais iniciais: Temperatura 39.8°C, FC 130bpm, FR 32rpm, SpO2 94%
            Criança Pedro Santos, 3 anos, trazido pela mãe com história de convulsão febril há 1 hora, duração aproximada de 3 minutos. Apresenta quadro de infecção de vias aéreas superiores há 2 dias, com coriza, tosse e inapetência.
            Desenvolvimento neuropsicomotor adequado para idade, calendário vacinal em dia. Sem antecedentes patológicos relevantes.
            Conduta: Administrado antitérmico, realizada coleta de exames (hemograma, PCR, eletrólitos) e iniciado monitorização. Após 6 horas de observação, paciente afebril, sem novos episódios convulsivos. Alta com orientações e acompanhamento ambulatorial.""",
            "output": """Criança de 3 anos, apresentou convulsão febril de 3 minutos, associada a infecção respiratória superior. Dados vitais iniciais alterados (T 39.8°C). Tratamento com antitérmico e observação por 6 horas, sem recorrências. Alta com orientações e acompanhamento."""
        }
    ]

def create_few_shot_prompt():
    examples = examples_few_shot_prompt()
    example_prompt = ChatPromptTemplate.from_messages([("human", "{input}"), ("ai", "{output}")])
    
    return FewShotChatMessagePromptTemplate(
        examples = examples,    
        example_prompt = example_prompt,
    )
    

// Supported languages and their display labels
const SUPPORTED_LANGS = [
  { code: 'en', label: 'EN', full: 'English' },
  { code: 'pt', label: 'PT', full: 'Português' },
  { code: 'es', label: 'ES', full: 'Español' },
];

const TRANSLATIONS = {
  // ─────────────────────────────────────────────────────────────────────────
  // ENGLISH (default)
  // ─────────────────────────────────────────────────────────────────────────
  en: {
    nav_title:        'CADTN Classifier',
    nav_about:        'About',

    panel_title:      'Analysis Panel',
    model_label:      'AI Model',
    image_label:      'Clinical / Microscopy Image',
    upload_prompt:    'Click to select an image',
    upload_hint:      'PNG, JPG, JPEG up to 10 MB',
    btn_analyze:      'Analyze Image',
    btn_processing:   'Processing on server…',

    results_title:    'Top-5 Results',
    results_empty:    'Upload an image to see the model predictions here.',
    btn_details:      'View clinical details',

    modal_desc:       'Description',
    modal_trans:      'Transmission',
    modal_symptoms:   'Clinical Symptoms',
    modal_treatment:  'Treatment',
    modal_close:      'Close',

    about_title:      'CADTN Project',
    about_subtitle:   'Computer-Assisted Classifier for Neglected Tropical Diseases.',
    about_researcher: 'Researcher',
    about_advisor:    'Advisor',
    about_institution:'Institution',
    about_lab:        'Laboratory',

    footer_lab:       'Geomatics Laboratory · UFMS © 2026',
    footer_project:   'Research Project · Mário de Araújo Carvalho',

    diseases: {
      microscopy_parasite_babesia: {
        name:        'Babesia spp. (Babesiosis)',
        description: 'Intraerythrocytic protozoa that cause babesiosis, a disease clinically similar to malaria. They infect and destroy the host\'s red blood cells.',
        transmission:'Transmitted by the bite of infected ticks (mainly genus Ixodes). Can also occur through blood transfusion.',
        symptoms:    'Fever, chills, profuse sweating, fatigue, and hemolytic anemia. Can be severe in elderly, immunocompromised, or asplenic patients.',
        treatment:   'Combination of antimicrobials: atovaquone + azithromycin, or clindamycin + quinine for severe cases.',
      },
      microscopy_parasite_plasmodium: {
        name:        'Plasmodium spp. (Malaria)',
        description: 'Obligate intracellular pathogens responsible for malaria. They infect and destroy red blood cells during their reproductive cycle.',
        transmission:'Transmitted by the bite of an infected female Anopheles mosquito.',
        symptoms:    'Cyclic high fever, chills, sweating, headache, and myalgia. Severe cases may cause cerebral malaria and severe anemia.',
        treatment:   'Artemisinin-based combination therapies (ACTs), chloroquine or other antimalarials depending on species and local resistance.',
      },
      microscopy_parasite_trichomonad: {
        name:        'Trichomonas vaginalis (Trichomoniasis)',
        description: 'A flagellate protozoan causing trichomoniasis, one of the most common non-viral STIs worldwide. It affects the human urogenital tract.',
        transmission:'Transmitted through unprotected sexual contact with an infected person.',
        symptoms:    'Often asymptomatic (especially in men). In women: profuse discharge, itching, genital irritation, and painful urination.',
        treatment:   'Single or multi-dose oral antiparasitic drugs: metronidazole or tinidazole.',
      },
      microscopy_parasite_leishmania: {
        name:        'Leishmania spp. (Leishmaniasis)',
        description: 'Intracellular protozoa causing visceral, cutaneous, and mucocutaneous leishmaniasis — a major Neglected Tropical Disease (NTD).',
        transmission:'Transmitted by the bite of infected phlebotomine sandflies.',
        symptoms:    'Cutaneous form: hard-to-heal skin ulcers. Visceral form (kala-azar): prolonged fever, splenomegaly, hepatomegaly, and bone marrow involvement.',
        treatment:   'Pentavalent antimonials, liposomal amphotericin B, or miltefosine depending on clinical presentation.',
      },
      microscopy_parasite_trypanosome: {
        name:        'Trypanosoma spp.',
        description: 'Flagellate protozoa causing severe NTDs: Chagas disease (T. cruzi, Americas) and Sleeping Sickness (T. brucei, Africa).',
        transmission:'T. cruzi: via feces of triatomine bugs (kissing bugs). T. brucei: via tsetse fly bite.',
        symptoms:    'Acute Chagas: fever, fatigue, Romaña\'s sign. Chronic phase: cardiac and digestive complications (megacolon/megaesophagus).',
        treatment:   'Benznidazole or Nifurtimox for Chagas disease — most effective when administered in the early acute phase.',
      },
      microscopy_parasite_toxoplasma: {
        name:        'Toxoplasma gondii (Toxoplasmosis)',
        description: 'An obligate intracellular protozoan that uses felines as definitive hosts but can infect virtually all warm-blooded animals, including humans.',
        transmission:'Contaminated water/food (cat oocysts), undercooked meat, or congenital (mother to fetus).',
        symptoms:    'Asymptomatic in healthy individuals. Dangerous for immunocompromised patients and pregnant women (risk of fetal neurological malformations).',
        treatment:   'Healthy adults: usually no treatment. Severe cases or pregnancy: pyrimethamine + sulfadiazine + folinic acid.',
      },
      microscopy_parasite_rbcs: {
        name:        'Healthy Red Blood Cells (Negative Control)',
        description: 'Control class of normal erythrocytes responsible for oxygen transport throughout the body.',
        transmission:'N/A — healthy human cell.',
        symptoms:    'Presence of this class indicates a clean blood sample with no visible intraerythrocytic parasites.',
        treatment:   'N/A. Diagnosis: Healthy Blood.',
      },
      microscopy_parasite_leukocyte: {
        name:        'Healthy Leukocytes (Negative Control)',
        description: 'Control class of normal white blood cells — the immune system\'s defense cells essential for fighting infections.',
        transmission:'N/A — healthy human cell.',
        symptoms:    'Correct detection means the AI has learned to distinguish the complex nucleus of a human immune cell from an invading parasite.',
        treatment:   'N/A. Diagnosis: Normal Immune Response / Healthy Blood.',
      },
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // PORTUGUESE
  // ─────────────────────────────────────────────────────────────────────────
  pt: {
    nav_title:        'Classificador CADTN',
    nav_about:        'Sobre',

    panel_title:      'Painel de Análise',
    model_label:      'Modelo de Inteligência Artificial',
    image_label:      'Imagem Clínica / Microscópica',
    upload_prompt:    'Clique para selecionar uma imagem',
    upload_hint:      'PNG, JPG, JPEG até 10 MB',
    btn_analyze:      'Analisar Imagem',
    btn_processing:   'Processando no servidor…',

    results_title:    'Resultados Top-5',
    results_empty:    'Envie uma imagem para ver as previsões do modelo aqui.',
    btn_details:      'Ver detalhes clínicos',

    modal_desc:       'Descrição',
    modal_trans:      'Transmissão',
    modal_symptoms:   'Sintomas Clínicos',
    modal_treatment:  'Tratamento',
    modal_close:      'Fechar',

    about_title:      'Projeto CADTN',
    about_subtitle:   'Classificador Assistido por Computador para Doenças Tropicais Negligenciadas.',
    about_researcher: 'Pesquisador',
    about_advisor:    'Orientador',
    about_institution:'Instituição',
    about_lab:        'Laboratório',

    footer_lab:       'Laboratório de Geomática · UFMS © 2026',
    footer_project:   'Projeto de Pesquisa · Mário de Araújo Carvalho',

    diseases: {
      microscopy_parasite_babesia: {
        name:        'Babesia spp. (Babesiose)',
        description: 'Protozoários intraeritrocitários causadores da babesiose, doença clinicamente semelhante à malária. Infectam e destroem as hemácias do hospedeiro.',
        transmission:'Transmitido pela picada de carrapatos infectados (gênero Ixodes). Também pode ocorrer por transfusão de sangue.',
        symptoms:    'Febre, calafrios, suores intensos, fadiga e anemia hemolítica. Pode ser grave em idosos, imunossuprimidos ou pacientes sem baço.',
        treatment:   'Combinação de antimicrobianos: atovaquona + azitromicina, ou clindamicina + quinino em casos graves.',
      },
      microscopy_parasite_plasmodium: {
        name:        'Plasmodium spp. (Malária)',
        description: 'Patógenos intracelulares obrigatórios responsáveis pela malária. Infectam e destroem os glóbulos vermelhos durante seu ciclo reprodutivo.',
        transmission:'Transmitido pela picada da fêmea infectada do mosquito Anopheles.',
        symptoms:    'Febre alta cíclica, calafrios, sudorese, cefaleia e mialgia. Casos graves: anemia severa e malária cerebral.',
        treatment:   'Terapias combinadas à base de artemisinina (ACTs), cloroquina ou outros antimaláricos conforme espécie e resistência local.',
      },
      microscopy_parasite_trichomonad: {
        name:        'Trichomonas vaginalis (Tricomoníase)',
        description: 'Protozoário flagelado que causa a tricomoníase, uma das ISTs não virais mais comuns no mundo. Afeta o trato urogenital humano.',
        transmission:'Transmitido por contato sexual desprotegido com pessoa infectada.',
        symptoms:    'Frequentemente assintomática (especialmente em homens). Nas mulheres: corrimento abundante, coceira, irritação e disúria.',
        treatment:   'Antiparasitários orais de dose única ou múltipla: metronidazol ou tinidazol.',
      },
      microscopy_parasite_leishmania: {
        name:        'Leishmania spp. (Leishmaniose)',
        description: 'Protozoários intracelulares causadores da leishmaniose visceral, cutânea e mucocutânea — uma importante DTN.',
        transmission:'Transmitido pela picada de mosquitos flebotomíneos infectados (birigui/mosquito-palha).',
        symptoms:    'Forma cutânea: úlceras de difícil cicatrização. Forma visceral (calazar): febre prolongada, esplenomegalia e hepatomegalia.',
        treatment:   'Antimoniais pentavalentes, anfotericina B lipossomal ou miltefosina conforme apresentação clínica.',
      },
      microscopy_parasite_trypanosome: {
        name:        'Trypanosoma spp.',
        description: 'Protozoários flagelados causadores de DTNs graves: Doença de Chagas (T. cruzi, Américas) e Doença do Sono (T. brucei, África).',
        transmission:'T. cruzi: fezes do triatomíneo (barbeiro). T. brucei: picada da mosca tsé-tsé.',
        symptoms:    'Chagas aguda: febre, fadiga, Sinal de Romaña. Fase crônica: complicações cardíacas e digestivas (megacólon/megaesôfago).',
        treatment:   'Benzonidazol ou Nifurtimox para Chagas — altamente eficazes apenas na fase aguda recente.',
      },
      microscopy_parasite_toxoplasma: {
        name:        'Toxoplasma gondii (Toxoplasmose)',
        description: 'Protozoário intracelular obrigatório com felinos como hospedeiros definitivos. Infecta quase todos os animais de sangue quente, incluindo humanos.',
        transmission:'Água/alimentos contaminados com oocistos, carne mal cozida ou transmissão congênita.',
        symptoms:    'Assintomática em pessoas saudáveis. Perigosa para imunodeprimidos e gestantes (risco de malformações neurológicas fetais).',
        treatment:   'Adultos saudáveis geralmente não necessitam tratamento. Casos graves: pirimetamina + sulfadiazina + ácido folínico.',
      },
      microscopy_parasite_rbcs: {
        name:        'Hemácias Saudáveis (Controle Negativo)',
        description: 'Classe de controle com eritrócitos normais responsáveis pelo transporte de oxigênio.',
        transmission:'N/A — célula humana saudável.',
        symptoms:    'Presença desta classe indica amostra de sangue limpa, sem parasitas intraeritrocitários visíveis.',
        treatment:   'N/A. Diagnóstico: Sangue Saudável.',
      },
      microscopy_parasite_leukocyte: {
        name:        'Leucócitos Saudáveis (Controle Negativo)',
        description: 'Classe de controle com glóbulos brancos normais — células de defesa do sistema imunológico essenciais no combate a infecções.',
        transmission:'N/A — célula humana saudável.',
        symptoms:    'Detecção correta indica que a IA aprendeu a diferenciar o núcleo de uma célula imune do núcleo de um parasita.',
        treatment:   'N/A. Diagnóstico: Resposta Imunológica Normal / Sangue Saudável.',
      },
    },
  },

  // ─────────────────────────────────────────────────────────────────────────
  // SPANISH
  // ─────────────────────────────────────────────────────────────────────────
  es: {
    nav_title:        'Clasificador CADTN',
    nav_about:        'Acerca de',

    panel_title:      'Panel de Análisis',
    model_label:      'Modelo de Inteligencia Artificial',
    image_label:      'Imagen Clínica / Microscópica',
    upload_prompt:    'Haga clic para seleccionar una imagen',
    upload_hint:      'PNG, JPG, JPEG hasta 10 MB',
    btn_analyze:      'Analizar Imagen',
    btn_processing:   'Procesando en el servidor…',

    results_title:    'Resultados Top-5',
    results_empty:    'Suba una imagen para ver las predicciones del modelo aquí.',
    btn_details:      'Ver detalles clínicos',

    modal_desc:       'Descripción',
    modal_trans:      'Transmisión',
    modal_symptoms:   'Síntomas Clínicos',
    modal_treatment:  'Tratamiento',
    modal_close:      'Cerrar',

    about_title:      'Proyecto CADTN',
    about_subtitle:   'Clasificador Asistido por Computadora para Enfermedades Tropicales Desatendidas.',
    about_researcher: 'Investigador',
    about_advisor:    'Director',
    about_institution:'Institución',
    about_lab:        'Laboratorio',

    footer_lab:       'Laboratorio de Geomática · UFMS © 2026',
    footer_project:   'Proyecto de Investigación · Mário de Araújo Carvalho',

    diseases: {
      microscopy_parasite_babesia: {
        name:        'Babesia spp. (Babesiosis)',
        description: 'Protozoos intraeritrocíticos causantes de la babesiosis, enfermedad clínicamente similar a la malaria. Infectan y destruyen los glóbulos rojos del huésped.',
        transmission:'Transmitida por la picadura de garrapatas infectadas (género Ixodes). También puede ocurrir por transfusión de sangre.',
        symptoms:    'Fiebre, escalofríos, sudoración intensa, fatiga y anemia hemolítica. Puede ser grave en ancianos, inmunodeprimidos o pacientes sin bazo.',
        treatment:   'Combinación de antimicrobianos: atovaquona + azitromicina, o clindamicina + quinina en casos graves.',
      },
      microscopy_parasite_plasmodium: {
        name:        'Plasmodium spp. (Malaria)',
        description: 'Patógenos intracelulares obligados responsables de la malaria. Infectan y destruyen los glóbulos rojos durante su ciclo reproductivo.',
        transmission:'Transmitida por la picadura de la hembra infectada del mosquito Anopheles.',
        symptoms:    'Fiebre alta cíclica, escalofríos, sudoración, cefalea y mialgia. Casos graves: anemia severa y malaria cerebral.',
        treatment:   'Terapias combinadas basadas en artemisinina (ACTs), cloroquina u otros antimaláricos según la especie y resistencia local.',
      },
      microscopy_parasite_trichomonad: {
        name:        'Trichomonas vaginalis (Tricomoniasis)',
        description: 'Protozoo flagelado que causa la tricomoniasis, una de las ITS no virales más comunes en el mundo. Afecta el tracto urogenital humano.',
        transmission:'Transmitida por contacto sexual desprotegido con una persona infectada.',
        symptoms:    'A menudo asintomática (especialmente en hombres). En mujeres: flujo abundante, picazón, irritación genital y disuria.',
        treatment:   'Antiparasitarios orales de dosis única o múltiple: metronidazol o tinidazol.',
      },
      microscopy_parasite_leishmania: {
        name:        'Leishmania spp. (Leishmaniasis)',
        description: 'Protozoos intracelulares causantes de leishmaniasis visceral, cutánea y mucocutánea — una importante Enfermedad Tropical Desatendida (ETD).',
        transmission:'Transmitida por la picadura de flebótomos infectados.',
        symptoms:    'Forma cutánea: úlceras de difícil cicatrización. Forma visceral (kala-azar): fiebre prolongada, esplenomegalia y hepatomegalia.',
        treatment:   'Antimoniales pentavalentes, anfotericina B liposomal o miltefosina según presentación clínica.',
      },
      microscopy_parasite_trypanosome: {
        name:        'Trypanosoma spp.',
        description: 'Protozoos flagelados causantes de ETDs graves: Enfermedad de Chagas (T. cruzi, Américas) y Enfermedad del Sueño (T. brucei, África).',
        transmission:'T. cruzi: heces del triatomino (chinche besucona). T. brucei: picadura de la mosca tse-tsé.',
        symptoms:    'Chagas agudo: fiebre, fatiga, signo de Romaña. Fase crónica: complicaciones cardíacas y digestivas (megacolon/megaesófago).',
        treatment:   'Benznidazol o Nifurtimox para Chagas — altamente eficaces solo en la fase aguda temprana.',
      },
      microscopy_parasite_toxoplasma: {
        name:        'Toxoplasma gondii (Toxoplasmosis)',
        description: 'Protozoo intracelular obligado con felinos como huéspedes definitivos. Puede infectar a casi todos los animales de sangre caliente, incluidos los humanos.',
        transmission:'Agua/alimentos contaminados con ooquistes, carne poco cocida, o transmisión congénita.',
        symptoms:    'Asintomática en personas sanas. Peligrosa para inmunodeprimidos y embarazadas (riesgo de malformaciones neurológicas fetales).',
        treatment:   'Adultos sanos: generalmente sin tratamiento. Casos graves: pirimetamina + sulfadiazina + ácido folínico.',
      },
      microscopy_parasite_rbcs: {
        name:        'Glóbulos Rojos Sanos (Control Negativo)',
        description: 'Clase de control de eritrocitos normales responsables del transporte de oxígeno en el organismo.',
        transmission:'N/A — célula humana sana.',
        symptoms:    'La presencia de esta clase indica una muestra de sangre limpia sin parásitos intraeritrocíticos visibles.',
        treatment:   'N/A. Diagnóstico: Sangre Sana.',
      },
      microscopy_parasite_leukocyte: {
        name:        'Leucocitos Sanos (Control Negativo)',
        description: 'Clase de control de glóbulos blancos normales — células de defensa del sistema inmunológico esenciales para combatir infecciones.',
        transmission:'N/A — célula humana sana.',
        symptoms:    'La detección correcta indica que la IA aprendió a diferenciar el núcleo de una célula inmune del de un parásito invasor.',
        treatment:   'N/A. Diagnóstico: Respuesta Inmune Normal / Sangre Sana.',
      },
    },
  },
};

// Returns a flat translation key for the current language, falling back to English
function t(lang, key) {
  const dict = TRANSLATIONS[lang] || TRANSLATIONS['en'];
  return dict[key] ?? TRANSLATIONS['en'][key] ?? key;
}

// Returns disease info object for a given class ID and language
function getDiseaseTranslation(lang, classId) {
  const dict = TRANSLATIONS[lang] || TRANSLATIONS['en'];
  return dict.diseases?.[classId] ?? TRANSLATIONS['en'].diseases?.[classId] ?? null;
}

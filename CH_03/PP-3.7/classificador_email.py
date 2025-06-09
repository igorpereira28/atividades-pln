import re
import string
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Pré-processamento simples
def preprocessar(texto):
    texto = texto.lower()
    texto = re.sub(rf"[{string.punctuation}]", "", texto)
    return texto.strip()

# Base de dados expandida (mínimo 15 por classe)
dados = [
    # INCIDENTE
    ("Meu computador não liga", "Incidente"),
    ("Erro ao abrir o sistema", "Incidente"),
    ("A internet está lenta", "Incidente"),
    ("O sistema travou no login", "Incidente"),
    ("Não consigo acessar meus arquivos", "Incidente"),
    ("Problema na rede", "Incidente"),
    ("Sistema caiu novamente", "Incidente"),
    ("Falha ao iniciar o aplicativo", "Incidente"),
    ("O teclado não responde", "Incidente"),
    ("Tela azul apareceu", "Incidente"),
    ("O mouse parou de funcionar", "Incidente"),
    ("Mensagem de erro ao salvar", "Incidente"),
    ("Sistema reiniciando sozinho", "Incidente"),
    ("Aplicativo fecha sozinho", "Incidente"),
    ("Erro de permissão negada", "Incidente"),

    # DÚVIDA
    ("Como alterar minha senha?", "Dúvida"),
    ("Qual o horário de atendimento?", "Dúvida"),
    ("Como instalo o programa?", "Dúvida"),
    ("Quais extensões são permitidas?", "Dúvida"),
    ("Onde encontro os relatórios?", "Dúvida"),
    ("Como acesso de casa?", "Dúvida"),
    ("Para que serve essa função?", "Dúvida"),
    ("Como faço backup dos arquivos?", "Dúvida"),
    ("Preciso de ajuda para preencher o formulário", "Dúvida"),
    ("Como configuro a impressora?", "Dúvida"),
    ("Onde altero meus dados?", "Dúvida"),
    ("É possível recuperar a conta?", "Dúvida"),
    ("Como envio um chamado?", "Dúvida"),
    ("Qual é o procedimento para troca de senha?", "Dúvida"),
    ("Tenho que reiniciar depois da instalação?", "Dúvida"),

    # ELOGIO
    ("Parabéns pelo atendimento", "Elogio"),
    ("A equipe foi muito atenciosa", "Elogio"),
    ("Gostei do suporte técnico", "Elogio"),
    ("Ótimo serviço prestado", "Elogio"),
    ("Funcionário muito educado", "Elogio"),
    ("Agradeço pela ajuda", "Elogio"),
    ("Excelente atendimento hoje", "Elogio"),
    ("Fui bem atendido", "Elogio"),
    ("Muito obrigado pela paciência", "Elogio"),
    ("Atendimento rápido e eficiente", "Elogio"),
    ("Vocês são demais!", "Elogio"),
    ("Sempre me ajudam com agilidade", "Elogio"),
    ("Agradeço de coração", "Elogio"),
    ("Impressionado com o suporte", "Elogio"),
    ("Top demais! Valeu!", "Elogio"),
    ("O sistema é muito bom", "Elogio"),
]

# Separar textos e rótulos
texts = [preprocessar(t[0]) for t in dados]
labels = [t[1] for t in dados]

# Vetorização Bag of Words
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts)

# Divisão treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Treinamento MLP
mlp = MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=42)
mlp.fit(X_train, y_train)

# Avaliação
y_pred = mlp.predict(X_test)
print("=== RELATÓRIO DE CLASSIFICAÇÃO ===")
print(classification_report(y_test, y_pred))

# Função para classificar novos textos
def classificar_texto(texto):
    texto = preprocessar(texto)
    vetor = vectorizer.transform([texto])
    return mlp.predict(vetor)[0]

# Teste do sistema
texto_teste = "Não consigo abrir o sistema desde ontem"
print("\n=== TESTE COM NOVO TEXTO ===")
print(f"Texto: {texto_teste}")
print(f"Classificação: {classificar_texto(texto_teste)}")
vetor = vectorizer.transform([texto_teste])
print("Índices ativados:", vetor.nonzero()[1])

# Salvar modelo e vetor
joblib.dump(mlp, 'modelo_mlp.pkl')
joblib.dump(vectorizer, 'vetor_bow.pkl')

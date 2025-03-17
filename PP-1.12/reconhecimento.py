import speech_recognition as sr;

#Função para ouvir e reconhecer a fala
def ouvir_microfone():
    #Habilitar o microfone do usuário
    microfone = sr.Recognizer()

    #Usando o microfone como fonte de entrada e with para garantir que o recurso será fechado após o uso
    with sr.Microphone() as source:

        #Chama um algoritmo de redução de ruídos de som
        microfone.adjust_for_ambient_noise(source)

        #Frase introdutório para dizer algo
        print("Diga algo:")

        #captura o áudio do microfone e armazena
        audio = microfone.listen(source)

    try:
        #Passa a variavel para o algoritmo reconhecer padrões
        frase = microfone.recognize_google(audio, language='pt-BR')

        #Retorna a frase pronunciada
        print("Você disse... " + frase)

    #Se não reconhecer o padrão de fala, exiba a seguinte mensagem
    except sr.UnknownValueError:
        print("Eu não entendi o que você falou...")

    return frase

ouvir_microfone()
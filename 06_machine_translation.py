# Machine Translation 📠
# Codédex

from translate import Translator

# Create a Translator object
translator = Translator(to_lang="es")  # Spanish

# Text to be translated
text = "Hello, how are you?"

# Perform the translation
translation = translator.translate(text)

# Print the translated text
print("Translated Text:", translation)




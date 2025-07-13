# PoemCraft
Trained a GPT-2-medium model on a custom dataset consisting of 6,500 poems and haikus to generate poetry in styles including Shakespeare, Dickinson, Poe, and haiku (5-7-5) given user input. 
Fine-tuned models in PyTorch with custom prompt tokens for style control and syllable-aware generation for haikus.
Evaluated outputs via cosine similarity to real poems and syllable count validation for haikus using nltk.corpus.cmudict.


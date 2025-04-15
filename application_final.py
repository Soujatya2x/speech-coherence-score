import speech_recognition as sr
import streamlit as st
from nltk.tokenize import word_tokenize
import time
from sentence_transformers import SentenceTransformer, util

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-V2')

model=load_model()

def coherence_score(sentences):
  embeddings=model.encode(sentences,convert_to_tensor=True)
  coherence_score=0

  for i in range(len(embeddings)-1):
    coherence_score+=util.cos_sim(embeddings[i],embeddings[i+1])

  average_coherence=coherence_score/(len(sentences)-1)
  st.write('average coherence',float(average_coherence))


st.title('🎙️ Interactive Voice Application for measuring coherence score')

# Initialize session state
if "transcript" not in st.session_state:
    st.session_state.transcript = ""

if "listening" not in st.session_state:
    st.session_state.listening = False

speech = sr.Recognizer()


# Sentence segmentation logic
def segment_sentences(text, max_sentence_length=12):
    words = word_tokenize(text.lower())
    sentences = []
    current = []

    for i, word in enumerate(words):
        current.append(word)

        next_word = words[i + 1] if i + 1 < len(words) else None
        sentence_starters = {'i', 'you', 'he', 'she', 'they', 'we', 'it', 'so', 'then', 'but'}

        if next_word in sentence_starters and len(current) >= 3:
            sentences.append(' '.join(current))
            current = []

        elif len(current) >= max_sentence_length:
            sentences.append(' '.join(current))
            current = []

    if current:
        sentences.append(' '.join(current))

    return sentences

# Start/Stop buttons
if st.button("🎤 Start Talking"):
    st.session_state.listening = True

if st.button("🛑 Stop Talking"):
    st.session_state.listening = False

# Listening logic
if st.session_state.listening:
    with sr.Microphone() as source:
        st.info("Listening...")
        speech.adjust_for_ambient_noise(source)
        try:
            audio = speech.listen(source, timeout=5, phrase_time_limit=5)
            text = speech.recognize_google(audio)
            st.session_state.transcript += " " + text
            st.success(f"You said: {text}")
        except sr.UnknownValueError:
            st.warning("Didn't catch that.")
        #except sr.WaitTimeoutError:
         #   st.warning("Listening timed out.")
        except Exception as e:
            st.error(f"Error: {e}")
    #time.sleep(0.5)

# Display full transcript
st.subheader("📝 Transcript:")
st.write(st.session_state.transcript)

# Segment sentences from transcript
sentences = segment_sentences(st.session_state.transcript, 100)
st.subheader("🧩 Segmented Sentences:")
st.write(sentences)

coherence_score(sentences)

st.write('high coherence score indicates normal cognitive functionality')
st.write('coherence score more than 0.3 can be considered as normal. enter more sentences to increase the score')
# Optional: reset button
if st.button("🔄 Reset Transcript"):
    st.session_state.transcript = ""

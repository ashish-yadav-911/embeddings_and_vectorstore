from langchain_core.documents import Document #to prepare a document for RAG

from langchain_chroma import Chroma            # chromadb is a vector store(database)
from langchain_openai import OpenAIEmbeddings  # langchain X embeddings
from openai import OpenAI                      # open ai

from langchain_core.runnables import RunnableLambda

from dotenv import load_dotenv
import os

load_dotenv()

# using openai embedding model
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

#preparing the document
documents = [
    Document(
        page_content="""Post-Traumatic Stress Disorder (PTSD) is a mental health condition that can develop after an individual has experienced or witnessed a traumatic event, such as a natural disaster, combat, a serious accident, or physical or sexual assault. People with PTSD often experience flashbacks, nightmares, and intense anxiety when reminded of the trauma, leading them to avoid places, activities, or people that might trigger these memories. They may also have difficulty sleeping, feel emotionally numb, or struggle with concentration. PTSD can profoundly affect daily life and relationships, but with appropriate treatment, including therapy (like Cognitive Behavioral Therapy or EMDR), medication, and support from loved ones, many people can manage their symptoms and work toward recovery. Early intervention and a supportive environment are key in helping individuals with PTSD heal and regain a sense of control over their lives.""",
        metadata={"source": "PTSD"},
    ),
    Document(
        page_content="""Attention Deficit Hyperactivity Disorder (ADHD) is a neurodevelopmental condition that affects both children and adults, characterized by symptoms of inattention, hyperactivity, and impulsivity. People with ADHD often find it challenging to focus on tasks, organize activities, and follow through on instructions, which can impact academic, professional, and personal aspects of life. While the exact cause of ADHD is not fully understood, research suggests that genetic, environmental, and neurological factors contribute to its development. ADHD symptoms can vary widely, from mild to severe, and are often managed through a combination of treatments including behavioral therapy, medication, and lifestyle modifications. Early diagnosis and support can make a significant difference, helping individuals with ADHD harness their unique strengths and develop strategies to cope with daily challenges.""",
        metadata={"source": "Attention Disorder"},
    ),
    Document(
        page_content="""Down syndrome is a genetic condition caused by the presence of an extra copy of chromosome 21, which leads to developmental and physical changes. People with Down syndrome often have distinctive physical features, such as almond-shaped eyes and a single deep crease across the center of the palm, as well as varying degrees of intellectual disability. However, each individual with Down syndrome is unique, with abilities and challenges that differ widely. Advances in healthcare, early intervention programs, and inclusive educational practices have significantly improved quality of life and opportunities for people with Down syndrome. With support, individuals with Down syndrome can lead fulfilling lives, participate actively in their communities, and achieve personal goals.""",
        metadata={"source": "Down Syndrome"},
    ),
    Document(
        page_content="""Anxiety is a common mental health condition characterized by feelings of worry, fear, or unease that can range from mild to severe. While occasional anxiety is a normal response to stress, anxiety disorders involve persistent and often overwhelming feelings that can interfere with daily life. Symptoms can include physical sensations like a racing heart, sweating, or muscle tension, as well as mental symptoms such as excessive worry, restlessness, and difficulty concentrating. Anxiety can be triggered by specific situations or can appear without a clear cause, and it often coexists with other mental health conditions, like depression. Fortunately, anxiety is highly treatable through various approaches, including cognitive-behavioral therapy (CBT), medication, mindfulness, and lifestyle changes. With appropriate treatment and support, individuals with anxiety can learn effective strategies to manage their symptoms and improve their quality of life.""",
        metadata={"source": "Anxiety"},
    ),
    Document(
        page_content="""Bipolar disorder is a mental health condition characterized by extreme mood swings, including emotional highs known as mania or hypomania, and lows, which present as depression. During manic episodes, individuals may feel intensely energetic, overly euphoric, or unusually irritable, often leading to impulsive decision-making, decreased need for sleep, and sometimes risky behaviors. In contrast, depressive episodes bring deep sadness, loss of energy, feelings of worthlessness, and difficulty with daily functioning. The intensity and duration of these mood swings vary from person to person and can significantly disrupt relationships, work, and quality of life. Although the exact causes of bipolar disorder are not fully understood, genetic, biological, and environmental factors are thought to play a role. Treatments, including medication, psychotherapy, and lifestyle adjustments, can help manage symptoms, allowing many individuals with bipolar disorder to lead stable, fulfilling lives.""",
        metadata={"source": "Bipolar Disorder"},
    ),
        Document(
        page_content="""Train run on raods and are very strongly designed to be able to work and lift an dmove other stuck vecihles.""",
        metadata={"source": "Trains"},
    )
]

#temperorary vectorstore of document's embeddings
vectorstore = Chroma.from_documents(
    documents,
    embedding=OpenAIEmbeddings(),
)

#running cosine similarity search and then printing score to test results
ss_train=vectorstore.similarity_search("A train Running on Road")
sws_train=vectorstore.similarity_search_with_score("A train Running on Road")

ss=vectorstore.similarity_search("My child has Downsyndrome")
sws_syndrome=vectorstore.similarity_search_with_score("My Child has Down Syndrome")


#comparasion between two sentences from Document
print("About Train: ",sws_train)
print("____NEXT____")
print("About Syndrome",sws_syndrome)

# lower the similarity search score, higher the relevance is
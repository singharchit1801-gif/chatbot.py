"""
CodeAlpha - Task 2: Chatbot for FAQs (Beginner Version)
------------------------------------------------------------------
A simple chatbot that answers questions by finding the FAQ question that
is most SIMILAR to what the user typed, and giving that FAQ's answer.

How it decides "most similar":
    1. It converts every question into numbers using TF-IDF
       (this basically counts how important each word is).
    2. It compares the user's question to every FAQ question using
       cosine similarity (a way to measure how "close" two pieces
       of text are).
    3. It picks the FAQ with the highest similarity score.

Before running, install these libraries:
    pip install scikit-learn

Run it with:
    python faq_chatbot_simple.py
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Step 1: Our list of FAQs (question + answer pairs)
# You can add as many as you want, or load them from a file.
faq_questions = [
    "What is CodeAlpha?",
    "How do I apply for the internship?",
    "How many tasks do I need to complete?",
    "Where do I submit my completed tasks?",
    "Do I get a certificate after the internship?",
    "Is the internship paid?",
    "How can I contact CodeAlpha for support?",
]

faq_answers = [
    "CodeAlpha is a software development company that offers internships in AI, web development, and more.",
    "You can apply through the official CodeAlpha website or the link shared in your onboarding email.",
    "You need to complete a minimum of two or three tasks to be eligible for the certificate.",
    "Submit your completed tasks using the Submission Form shared in your WhatsApp group.",
    "Yes, you receive a QR-verified Completion Certificate after finishing the required tasks.",
    "This internship is unpaid, but it gives you a certificate, recommendation letter, and resume support.",
    "You can email services@codealpha.tech or reach out on WhatsApp.",
]


def build_vectorizer(questions):
    """
    This turns our list of FAQ questions into numbers (vectors)
    so the computer can compare them mathematically.
    """
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)
    return vectorizer, question_vectors


def get_best_answer(user_question, vectorizer, question_vectors):
    """
    Compares the user's question to all FAQ questions and
    returns the answer for the closest match.
    """
    # Turn the user's question into the same kind of number vector
    user_vector = vectorizer.transform([user_question])

    # Compare the user's question with every FAQ question
    similarity_scores = cosine_similarity(user_vector, question_vectors)

    # Find the index (position) of the highest score
    best_match_index = similarity_scores.argmax()
    best_score = similarity_scores[0][best_match_index]

    # If the best score is too low, we admit we don't know the answer
    if best_score < 0.2:
        return "Sorry, I don't have an answer for that. Try asking something else!"

    return faq_answers[best_match_index]


def main():
    print("===================================")
    print("      Simple FAQ Chatbot")
    print("===================================")
    print("Ask me a question about the CodeAlpha internship.")
    print("Type 'exit' to quit.\n")

    # Build the vectorizer once, before the chat starts
    vectorizer, question_vectors = build_vectorizer(faq_questions)

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("Bot: Goodbye!")
            break

        answer = get_best_answer(user_input, vectorizer, question_vectors)
        print("Bot:", answer)
        print()


if __name__ == "__main__":
    main()

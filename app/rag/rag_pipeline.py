from app.retrieval.retriever import (
    search_documents
)

from app.llm.generator import (
    generate_answer
)

from app.retrieval.hybrid_retriever import (
    hybrid_search
)

def ask(question):

    # we could have used search_documents if do not want to use hybrid search. But hybrid search gives better results by combining vector and keyword search.
    documents = hybrid_search(
        question
    )
    if len(documents) == 0:

        return {
            "answer":
            "I couldn't find relevant information.",
            "sources":[]
        }

    context = "\n\n".join(
        [
            doc["text"]
            for doc in documents
        ]
    )

    answer = generate_answer(
        question,
        context
    )

    sources = list(
        set(
            [
                doc["source"]
                for doc in documents
            ]
        )
    )

    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":

    while True:

        question = input(
            "\nQuestion: "
        )

        if question.lower() == "exit":
            break

        result = ask(
            question
        )

        print("\nAnswer:\n")

        print(
            result["answer"]
        )

        print("\nSources:")

        for source in result[
            "sources"
        ]:

            print(source)
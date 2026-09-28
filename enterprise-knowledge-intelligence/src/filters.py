def filter_documents(documents, filters):

    if not filters:
        return documents

    filtered = []

    for document in documents:

        matches = True

        for key, value in filters.items():

            if document.metadata.get(key) != value:
                matches = False
                break

        if matches:
            filtered.append(document)

    return filtered
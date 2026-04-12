def add_item(item, box=None):
    if box is None:
        box = []  # Create a new list for this specific call
    box.append(item)
    return box
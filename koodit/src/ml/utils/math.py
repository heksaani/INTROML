def euclidean_distance(p, q) -> float:
    # IMPLEMENT This should return the Euclidean distance between two points.
    squares = [(p-q) ** 2 for p, q in zip(p, q)]
    return sum(squares) ** 0.5


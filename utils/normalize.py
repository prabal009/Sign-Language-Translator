import math

def normalize_hand(hand):
    """
    Normalize one hand (21 landmarks).

    Input:
        hand = [x1,y1,z1,x2,y2,z2,...]

    Output:
        normalized coordinates
    """

    if not any(hand):
        return hand

    wrist_x = hand[0]
    wrist_y = hand[1]
    wrist_z = hand[2]

    normalized = []

    # Move wrist to origin
    for i in range(0, len(hand), 3):

        x = hand[i] - wrist_x
        y = hand[i + 1] - wrist_y
        z = hand[i + 2] - wrist_z

        normalized.extend([x, y, z])

    # Scale using the farthest landmark
    max_distance = 0

    for i in range(0, len(normalized), 3):

        x = normalized[i]
        y = normalized[i + 1]
        z = normalized[i + 2]

        distance = math.sqrt(x*x + y*y + z*z)

        if distance > max_distance:
            max_distance = distance

    if max_distance == 0:
        return normalized

    scaled = []

    for value in normalized:
        scaled.append(value / max_distance)

    return scaled
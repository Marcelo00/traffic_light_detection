WIDTH_RGB = 1280
HEIGHT_RGB = 720

WIDTH_RIIB = 1280
HEIGHT_RIIB = 736

NUM_CLASSES = 4

SIMPLIFIED_CLASSES = {
    'Green': 'Green',
    'GreenLeft': 'Green',
    'GreenRight': 'Green',
    'GreenStraight': 'Green',
    'GreenStraightRight': 'Green',
    'GreenStraightLeft': 'Green',
    'Yellow': 'Yellow',
    'Red': 'Red',
    'RedLeft': 'Red',
    'RedRight': 'Red',
    'RedStraight': 'Red',
    'RedStraightLeft': 'Red',
    'Off': 'Off',
    'off': 'Off',
}

CLASS_COLORS = {
    1: (255, 255, 255),
    2: (0, 255, 0),
    3: (0, 255, 255),
    4: (0, 0, 255),
}

CLASSES_TO_ID = {
    'Off': 0,
    'Green': 1,
    'Yellow': 2,
    'Red': 3,
}
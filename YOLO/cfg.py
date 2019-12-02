
IMG_HEIGHT = 480
IMG_WIDTH = 640

CLASS_NUM = 3

ANCHORS_GROUP = {
    32:[[730,145],[288,190],[216,540]],
    16:[[365,73],[144,95],[108,270]],
    8:[[180,37],[72,44],[54,135]]
}

ANCHORS_GROUP_AREA = {
    32: [x * y for x, y in ANCHORS_GROUP[32]],
    16: [x * y for x, y in ANCHORS_GROUP[16]],
    8: [x * y for x, y in ANCHORS_GROUP[8]],
}

color=['#ff0000','#0000ff','#00ff00']


